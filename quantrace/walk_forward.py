"""Walk-Forward-Validation — robuster Test gegen Overfitting.

Nutzt `split_walk_forward` aus evaluation.py um die Datenreihen zu schneiden.
Pro Fold wird ein In-Sample-Sweep durchgeführt, die besten Parameter werden
gewählt und Out-of-Sample getestet.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from quantrace.backtest_runner import run_backtest
from quantrace.evaluation import split_walk_forward
from quantrace.models import (
    BacktestConfig,
    FoldResult,
    MarketData,
    StrategySpec,
    WalkForwardResult,
)
from quantrace.stats import (
    DEFAULT_FDR_ALPHA,
    annualised_sharpe,
    benjamini_hochberg,
    bootstrap_sharpe_ci,
    sharpe_p_value,
)
from quantrace.sweep import _run_return_stats, _run_returns, sweep

log = logging.getLogger(__name__)


def _slice_market_data(md: MarketData, start: pd.Timestamp, end: pd.Timestamp) -> MarketData:
    """Schneidet das MarketData-Objekt auf einen Zeitraum zu."""
    sliced_frame = md.frame.loc[start:end]
    if sliced_frame.empty:
        raise ValueError(f"Slice {start.date()}..{end.date()} ist leer.")

    # Update properties based on the actual slice
    actual_start = sliced_frame.index[0].date()
    actual_end = sliced_frame.index[-1].date()

    return MarketData(
        universe=md.universe,
        symbols=md.symbols,
        timeframe=md.timeframe,
        provider=md.provider,
        start=actual_start,
        end=actual_end,
        adjusted=md.adjusted,
        frame=sliced_frame,
    )


def _infer_embargo(spec: StrategySpec) -> int:
    """Longest integer lookback under test = the leakage horizon to embargo.

    A walk-forward OOS window must not let an indicator with lookback L read
    prices from the train window. The longest L any swept config uses is the
    largest integer in the param space (windows dwarf threshold params like
    entry_z), so embargoing that many bars makes the boundary leak-safe.
    Returns 0 when the param space holds no integer-valued knobs.
    """
    vals: list[int] = []
    for v in (spec.param_space or {}).values():
        candidates = v if isinstance(v, (list, tuple)) else [v]
        for x in candidates:
            if isinstance(x, bool):
                continue
            if isinstance(x, int):
                vals.append(x)
            elif isinstance(x, float) and x.is_integer():
                vals.append(int(x))
    return max(vals) if vals else 0


def walk_forward(
    spec: StrategySpec,
    data: MarketData,
    config: BacktestConfig | None = None,
    n_folds: int = 4,
    train_ratio: float = 0.6,
    rank_by: str = "sharpe",
    embargo: int | None = None,
) -> WalkForwardResult:
    """Führt eine Walk-Forward-Validation über die übergebenen Daten aus.

    Args:
        spec: StrategySpec mit param_space für In-Sample-Sweep.
        data: Gesamtdatensatz.
        config: Backtest-Config.
        n_folds: Anzahl der Walk-Forward-Epochen.
        train_ratio: Anteil der In-Sample-Daten pro Fold.
        rank_by: Kriterium zur Auswahl der besten In-Sample-Parameter.
        embargo: Bars zwischen Train-Ende und OOS-Start. ``None`` → aus dem
            param_space abgeleitet (längster Lookback), damit der OOS-Rand
            leak-frei ist. Degenerierte Folds (zu kurzes Train) werden
            übersprungen, ``len(result.folds)`` kann also < ``n_folds`` sein.

    Returns:
        WalkForwardResult mit aggregierten In-Sample- und Out-of-Sample-Metriken.
    """
    config = config or BacktestConfig()
    if embargo is None:
        embargo = _infer_embargo(spec)
    splits = split_walk_forward(
        data.frame.index, n_folds=n_folds, train_ratio=train_ratio, embargo=embargo
    )

    folds: list[FoldResult] = []
    # OOS-Return-Pfade in Fold-Reihenfolge (= chronologisch, Test-Fenster
    # sind disjunkt) — Grundlage der gestitchten OOS-Inferenz unten.
    stitched_oos_returns: list[np.ndarray] = []
    # Die zugehörigen Equity-Kurven (pd.Series mit DatetimeIndex) für den
    # persistierten gestitchten OOS-Pfad.
    stitched_oos_curves: list[pd.Series] = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(splits, 1):
        log.info(
            "Fold %d/%d: Train %s..%s, Embargo %d, Test %s..%s",
            i,
            len(splits),
            train_start.date(),
            train_end.date(),
            embargo,
            test_start.date(),
            test_end.date(),
        )

        # train_end liegt bereits `embargo`+1 Bars vor test_start (siehe
        # split_walk_forward) — Slices sind disjunkt mit Embargo-Lücke dazwischen.
        try:
            train_data = _slice_market_data(data, train_start, train_end)
            test_data = _slice_market_data(data, test_start, test_end)
        except ValueError as e:
            log.warning("Fold %d übersprungen: %s", i, e)
            continue

        # 2. In-Sample Sweep — ohne DSR/FDR-Statistik: die gilt der Selektion
        # innerhalb des Folds und wird hier verworfen (nur beste Params zählen);
        # die OOS-Signifikanz wird unten über die Folds gerechnet.
        log.info("  IS Sweep Fold %d...", i)
        sweep_res = sweep(spec, train_data, config=config, rank_by=rank_by, selection_stats=False)

        if not sweep_res.best_run:
            log.warning("Fold %d: Sweep hat keine Ergebnisse geliefert.", i)
            continue

        chosen_params = sweep_res.best_params
        train_result = sweep_res.best_run.result

        # 3. Out-of-Sample Backtest mit gewählten Parametern
        log.info("  OOS Test Fold %d mit %s", i, chosen_params)
        # Merge wie in sweep(): Basis-Params (z.B. `graph`) überleben, nur die
        # im IS-Sweep gewählten Grid-Keys variieren.
        oos_spec = spec.model_copy(update={"params": {**spec.params, **chosen_params}})
        test_result = run_backtest(oos_spec, test_data, config=config)

        # 4. OOS-Signifikanz: p-Wert für H0 "true SR ≤ 0" aus den echten
        # Test-Returns (Mertens-SE mit Fold-eigenen Skew/Kurtosis-Momenten).
        # Der Return-Pfad selbst wird für die gestitchte OOS-Inferenz gesammelt.
        oos_rets = _run_returns(test_result)
        if oos_rets is not None:
            stitched_oos_returns.append(oos_rets)
            if test_result.equity_curve is not None:
                stitched_oos_curves.append(test_result.equity_curve)
        oos_stats = _run_return_stats(test_result)
        test_p_value = None
        test_n_obs = None
        if oos_stats is not None:
            test_n_obs = oos_stats.n_obs
            test_p_value = sharpe_p_value(
                sharpe_period=oos_stats.sr_period,
                n_obs=oos_stats.n_obs,
                skew=oos_stats.skew,
                kurt=oos_stats.kurt,
            )

        # 5. Resultat festhalten
        folds.append(
            FoldResult(
                fold_index=i,
                train_start=train_start.date(),
                train_end=train_end.date(),
                test_start=test_data.start,  # erster echter Out-of-Sample-Bar
                test_end=test_data.end,
                chosen_params=chosen_params,
                train_sharpe=train_result.sharpe,
                train_cagr=train_result.cagr,
                test_sharpe=test_result.sharpe,
                test_cagr=test_result.cagr,
                test_max_drawdown=test_result.max_drawdown,
                test_n_trades=test_result.trades.n_trades,
                test_turnover_annual=test_result.turnover_annual,
                test_n_obs=test_n_obs,
                test_p_value=test_p_value,
            )
        )

    if not folds:
        raise ValueError("Kein Fold konnte erfolgreich evaluiert werden.")

    # FDR über die *testbaren* OOS-Folds: kontrolliert, wie viele
    # "signifikante" Folds bei m Tests als Zufallstreffer erwartbar wären.
    # Folds ohne p-Wert (degenerierte Equity) sind keine Tests — sie in die
    # Familie zu zählen würde nur die Schwellen der echten Folds verschärfen;
    # sie bleiben als n_untested sichtbar.
    fdr_summary = None
    tested = [f for f in folds if f.test_p_value is not None]
    if len(tested) >= 2:
        fdr_res = benjamini_hochberg(
            [f.test_p_value for f in tested], alpha=DEFAULT_FDR_ALPHA
        )
        for f, q, sig in zip(tested, fdr_res.q_values, fdr_res.significant, strict=True):
            f.test_q_value = q
            f.fdr_significant = sig
        fdr_summary = {
            "method": "benjamini_hochberg",
            "alpha": fdr_res.alpha,
            "n_tests": fdr_res.n_tests,
            "n_untested": len(folds) - len(tested),
            "n_significant": fdr_res.n_significant,
            "all_significant": fdr_res.n_significant == fdr_res.n_tests,
            "scope": "oos_folds",
        }

    # Gestitchte OOS-Inferenz: die konkatenierten Test-Fenster sind die
    # Rendite-Reihe, die ein Live-Deployment der Fold-Winner tatsächlich
    # erlebt hätte. Fold-Mittelwerte gewichten kurze Folds über; der
    # gestitchte Sharpe + Bootstrap-KI beantworten die eigentliche Frage:
    # war der ganze OOS-Pfad von 0 unterscheidbar?
    # Gestitchter OOS-Equity-Pfad: Fold-Kurven kettennormiert aneinander —
    # jede Kurve startet dort, wo die vorige endete. Persistiert als
    # [{date, value}], damit Downstream (Uniqueness/Attribution/Bootstrap)
    # den Deployment-Pfad hat, nicht nur seine Summary.
    oos_equity = None
    if stitched_oos_curves:
        points: list[dict[str, Any]] = []
        level = 1.0
        for curve in stitched_oos_curves:
            vals = curve.to_numpy(dtype=float)
            first = vals[0] if vals.size and np.isfinite(vals[0]) and vals[0] != 0 else None
            if first is None:
                continue
            scaled = vals / first * level
            points.extend(
                {"date": ts.date().isoformat(), "value": float(v)}
                for ts, v in zip(curve.index, scaled, strict=True)
                if np.isfinite(v)
            )
            level = float(scaled[-1])
        if len(points) >= 8:
            oos_equity = points

    oos_inference = None
    if stitched_oos_returns:
        stitched = np.concatenate(stitched_oos_returns)
        if stitched.size >= 8:
            try:
                boot = bootstrap_sharpe_ci(stitched)
                oos_inference = {
                    "sharpe_annual": annualised_sharpe(stitched),
                    "ci_low": boot.ci_low,
                    "ci_high": boot.ci_high,
                    "confidence": boot.confidence,
                    "p_value": boot.p_value,
                    "n_obs": int(stitched.size),
                    "n_folds_stitched": len(stitched_oos_returns),
                    "method": "stitched_oos_stationary_bootstrap",
                }
            except ValueError as exc:
                log.warning("Gestitchte OOS-Inferenz übersprungen: %s", exc)

    # Aggregation
    is_sharpe_mean = sum(f.train_sharpe for f in folds) / len(folds)
    oos_sharpe_mean = sum(f.test_sharpe for f in folds) / len(folds)

    degradation = 0.0
    if is_sharpe_mean > 0:
        degradation = max(0.0, oos_sharpe_mean / is_sharpe_mean)

    return WalkForwardResult(
        strategy_id=spec.strategy_id,
        periods_per_year=float(data.periods_per_year),
        n_folds=len(folds),  # tatsächlich evaluierte Folds (degenerierte übersprungen)
        rank_by=rank_by,
        folds=folds,
        is_sharpe_mean=is_sharpe_mean,
        oos_sharpe_mean=oos_sharpe_mean,
        degradation=degradation,
        fdr=fdr_summary,
        oos_inference=oos_inference,
        oos_equity=oos_equity,
        # Ein Walk-Forward rechnet jeden Fold mit denselben Annahmen — die
        # Config gehört deshalb einmal an die Spitze, nicht pro Fold.
        config=config,
    )
