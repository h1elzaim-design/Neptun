"""Walk-Forward-Validation — robuster Test gegen Overfitting.

Nutzt `split_walk_forward` aus evaluation.py um die Datenreihen zu schneiden.
Pro Fold wird ein In-Sample-Sweep durchgeführt, die besten Parameter werden
gewählt und Out-of-Sample getestet.
"""

from __future__ import annotations

import logging
import math
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
        # Ein Fold schneidet die Zeit, nicht das Universum: welche Symbole im
        # Lake fehlten, gilt für jeden Fold gleich (#307). Das Fenster dagegen
        # ist hier per Konstruktion vollständig — `start`/`end` sind die
        # Grenzen des Schnitts selbst.
        missing_symbols=list(md.missing_symbols),
        frame=sliced_frame,
    )


def _bars(werte: Any) -> list[int]:
    """Die Bar-Zahlen aus einem Parameterwert — Floats **aufgerundet**.

    Aufgerundet statt verworfen: ein Lookback von 20,5 Bars liest bis in den
    21. zurück. Die alte Fassung nahm nur ganze Zahlen, also kam
    ``halflife: 20.5`` als *nichts* an — und ein Embargo von 0 heisst, dass
    der OOS-Rand aus Train-Preisen rechnet.

    ``bool`` ist in Python ein ``int``; ohne den Ausschluss wäre
    ``use_log: [True, False]`` ein Embargo von einem Bar.
    """
    aus: list[int] = []
    for x in werte if isinstance(werte, (list, tuple, np.ndarray)) else [werte]:
        # `bool` ist ein `int`, `np.bool_` ist es nicht — beide raus.
        if isinstance(x, (bool, np.bool_)):
            continue
        # **Nicht auf `isinstance(x, int)` prüfen.** `np.int64` ist kein
        # Python-`int`, `np.float64` dagegen erbt von `float`: eine
        # Typprüfung liesse ein numpy-Grid stillschweigend durchfallen und
        # ein numpy-Float durch. Ein Grid aus `np.arange(...)` ergab so
        # Embargo 0 — mit dem Stempel „deklariert".
        try:
            wert = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(wert):
            aus.append(math.ceil(wert))
    return aus


def _raten(raum: dict[str, Any]) -> int:
    """Der alte Weg: die grösste **ganze** Zahl im Grid.

    Unverändert, damit bestehende Ergebnisse reproduzierbar bleiben. Das
    Aufrunden von Floats gilt nur dort, wo jemand den Parameter ausdrücklich
    als Lookback benannt hat — hier wäre es geraten auf geraten.
    """
    vals: list[int] = []
    for v in raum.values():
        for x in v if isinstance(v, (list, tuple, np.ndarray)) else [v]:
            if isinstance(x, (bool, np.bool_)):
                continue
            try:
                wert = float(x)
            except (TypeError, ValueError):
                continue
            if math.isfinite(wert) and wert.is_integer():
                vals.append(int(wert))
    return max(vals) if vals else 0


def _infer_embargo(spec: StrategySpec) -> tuple[int | None, str]:
    """Wieviele Bars zwischen Train-Ende und OOS-Start — und woher die Zahl kommt.

    Ein OOS-Fenster darf einem Indikator mit Lookback ``L`` nicht erlauben,
    Preise aus dem Train-Fenster zu lesen. Das Embargo ist deshalb der längste
    Lookback, den irgendeine gesweepte Konfiguration benutzt.

    Drei Fälle, und die Unterscheidung ist der ganze Punkt:

    ``lookback_keys is None``
        Nicht deklariert → geraten, mit Warnung.
    ``lookback_keys == ()``
        Ausdrücklich „blickt nicht zurück" (``buy_and_hold``) → Embargo 0,
        **ohne** Warnung. Das ist eine Antwort, kein Fehlen.
    ``lookback_keys == (…)``
        Deklariert. Gefunden wird über ``param_space`` **und** ``params``:
        ein fester Lookback blickt genauso weit zurück wie ein gesweepter,
        und ein leerer Grid-Eintrag darf den festen Wert nicht überschatten.

    **Und wenn eine Deklaration nichts hergibt, wird geraten — nicht null
    behauptet.** Ein vertippter Schlüssel, ein leeres Grid, ein Grid aus
    numpy-Typen: all das ergab in der ersten Fassung ``(0, "deklariert")``,
    also *auf nichts geraten und als belegt ausgewiesen*. Das war strikt
    schlechter als der alte Rateweg, der wenigstens sagte, dass er rät.

    Gibt ``(bars, quelle)`` zurück. ``bars`` ist ``None``, wenn sich keine
    Zahl bestimmen liess — **nicht** 0: „unbekannt" und „braucht keins" sind
    verschiedene Aussagen, und die Verwechslung der beiden ist in diesem
    Projekt schon einmal als ``realism 0.00`` aufgeschlagen.
    """
    raum = spec.param_space or {}
    fest = spec.params or {}

    if spec.lookback_keys == ():
        return 0, "kein_lookback"

    if spec.lookback_keys:
        vals: list[int] = []
        for schluessel in spec.lookback_keys:
            # **Beide Quellen, kein `elif`.** Ein Schlüssel kann im Grid
            # stehen und dort leer sein (`{"w": []}`), während `params` den
            # echten Wert trägt.
            if schluessel in raum:
                vals += _bars(raum[schluessel])
            if schluessel in fest:
                vals += _bars(fest[schluessel])
        if vals:
            return max(vals), "deklariert"

        geraten = _raten(raum)
        log.warning(
            "%s: `lookback_keys=%s` ergab keinen einzigen Wert — Tippfehler, "
            "leeres Grid oder ein Typ, den `_bars` nicht liest? Es wird "
            "geraten (%d Bars) statt 0 zu behaupten; die Deklaration greift "
            "hier nicht (#320).",
            spec.strategy_id,
            tuple(spec.lookback_keys),
            geraten,
        )
        return (geraten or None), "geraten"

    geraten = _raten(raum)
    if geraten:
        log.warning(
            "%s: Embargo %d Bars **geraten** (grösste Zahl im param_space) — "
            "die Spec deklariert keine `lookback_keys`. Steht dort ein "
            "Parameter, der nicht zurückblickt (n_positions, capital), fällt "
            "der Wert aus jedem Fold heraus und Folds verschwinden still (#320).",
            spec.strategy_id,
            geraten,
        )
        return geraten, "geraten"

    # **Keine Zahl bestimmbar.** `kalman_trend` (`delta: 1e-4`) ist der Fall.
    # `None` und nicht 0: der OOS-Rand ist ungeschützt, und das ist etwas
    # anderes als „braucht keinen Schutz".
    log.warning(
        "%s: Embargo **unbestimmbar** — die Spec deklariert keine "
        "`lookback_keys`, und im param_space steht keine ganze Zahl, aus der "
        "sich eine ableiten liesse. Gerechnet wird ohne Embargo, der OOS-Rand "
        "ist damit ungeschützt (#320).",
        spec.strategy_id,
    )
    return None, "geraten"


def walk_forward(
    spec: StrategySpec,
    data: MarketData,
    config: BacktestConfig | None = None,
    n_folds: int = 4,
    train_ratio: float = 0.6,
    rank_by: str = "sharpe",
    embargo: int | None = None,
    max_workers: int | None = None,
) -> WalkForwardResult:
    """Führt eine Walk-Forward-Validation über die übergebenen Daten aus.

    Args:
        spec: StrategySpec mit param_space für In-Sample-Sweep.
        data: Gesamtdatensatz.
        config: Backtest-Config.
        n_folds: Anzahl der Walk-Forward-Epochen.
        train_ratio: Anteil der In-Sample-Daten pro Fold.
        rank_by: Kriterium zur Auswahl der besten In-Sample-Parameter.
        embargo: Bars zwischen Train-Ende und OOS-Start. ``None`` → abgeleitet
            aus ``spec.lookback_keys``, ersatzweise geraten (`_infer_embargo`);
            das Ergebnis trägt Zahl und Herkunft in ``embargo`` /
            ``embargo_source``, damit Geratenes nicht wie Belegtes aussieht.
            Der **erste** Fold entfällt immer (kein Train davor), darüber
            hinaus werden degenerierte Folds übersprungen —
            ``len(result.folds)`` kann also < ``n_folds - 1`` sein.
        max_workers: Wird an den **inneren** Sweep pro Fold durchgereicht
            (#210). Die Folds selbst laufen weiter nacheinander — sie sind
            wenige (typisch 4–6), das Grid ist die große Zahl, und geschachtelte
            Pools würden die Kerne doppelt vergeben. ``1`` = alles seriell.

    Returns:
        WalkForwardResult mit aggregierten In-Sample- und Out-of-Sample-Metriken.
    """
    config = config or BacktestConfig()
    if embargo is None:
        embargo_bars, embargo_quelle = _infer_embargo(spec)
    else:
        embargo_bars, embargo_quelle = embargo, "vorgegeben"
    # `None` heisst „unbestimmbar" und wird hier zu 0 **gerechnet**, aber nicht
    # zu 0 **gespeichert**: der Splitter braucht eine Zahl, das Ergebnis soll
    # den Unterschied zwischen „kein Embargo nötig" und „keins bestimmbar"
    # behalten.
    embargo = 0 if embargo_bars is None else embargo_bars
    splits = split_walk_forward(
        data.frame.index, n_folds=n_folds, train_ratio=train_ratio, embargo=embargo
    )
    # **Die Basislinie ist `n_folds - 1`, nicht `n_folds`.**
    # `split_walk_forward` beginnt bei `k=0` mit `test_start_i = 0` — davor
    # liegt kein Train, also entfällt der erste Fold **immer**, unabhängig vom
    # Embargo (gemessen: 4 angefordert → 3 Splits, auch bei Embargo 0). Gegen
    # `n_folds` verglichen feuerte diese Warnung bei jedem einzelnen Lauf und
    # schob es aufs Embargo; ein Signal, das immer angeht, ist keins.
    if len(splits) < n_folds - 1:
        # **Nicht nur ins Log.** Dass degenerierte Folds übersprungen werden,
        # ist richtig (der alte Bug war ein Ein-Bar-Train). Dass es niemandem
        # auffällt, ist es nicht: eine Validierung über zwei statt sechs Folds
        # ist eine andere Aussage, nicht dieselbe mit weniger Zeilen. Die Zahl
        # landet unten im Ergebnis, hier steht der Grund (#320).
        log.warning(
            "Nur %d von %d angeforderten Folds sind rechenbar — Embargo %d Bars "
            "(%s) frisst das Fenster. Bei %d Bars Historie bleibt je Fold zu "
            "wenig Train übrig.",
            len(splits),
            n_folds,
            embargo,
            embargo_quelle,
            len(data.frame.index),
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
        sweep_res = sweep(
            spec,
            train_data,
            config=config,
            rank_by=rank_by,
            selection_stats=False,
            max_workers=max_workers,
        )

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
        # Was der Splitter überhaupt liefern kann — der erste Fold hat per
        # Konstruktion kein Train davor. `n_folds` roh zu speichern hiesse,
        # eine Differenz zu behaupten, die es immer gibt.
        n_folds_expected=max(n_folds - 1, 0),
        embargo=embargo_bars,
        embargo_source=embargo_quelle,
        rank_by=rank_by,
        folds=folds,
        coverage=data.coverage,
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
