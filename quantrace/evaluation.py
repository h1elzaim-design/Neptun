"""Evaluation Agent — Score über sechs Dimensionen + Guardrails.

Wichtig: Score-Schwellen und Gewichte stehen in config/research_rules/governance.yaml.
Diese Datei ist menschlich verwaltet. Eine Score-Änderung ist eine
Governance-Entscheidung, kein Implementierungsdetail.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quantrace.models import BacktestResult, EvaluationReport, StrategySpec, WalkForwardResult

log = logging.getLogger(__name__)

DEFAULT_GOVERNANCE = Path("config/research_rules/governance.yaml")


def load_governance(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_GOVERNANCE
    if not p.exists():
        raise FileNotFoundError(f"Governance-Config fehlt: {p}")
    return yaml.safe_load(p.read_text())


def evaluate(
    spec: StrategySpec,
    backtests: list[BacktestResult] | WalkForwardResult,
    governance: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Bewertet ein Strategie-Set (in-sample + ggf. OOS / Walk-Forward) gesamt."""
    gov = governance or load_governance()

    is_wf = isinstance(backtests, WalkForwardResult)
    if not is_wf and not backtests:
        raise ValueError("Mindestens ein Backtest erforderlich")

    if is_wf:
        primary_sharpe = backtests.oos_sharpe_mean
        primary_cagr = sum(f.test_cagr for f in backtests.folds) / max(1, len(backtests.folds))
        primary_max_dd = min(
            f.test_max_drawdown for f in backtests.folds
        )  # minimum is most negative
        primary_trades = sum(f.test_n_trades for f in backtests.folds)

        target_sharpe = gov.get("targets", {}).get("sharpe", 1.5)
        target_cagr = gov.get("targets", {}).get("cagr", 0.15)
        s_perf = _clip01(0.6 * primary_sharpe / target_sharpe + 0.4 * primary_cagr / target_cagr)

        tolerable_dd = abs(gov.get("targets", {}).get("max_drawdown", -0.25))
        dd_score = 1 - min(abs(primary_max_dd) / tolerable_dd, 1.5)
        s_risk = _clip01(dd_score)  # Simplified ulcer for WF

        sharpes = np.array([f.test_sharpe for f in backtests.folds])
        mean, std = sharpes.mean(), sharpes.std(ddof=0)
        cv = std / abs(mean) if abs(mean) > 1e-9 else 1.0
        s_stab = _clip01(1 - cv)

        s_real = 1.0  # Assume WF is realistic for now
        s_gen = _clip01(backtests.degradation)
        s_simp = _score_simplicity(spec, gov)

        bt_ids = [f"wf_fold_{f.fold_index}" for f in backtests.folds]
    else:
        primary = backtests[0]
        s_perf = _score_performance(primary, gov)
        s_risk = _score_risk(primary, gov)
        s_stab = _score_stability(backtests, gov)
        s_real = _score_realism(primary, gov)
        s_gen = _score_generalization(backtests, gov)
        s_simp = _score_simplicity(spec, gov)

        primary_sharpe = primary.sharpe
        primary_max_dd = primary.max_drawdown
        primary_trades = primary.trades.n_trades
        bt_ids = [f"{b.strategy_id}@{b.start}..{b.end}" for b in backtests]

    weights = gov.get("score_weights", {})
    components = {
        "performance": (s_perf, weights.get("performance", 0.25)),
        "risk": (s_risk, weights.get("risk", 0.20)),
        "stability": (s_stab, weights.get("stability", 0.15)),
        "realism": (s_real, weights.get("realism", 0.10)),
        "generalization": (s_gen, weights.get("generalization", 0.20)),
        "simplicity": (s_simp, weights.get("simplicity", 0.10)),
    }
    total = sum(v * w for v, w in components.values()) / sum(w for _, w in components.values())

    reasons: list[str] = []
    guardrails = gov.get("guardrails", {})
    if primary_max_dd < guardrails.get("max_drawdown_floor", -0.5):
        reasons.append(f"max_drawdown {primary_max_dd:.2%} unter Floor")
    if primary_sharpe < guardrails.get("min_sharpe", 0.5):
        reasons.append(f"sharpe {primary_sharpe:.2f} unter Minimum")
    if primary_trades < guardrails.get("min_trades", 30):
        reasons.append(f"n_trades {primary_trades} zu gering")
    if not is_wf and len(backtests) < 2 and guardrails.get("require_oos", True):
        reasons.append("Kein Out-of-sample Backtest geliefert")

    hint = "Mehr Folds testen." if is_wf else _suggest_variation(spec, backtests[0])

    return EvaluationReport(
        strategy_id=spec.strategy_id,
        backtest_ids=bt_ids,
        score_performance=float(s_perf),
        score_risk=float(s_risk),
        score_stability=float(s_stab),
        score_realism=float(s_real),
        score_generalization=float(s_gen),
        score_simplicity=float(s_simp),
        score_total=float(total),
        passed_guardrails=not reasons,
        rejection_reasons=reasons,
        next_variation_hint=hint,
    )


# -- Score components ---------------------------------------------------------


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _score_performance(bt: BacktestResult, gov: dict[str, Any]) -> float:
    target_sharpe = gov.get("targets", {}).get("sharpe", 1.5)
    target_cagr = gov.get("targets", {}).get("cagr", 0.15)
    return _clip01(0.6 * bt.sharpe / target_sharpe + 0.4 * bt.cagr / target_cagr)


def _score_risk(bt: BacktestResult, gov: dict[str, Any]) -> float:
    tolerable_dd = abs(gov.get("targets", {}).get("max_drawdown", -0.25))
    dd_score = 1 - min(abs(bt.max_drawdown) / tolerable_dd, 1.5)
    ulcer_score = 1 - min(bt.ulcer_index / 0.20, 1.5)
    return _clip01(0.6 * dd_score + 0.4 * ulcer_score)


def _score_stability(bts: list[BacktestResult], gov: dict[str, Any]) -> float:
    if len(bts) < 2:
        return 0.5
    sharpes = np.array([b.sharpe for b in bts])
    mean, std = sharpes.mean(), sharpes.std(ddof=0)
    cv = std / abs(mean) if abs(mean) > 1e-9 else 1.0
    return _clip01(1 - cv)


def _score_realism(bt: BacktestResult, gov: dict[str, Any]) -> float:
    min_fees_bps = gov.get("targets", {}).get("min_fees_bps", 2.0)
    fee_ok = bt.config.fees_bps >= min_fees_bps
    slip_ok = bt.config.slippage_bps >= gov.get("targets", {}).get("min_slippage_bps", 3.0)
    return _clip01(0.5 * float(fee_ok) + 0.5 * float(slip_ok))


def _score_generalization(bts: list[BacktestResult], gov: dict[str, Any]) -> float:
    if len(bts) < 2:
        return 0.3
    is_, oos = bts[0], bts[-1]
    if is_.sharpe <= 0:
        return 0.0
    ratio = oos.sharpe / is_.sharpe
    return _clip01(ratio)


def _score_simplicity(spec: StrategySpec, gov: dict[str, Any]) -> float:
    n_params = len(spec.params)
    cap = gov.get("targets", {}).get("max_params", 6)
    return _clip01(1 - n_params / cap)


def _suggest_variation(spec: StrategySpec, bt: BacktestResult) -> str | None:
    """Heuristik für den Research-Agenten: was als Nächstes probieren?"""
    if bt.sharpe < 0.5:
        return "Andere Strategieklasse oder andere Universum-Cuts versuchen."
    if abs(bt.max_drawdown) > 0.3:
        return "Volatility-Filter oder Position-Sizing nach ATR hinzufügen."
    if bt.trades.n_trades < 30:
        return "Schnellere Signale (kürzere Lookback-Fenster) für mehr Trades."
    if bt.trades.win_rate < 0.4 and bt.trades.profit_factor > 1.5:
        return "Trailing-Stop testen, um Winner laufen zu lassen."
    return "Walk-Forward auf 3 Sub-Perioden splitten und Stabilität messen."


def split_walk_forward(
    md_index: pd.DatetimeIndex,
    n_folds: int = 4,
    train_ratio: float = 0.6,
    embargo: int = 0,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Liefert (train_start, train_end, test_start, test_end)-Tupel für Walk-Forward.

    Disziplin gegen Leakage:

    * **Kein degenerierter Fold.** Das erste Test-Segment hat keine Historie
      davor; statt es auf einem 1-Bar-Train laufen zu lassen (der alte Bug),
      werden Folds ohne ausreichend langes Train-Fenster übersprungen. Die
      zurückgegebene Liste kann daher **weniger** als ``n_folds`` Einträge haben.
    * **Embargo.** Zwischen ``train_end`` und ``test_start`` liegt eine Lücke von
      ``embargo`` Bars (plus dem ohnehin nötigen 1-Bar-Abstand). So kann ein
      Indikator mit Lookback L am OOS-Rand nicht aus Train-Preisen berechnet
      werden — der subtile Boundary-Leak, den ein reiner ``.loc``-Split übrig lässt.

    Test-Segmente sind disjunkt und chronologisch. ``test_start`` ist explizit
    im Tupel (nicht aus ``train_end`` abgeleitet), damit die Embargo-Lücke real
    ist und nicht vom Wrapper wieder zugeschüttet wird.
    """
    if len(md_index) < 200:
        raise ValueError("Zu wenig Daten für Walk-Forward")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio muss in (0, 1) liegen")
    if embargo < 0:
        raise ValueError("embargo darf nicht negativ sein")

    timestamps = md_index.sort_values()
    n = len(timestamps)
    fold_len = n // n_folds
    if fold_len < 2:
        raise ValueError("Zu wenig Daten für so viele Folds")

    # Gewünschte Train-Spanne aus dem Ratio (wie zuvor: ratio/(1-ratio) · fold_len).
    desired_train = int(round(fold_len * train_ratio / (1.0 - train_ratio)))
    # Mindest-Train, damit der Fold nicht degeneriert ist und die Strategie ihre
    # Lookbacks (≈ embargo) im Train überhaupt berechnen kann.
    min_train = max(2 * embargo, 30)

    out: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for k in range(n_folds):
        test_start_i = k * fold_len
        test_end_i = min((k + 1) * fold_len - 1, n - 1)
        train_end_i = test_start_i - 1 - embargo  # 1-Bar-Abstand + Embargo-Lücke
        train_start_i = max(0, test_start_i - desired_train - embargo)
        if train_end_i < 0:
            continue  # kein OOS-Vorlauf (typischerweise der erste Fold)
        train_len = train_end_i - train_start_i + 1
        if train_len < min_train:
            continue  # zu kurzes Train-Fenster → Fold überspringen statt verfälschen
        out.append(
            (
                timestamps[train_start_i],
                timestamps[train_end_i],
                timestamps[test_start_i],
                timestamps[test_end_i],
            )
        )

    if not out:
        raise ValueError(
            "Kein gültiger Walk-Forward-Fold mit diesen Parametern "
            "(Daten zu kurz für Train-Fenster + Embargo)."
        )
    return out
