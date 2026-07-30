"""Regime-conditioned performance metrics for a completed backtest.

Given an equity curve and price data, fits a RegimeDetector on the prices
and partitions the equity curve returns by regime label. Returns per-regime
statistics for evaluation scoring and the analytics API.

Computation is offline (call once after run_backtest while the equity_curve
is still in memory). Results are stored in BacktestResult.regime_metrics and
persisted in the backtest JSON artefact.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from quantrace.regime.detector import RegimeDetector

log = logging.getLogger(__name__)

_ANN = 252  # trading days per year


def regime_conditioned_metrics(
    equity_curve: pd.Series,
    prices: pd.DataFrame | pd.Series,
    *,
    n_states: int = 3,
    feature_window: int = 21,
) -> dict[str, Any] | None:
    """Fit a RegimeDetector on *prices* and return per-regime performance stats.

    Parameters
    ----------
    equity_curve:
        Daily portfolio value from the backtest (DatetimeIndex, floats).
    prices:
        Close prices used to fit the HMM. Same universe as the backtest.
    n_states:
        Number of HMM states (2–5).
    feature_window:
        Rolling window for trend/vol features (used as burn-in drop).

    Returns
    -------
    dict or None
        ``{"n_states": int, "feature_window": int, "rows": [...]}``
        where each row contains per-regime metrics. Returns None if the
        computation fails (too little data, degenerate HMM).
    """
    try:
        detector = RegimeDetector(n_states=n_states, feature_window=feature_window)
        detector.fit(prices)
        regime_series = detector.regime_series(prices, mode="filter")
    except Exception as exc:
        log.warning("regime_conditioned_metrics: HMM fit failed: %s", exc)
        return None

    # Align equity_curve and regime_series on common DatetimeIndex.
    equity = equity_curve.copy()
    equity.index = pd.to_datetime(equity.index)
    regime_series.index = pd.to_datetime(regime_series.index)

    common = equity.index.intersection(regime_series.index)
    if len(common) < feature_window + 10:
        log.warning(
            "regime_conditioned_metrics: only %d aligned points — skipping", len(common)
        )
        return None

    equity = equity.loc[common].astype(float)
    labels = regime_series.loc[common]
    returns = equity.pct_change().fillna(0.0)
    total_days = len(common)

    rows: list[dict[str, Any]] = []
    for label in detector.labels:
        mask = labels == label
        n_days = int(mask.sum())
        if n_days < 5:
            continue

        reg_rets = returns.loc[mask]
        sharpe, sortino, cagr, vol = _risk_metrics(reg_rets)
        max_dd, avg_dd, uw_pct = _dd_stats(equity.loc[mask])
        dates = labels.index[mask]

        rows.append(
            {
                "label": label,
                "n_days": n_days,
                "pct_time": round(n_days / total_days, 4),
                "period_start": str(dates[0].date()) if len(dates) else "",
                "period_end": str(dates[-1].date()) if len(dates) else "",
                "cagr": round(cagr, 6),
                "vol": round(vol, 6),
                "sharpe": round(sharpe, 4),
                "sortino": round(sortino, 4),
                "max_drawdown": round(max_dd, 6),
                "avg_drawdown": round(avg_dd, 6),
                "time_underwater_pct": round(uw_pct, 4),
            }
        )

    if not rows:
        return None

    return {"n_states": n_states, "feature_window": feature_window, "rows": rows}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _risk_metrics(rets: pd.Series) -> tuple[float, float, float, float]:
    """Return (annualised_sharpe, annualised_sortino, annualised_cagr, annualised_vol)."""
    if rets.empty:
        return 0.0, 0.0, 0.0, 0.0
    mu = float(rets.mean())
    std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    down = float(rets[rets < 0].std(ddof=1)) if (rets < 0).any() else 0.0
    sharpe = np.sqrt(_ANN) * mu / std if std > 1e-12 else 0.0
    sortino = np.sqrt(_ANN) * mu / down if down > 1e-12 else 0.0
    vol = std * np.sqrt(_ANN)
    # Annualised CAGR from summed log-returns across discontinuous regime days.
    n_years = len(rets) / _ANN
    log_sum = float(np.log1p(rets).sum())
    cagr = np.exp(log_sum / n_years) - 1 if n_years > 1e-9 else 0.0
    return (
        float(sharpe) if np.isfinite(sharpe) else 0.0,
        float(sortino) if np.isfinite(sortino) else 0.0,
        float(cagr) if np.isfinite(cagr) else 0.0,
        float(vol) if np.isfinite(vol) else 0.0,
    )


def _dd_stats(equity: pd.Series) -> tuple[float, float, float]:
    """Return (max_drawdown, avg_drawdown, time_underwater_pct) on a regime slice."""
    if len(equity) < 2:
        return 0.0, 0.0, 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1
    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    uw_pct = float((dd < 0).sum() / len(dd))
    return (
        max_dd if np.isfinite(max_dd) else 0.0,
        avg_dd if np.isfinite(avg_dd) else 0.0,
        uw_pct if np.isfinite(uw_pct) else 0.0,
    )
