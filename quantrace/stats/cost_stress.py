"""Cost-stress: re-evaluate a strategy under harsher transaction-cost assumptions.

A strategy that only survives at unrealistically low costs is not a real
strategy. This module re-derives Sharpe / CAGR / Max-DD on a return series
after applying a multiplicative drag derived from baseline slippage and fee
assumptions plus a turnover estimate.

Approximation
-------------
We don't always have trade-level data — only a return series. Define the
per-period drag added by stress multipliers (m_s, m_f) as::

    Δdrag_t = (m_s − 1) · slippage_bps_per_trade · trades_per_period_t
            + (m_f − 1) · fee_bps_per_trade      · trades_per_period_t
            ──────────────────────────────────────────────────────────
                                      10 000

If `trades_per_period` is constant (or unknown), pass a scalar; otherwise pass
a same-length sequence. When unknown the safe default is 1 round-trip per
period (worst-case for liquid daily strategies).

Compared to a full trade-level resimulation this is conservative *only when*
the baseline cost model already deducted the assumed slippage from returns —
in our pipeline the equity curves coming out of `quantrace backtest` already
include a baseline cost deduction, so this stress represents an *additional*
drag on top.

The function is deterministic — identical inputs always produce identical
outputs — and never mutates its inputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CostStressResult:
    slippage_multiplier: float
    fee_multiplier: float
    baseline_slippage_bps: float
    baseline_fee_bps: float
    trades_per_period: float
    stressed_returns: list[float]
    stressed_cagr: float
    stressed_sharpe_annual: float
    stressed_vol_annual: float
    stressed_max_drawdown: float
    survives: bool


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    return float(((equity - peaks) / peaks).min())


def _summary(returns: np.ndarray, periods_per_year: float) -> tuple[float, float, float, float]:
    if returns.size < 2:
        return 0.0, 0.0, 0.0, 0.0
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    sharpe_annual = (mu / sigma * math.sqrt(periods_per_year)) if sigma > 0 else 0.0
    vol_annual = sigma * math.sqrt(periods_per_year)
    cumret = float(np.prod(1.0 + returns) - 1.0)
    n_years = returns.size / periods_per_year
    cagr = (1.0 + cumret) ** (1.0 / max(n_years, 1e-9)) - 1.0
    max_dd = _max_drawdown(returns)
    return cagr, sharpe_annual, vol_annual, max_dd


def apply_cost_stress(
    returns: Sequence[float],
    *,
    slippage_multiplier: float = 1.0,
    fee_multiplier: float = 1.0,
    baseline_slippage_bps: float = 3.0,
    baseline_fee_bps: float = 0.35,
    trades_per_period: float | Sequence[float] = 1.0,
    periods_per_year: float = 252.0,
    survival_min_sharpe: float = 0.0,
) -> CostStressResult:
    """Apply a (slippage × fee) cost stress and recompute headline stats.

    Parameters
    ----------
    returns:
        Per-period returns (already net of baseline cost assumption).
    slippage_multiplier, fee_multiplier:
        Factors ≥ 1.0. A multiplier of 1.0 leaves returns untouched.
    baseline_slippage_bps, baseline_fee_bps:
        Baseline assumed at backtest time. Defaults model a liquid ETF with
        IBKR-tiered commission (~0.35 bps) and ~3 bps round-trip slippage.
    trades_per_period:
        Scalar or sequence aligned with `returns`. 1.0 ≈ daily round-trip.
        Buy-and-hold–style strategies should use much smaller values.
    periods_per_year:
        For annualisation. 252 (trading days) is the default.
    survival_min_sharpe:
        A strategy is reported as `survives=False` if its stressed
        annualised Sharpe falls below this threshold. Default 0 — the
        strategy must at least be profitable above the riskless rate after
        stress.

    Returns
    -------
    CostStressResult with the re-computed series and headline stats.
    """
    if slippage_multiplier < 0 or fee_multiplier < 0:
        raise ValueError("Multipliers must be non-negative")

    r = np.asarray(returns, dtype=float).copy()
    if r.size == 0:
        return CostStressResult(
            slippage_multiplier, fee_multiplier, baseline_slippage_bps,
            baseline_fee_bps, _scalar_tpp(trades_per_period),
            [], 0.0, 0.0, 0.0, 0.0, False,
        )

    tpp = np.asarray(
        trades_per_period
        if hasattr(trades_per_period, "__len__")
        else np.full(r.size, float(trades_per_period)),
        dtype=float,
    )
    if tpp.size != r.size:
        raise ValueError(
            f"trades_per_period length {tpp.size} must equal returns length {r.size}"
        )

    extra_slip = (slippage_multiplier - 1.0) * baseline_slippage_bps / 1e4
    extra_fee = (fee_multiplier - 1.0) * baseline_fee_bps / 1e4
    delta_drag = (extra_slip + extra_fee) * tpp

    stressed = r - delta_drag
    cagr, sharpe, vol, max_dd = _summary(stressed, periods_per_year)
    survives = sharpe > survival_min_sharpe

    return CostStressResult(
        slippage_multiplier=slippage_multiplier,
        fee_multiplier=fee_multiplier,
        baseline_slippage_bps=baseline_slippage_bps,
        baseline_fee_bps=baseline_fee_bps,
        trades_per_period=float(np.mean(tpp)),
        stressed_returns=stressed.tolist(),
        stressed_cagr=cagr,
        stressed_sharpe_annual=sharpe,
        stressed_vol_annual=vol,
        stressed_max_drawdown=max_dd,
        survives=survives,
    )


def _scalar_tpp(v: float | Sequence[float]) -> float:
    if hasattr(v, "__len__"):
        seq = list(v)  # type: ignore[arg-type]
        return float(sum(seq) / len(seq)) if seq else 0.0
    return float(v)
