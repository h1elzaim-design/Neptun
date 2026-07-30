"""Stationary-bootstrap inference for backtest return series.

Reference
---------
Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap."
    Journal of the American Statistical Association, 89(428).

Problem
-------
Point metrics (Sharpe, max drawdown) off a single historical path say nothing
about their sampling variability — and financial returns are autocorrelated
and heteroskedastic, so the classic i.i.d. bootstrap understates that
variability. The stationary bootstrap resamples *blocks* of consecutive
observations whose lengths are geometrically distributed with mean ``L``,
which preserves short-range dependence while keeping the resampled series
stationary (unlike the fixed-block bootstrap).

Provided here:

- :func:`stationary_bootstrap_indices` — the resampling core (vectorised,
  deterministic given a seed).
- :func:`bootstrap_sharpe_ci` — percentile confidence interval for the
  annualised Sharpe plus a one-sided bootstrap p-value against SR ≤ 0.
- :func:`bootstrap_drawdown_distribution` — the distribution of maximum
  drawdown across resampled paths ("how bad could this exact return stream
  have been in a different order, dependence preserved?").

Block length
------------
The optimal expected block length grows like T^(1/3) (Politis & Romano 1994;
Politis & White 2004 refine the constant). We default to ``round(T ** (1/3))``
— for daily data that is ~13 bars at 10y history — and expose the parameter
for callers that want to stress it.

Determinism
-----------
All randomness flows through one ``numpy.random.default_rng(seed)``; identical
inputs and seed give bit-identical results.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from quantrace.stats.sharpe import annualised_sharpe

DEFAULT_N_RESAMPLES = 1000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 7


def default_block_length(n_obs: int) -> float:
    """T^(1/3) rule for the expected block length (≥ 1)."""
    return max(1.0, round(float(n_obs) ** (1.0 / 3.0)))


def stationary_bootstrap_indices(
    n_obs: int,
    *,
    n_resamples: int,
    avg_block_len: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Index matrix (n_resamples × n_obs) for the stationary bootstrap.

    Each row is one resampled path: blocks start at uniform positions and
    continue (wrapping around the series end, as in the original paper) with
    probability 1 − 1/L per step, so block lengths are geometric with mean L.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be ≥ 2")
    if n_resamples < 1:
        raise ValueError("n_resamples must be ≥ 1")
    L = float(avg_block_len) if avg_block_len is not None else default_block_length(n_obs)  # noqa: N806
    if L < 1.0:
        raise ValueError("avg_block_len must be ≥ 1")
    rng = rng or np.random.default_rng(DEFAULT_SEED)

    p = 1.0 / L
    # new_block[b, t] — does a fresh block start at position t? (t=0: always)
    new_block = rng.random((n_resamples, n_obs)) < p
    new_block[:, 0] = True
    starts = rng.integers(0, n_obs, size=(n_resamples, n_obs))

    cols = np.arange(n_obs)
    # Position of the most recent block start at each t (per row):
    # forward-fill the column index over the new-block mask.
    start_pos = np.maximum.accumulate(np.where(new_block, cols, 0), axis=1)
    # Start index of the active block, and the offset within it.
    block_start = np.take_along_axis(starts, start_pos, axis=1)
    offset = cols - start_pos
    return (block_start + offset) % n_obs


@dataclass(frozen=True, slots=True)
class BootstrapSharpeResult:
    """Output of :func:`bootstrap_sharpe_ci`.

    Attributes
    ----------
    sharpe_annual:
        Annualised Sharpe of the observed series.
    ci_low, ci_high:
        Percentile bootstrap confidence bounds for the annualised Sharpe.
    confidence:
        Two-sided confidence level of the interval (e.g. 0.95).
    p_value:
        One-sided bootstrap p-value for H0: true SR ≤ 0, computed as the
        add-one-corrected fraction of resampled Sharpes ≤ 0.
    n_resamples, n_obs, avg_block_len, seed:
        Resampling configuration, persisted for reproducibility.
    """

    sharpe_annual: float
    ci_low: float
    ci_high: float
    confidence: float
    p_value: float
    n_resamples: int
    n_obs: int
    avg_block_len: float
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        """JSON-friendly payload (mirrors the `fdr` summary pattern)."""
        return {
            "sharpe_annual": self.sharpe_annual,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "p_value": self.p_value,
            "n_resamples": self.n_resamples,
            "n_obs": self.n_obs,
            "avg_block_len": self.avg_block_len,
            "seed": self.seed,
            "method": "stationary_bootstrap",
        }


def bootstrap_sharpe_ci(
    returns: Sequence[float],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    avg_block_len: float | None = None,
    periods_per_year: float = 252.0,
    seed: int = DEFAULT_SEED,
) -> BootstrapSharpeResult:
    """Percentile-bootstrap CI and p-value for the annualised Sharpe ratio.

    A CI that straddles zero is the plainest possible statement that the
    sample cannot distinguish the strategy's edge from noise — regardless of
    how attractive the point Sharpe looks.
    """
    if not (0.5 < confidence < 1.0):
        raise ValueError("confidence must be in (0.5, 1.0)")
    r = np.asarray(list(returns), dtype=float)
    r = r[np.isfinite(r)]
    T = int(r.size)  # noqa: N806
    if T < 8:
        raise ValueError("need ≥ 8 return observations to bootstrap a Sharpe CI")

    L = float(avg_block_len) if avg_block_len is not None else default_block_length(T)  # noqa: N806
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(T, n_resamples=n_resamples, avg_block_len=L, rng=rng)
    paths = r[idx]  # (B, T)

    mu = paths.mean(axis=1)
    sd = paths.std(axis=1, ddof=1)
    root = math.sqrt(periods_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        srs = np.where(sd > 1e-12, mu / sd * root, 0.0)

    alpha = 1.0 - confidence
    lo, hi = np.quantile(srs, [alpha / 2.0, 1.0 - alpha / 2.0])
    # Add-one correction keeps the p-value away from an over-confident 0.
    p_value = (1.0 + float((srs <= 0.0).sum())) / (n_resamples + 1.0)

    return BootstrapSharpeResult(
        sharpe_annual=annualised_sharpe(r, periods_per_year=periods_per_year),
        ci_low=float(lo),
        ci_high=float(hi),
        confidence=confidence,
        p_value=p_value,
        n_resamples=n_resamples,
        n_obs=T,
        avg_block_len=L,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class BootstrapDrawdownResult:
    """Output of :func:`bootstrap_drawdown_distribution`.

    Attributes
    ----------
    max_dd_observed:
        Maximum drawdown of the observed path (negative fraction).
    quantiles:
        Mapping quantile → max drawdown across resampled paths, e.g.
        ``{0.05: -0.41, 0.5: -0.22, 0.95: -0.12}``. The 5%-quantile answers
        "with these same returns in a dependence-preserving reshuffle, how
        deep does the worst 1-in-20 path go?" — the honest sizing input,
        rather than the single historical realisation.
    prob_worse_than_observed:
        Fraction of resampled paths whose max drawdown is deeper (more
        negative) than the observed one.
    n_resamples, n_obs, avg_block_len, seed:
        Resampling configuration.
    """

    max_dd_observed: float
    quantiles: dict[float, float]
    prob_worse_than_observed: float
    n_resamples: int
    n_obs: int
    avg_block_len: float
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "max_dd_observed": self.max_dd_observed,
            "quantiles": {f"{q:g}": v for q, v in self.quantiles.items()},
            "prob_worse_than_observed": self.prob_worse_than_observed,
            "n_resamples": self.n_resamples,
            "n_obs": self.n_obs,
            "avg_block_len": self.avg_block_len,
            "seed": self.seed,
            "method": "stationary_bootstrap",
        }


def _max_drawdown_paths(paths: np.ndarray) -> np.ndarray:
    """Max drawdown per row of a (B, T) matrix of per-period returns."""
    equity = np.cumprod(1.0 + paths, axis=1)
    peaks = np.maximum.accumulate(equity, axis=1)
    dd = equity / peaks - 1.0
    return dd.min(axis=1)


def bootstrap_drawdown_distribution(
    returns: Sequence[float],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    avg_block_len: float | None = None,
    seed: int = DEFAULT_SEED,
) -> BootstrapDrawdownResult:
    """Distribution of max drawdown across stationary-bootstrap resamples."""
    r = np.asarray(list(returns), dtype=float)
    r = r[np.isfinite(r)]
    T = int(r.size)  # noqa: N806
    if T < 8:
        raise ValueError("need ≥ 8 return observations to bootstrap drawdowns")

    L = float(avg_block_len) if avg_block_len is not None else default_block_length(T)  # noqa: N806
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(T, n_resamples=n_resamples, avg_block_len=L, rng=rng)
    dds = _max_drawdown_paths(r[idx])
    observed = float(_max_drawdown_paths(r.reshape(1, -1))[0])

    qs = sorted(float(q) for q in quantiles)
    q_vals = np.quantile(dds, qs)
    return BootstrapDrawdownResult(
        max_dd_observed=observed,
        quantiles={q: float(v) for q, v in zip(qs, q_vals, strict=True)},
        prob_worse_than_observed=float((dds < observed).mean()),
        n_resamples=n_resamples,
        n_obs=T,
        avg_block_len=L,
        seed=seed,
    )
