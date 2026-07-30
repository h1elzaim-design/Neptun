"""Factor attribution — is the "edge" alpha, or repackaged beta?

References
----------
Newey, W. K. & West, K. D. (1987). "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
    Econometrica, 55(3).
Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*, ch. 17
    (performance attribution via factor regression).

Model
-----
OLS time-series regression of strategy returns on K factor returns:

    r_t = α + Σ_k β_k · f_{k,t} + ε_t

- **α (annualised)** is the return the factor set cannot explain — the only
  part of a backtest worth paying for. A strategy whose α t-stat is
  indistinguishable from zero after regressing on SPY/momentum/value/quality
  is beta in a trench coat.
- **β_k** are the factor loadings (hedgeable exposures).
- Standard errors are **Newey–West HAC** (Bartlett kernel) because daily
  strategy returns are autocorrelated and heteroskedastic; plain OLS
  t-stats overstate significance. Default lag: ⌊4·(T/100)^(2/9)⌋.
- **residual_sharpe_annual** is α divided by residual volatility, annualised
  — the information ratio against this factor model.

Everything is from-scratch numpy (no statsmodels), matching the rest of
``quantrace.stats``. Pure functions, no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantrace.stats.sharpe import _phi


def default_hac_lags(n_obs: int) -> int:
    """Newey–West rule-of-thumb lag truncation: ⌊4·(T/100)^(2/9)⌋."""
    return int(math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


@dataclass(frozen=True, slots=True)
class FactorExposure:
    """One factor's loading in the regression."""

    name: str
    beta: float
    t_stat: float
    p_value: float  # two-sided, normal approximation

    def to_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "beta": self.beta,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
        }


@dataclass(frozen=True, slots=True)
class FactorAttributionResult:
    """Output of :func:`factor_attribution`.

    Attributes
    ----------
    alpha_annual:
        Intercept scaled by periods_per_year — unexplained annual return.
    alpha_t_stat, alpha_p_value:
        HAC-robust significance of the intercept (H0: α = 0, two-sided).
    exposures:
        Per-factor loadings with HAC t-stats.
    r_squared, adj_r_squared:
        Fraction of return variance the factors explain. High R² + flat α
        means the backtest is a factor portfolio, not a strategy.
    residual_sharpe_annual:
        Annualised α / residual volatility — information ratio vs the model.
    n_obs, hac_lags, periods_per_year:
        Regression configuration, persisted for reproducibility.
    """

    alpha_annual: float
    alpha_t_stat: float
    alpha_p_value: float
    exposures: list[FactorExposure]
    r_squared: float
    adj_r_squared: float
    residual_sharpe_annual: float
    n_obs: int
    hac_lags: int
    periods_per_year: float

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha_annual": self.alpha_annual,
            "alpha_t_stat": self.alpha_t_stat,
            "alpha_p_value": self.alpha_p_value,
            "exposures": [e.to_dict() for e in self.exposures],
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "residual_sharpe_annual": self.residual_sharpe_annual,
            "n_obs": self.n_obs,
            "hac_lags": self.hac_lags,
            "periods_per_year": self.periods_per_year,
            "method": "ols_newey_west",
        }


def _newey_west_cov(x: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """HAC covariance of the OLS estimator: (X'X)⁻¹ · S · (X'X)⁻¹.

    S is the Bartlett-weighted long-run covariance of the score x_t·ε_t.
    """
    t_obs = x.shape[0]
    xtx_inv = np.linalg.pinv(x.T @ x)
    scores = x * resid[:, None]  # (T, K+1)
    s = scores.T @ scores  # lag-0 term
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        s += w * (gamma + gamma.T)
    _ = t_obs  # sandwich already carries the sample size through X'X
    return xtx_inv @ s @ xtx_inv


def factor_attribution(
    strategy_returns: np.ndarray,
    factor_returns: np.ndarray,
    factor_names: list[str],
    *,
    periods_per_year: float = 252.0,
    hac_lags: int | None = None,
) -> FactorAttributionResult:
    """OLS factor regression with Newey–West standard errors.

    Parameters
    ----------
    strategy_returns:
        (T,) per-period strategy returns.
    factor_returns:
        (T × K) per-period factor returns, time-aligned with the strategy.
        Callers own the alignment (intersect dates, drop NaN) — this function
        rejects non-finite input rather than silently dropping rows.
    factor_names:
        K names, in column order.
    periods_per_year:
        Annualisation factor for α and the residual Sharpe.
    hac_lags:
        Newey–West lag truncation; ``None`` → ⌊4·(T/100)^(2/9)⌋.

    Raises
    ------
    ValueError
        Shape mismatch, non-finite values, or too few observations
        (needs T ≥ K + 10 for a regression worth reporting).
    """
    y = np.asarray(strategy_returns, dtype=float).ravel()
    f = np.asarray(factor_returns, dtype=float)
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    t_obs, k = f.shape
    if y.size != t_obs:
        raise ValueError(f"length mismatch: strategy T={y.size}, factors T={t_obs}")
    if len(factor_names) != k:
        raise ValueError(f"{k} factor columns but {len(factor_names)} names")
    if not (np.isfinite(y).all() and np.isfinite(f).all()):
        raise ValueError("inputs contain non-finite values — align/drop NaN upstream")
    if t_obs < k + 10:
        raise ValueError(f"need ≥ {k + 10} observations for {k} factors, got {t_obs}")

    x = np.column_stack([np.ones(t_obs), f])  # intercept first
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta

    lags = default_hac_lags(t_obs) if hac_lags is None else int(hac_lags)
    if lags < 0:
        raise ValueError("hac_lags must be ≥ 0")
    cov = _newey_west_cov(x, resid, lags)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 1e-15, beta / se, 0.0)

    def p_two_sided(t: float) -> float:
        return 2.0 * (1.0 - _phi(abs(t)))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-24 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (t_obs - 1) / max(1, t_obs - k - 1)

    resid_std = float(resid.std(ddof=k + 1)) if t_obs > k + 1 else 0.0
    alpha = float(beta[0])
    residual_sharpe = (
        alpha / resid_std * math.sqrt(periods_per_year) if resid_std > 1e-12 else 0.0
    )

    exposures = [
        FactorExposure(
            name=name,
            beta=float(beta[i + 1]),
            t_stat=float(t_stats[i + 1]),
            p_value=p_two_sided(float(t_stats[i + 1])),
        )
        for i, name in enumerate(factor_names)
    ]

    return FactorAttributionResult(
        alpha_annual=alpha * periods_per_year,
        alpha_t_stat=float(t_stats[0]),
        alpha_p_value=p_two_sided(float(t_stats[0])),
        exposures=exposures,
        r_squared=float(r2),
        adj_r_squared=float(adj_r2),
        residual_sharpe_annual=float(residual_sharpe),
        n_obs=int(t_obs),
        hac_lags=lags,
        periods_per_year=periods_per_year,
    )


@dataclass(frozen=True, slots=True)
class RollingExposurePoint:
    """One window of :func:`rolling_factor_attribution`.

    ``end_index`` is the (0-based) index of the window's last observation in
    the input arrays — the caller maps it back to a date.
    """

    end_index: int
    alpha_annual: float
    betas: tuple[float, ...]  # in factor_names order
    r_squared: float


def rolling_factor_attribution(
    strategy_returns: np.ndarray,
    factor_returns: np.ndarray,
    *,
    window: int = 126,
    step: int = 5,
    periods_per_year: float = 252.0,
) -> list[RollingExposurePoint]:
    """Rolling-window OLS betas — the factor-exposure *timeline*.

    Answers the drift question the full-sample regression averages away:
    a strategy that is market-neutral on average but swings between β=+1
    and β=−1 has a very different risk profile than one that is flat at 0.

    Plain OLS point estimates per window — no HAC errors here; per-window
    inference on 126 observations is noise, the timeline's information is the
    *path* of the betas. Use :func:`factor_attribution` on the full sample
    for significance.

    Parameters
    ----------
    strategy_returns, factor_returns:
        (T,) and (T × K), time-aligned, finite (same contract as
        :func:`factor_attribution`).
    window:
        Observations per regression window (default 126 ≈ 6 months daily).
    step:
        Windows advance by this many observations (default 5 ≈ weekly),
        trading resolution for compute/payload size.

    Raises
    ------
    ValueError
        Shape mismatch / non-finite input, window too small for K factors,
        or T < window (no complete window).
    """
    y = np.asarray(strategy_returns, dtype=float).ravel()
    f = np.asarray(factor_returns, dtype=float)
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    t_obs, k = f.shape
    if y.size != t_obs:
        raise ValueError(f"length mismatch: strategy T={y.size}, factors T={t_obs}")
    if not (np.isfinite(y).all() and np.isfinite(f).all()):
        raise ValueError("inputs contain non-finite values — align/drop NaN upstream")
    if window < k + 10:
        raise ValueError(f"window={window} too small for {k} factors (need ≥ {k + 10})")
    if step < 1:
        raise ValueError("step must be ≥ 1")
    if t_obs < window:
        raise ValueError(f"T={t_obs} shorter than window={window}")

    out: list[RollingExposurePoint] = []
    for end in range(window, t_obs + 1, step):
        ys = y[end - window : end]
        xs = np.column_stack([np.ones(window), f[end - window : end]])
        beta, *_ = np.linalg.lstsq(xs, ys, rcond=None)
        resid = ys - xs @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-24 else 0.0
        out.append(
            RollingExposurePoint(
                end_index=end - 1,
                alpha_annual=float(beta[0]) * periods_per_year,
                betas=tuple(float(b) for b in beta[1:]),
                r_squared=float(r2),
            )
        )
    return out
