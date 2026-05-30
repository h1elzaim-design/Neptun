"""Sharpe ratio variants with proper statistical correction.

References
----------
Bailey, D. H. & López de Prado, M. (2012/2014).
    "The Sharpe Ratio Efficient Frontier" / "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
    Journal of Portfolio Management.

Formulas
--------
Let SR̂ be the observed Sharpe ratio of T return observations with sample
skewness γ₃ and (non-excess) kurtosis γ₄ (γ₄ = 3 under normality).

Standard error of the Sharpe estimate (Mertens, 2002):

    σ(SR̂)² = (1 / (T − 1)) · (1 − γ₃·SR̂ + (γ₄ − 1)/4 · SR̂²)

The Probabilistic Sharpe Ratio is the cumulative-normal probability that the
true Sharpe exceeds a benchmark SR★:

    PSR(SR★) = Φ((SR̂ − SR★) / σ(SR̂))

The Deflated Sharpe Ratio applies a multiple-testing correction. Given N
trials with cross-trial Sharpe-ratio variance V[SR], the expected maximum
Sharpe under the null is:

    SR★₀ = √V[SR] · [(1 − γ_e) · Φ⁻¹(1 − 1/N) + γ_e · Φ⁻¹(1 − 1/(N·e))]

with Euler-Mascheroni constant γ_e ≈ 0.5772.

    DSR = PSR(SR★₀)

A DSR close to 1 means the observed strategy is unlikely to be a fluke;
< 0.95 should be treated with serious skepticism in a sweep context.

All Sharpe inputs to these functions are **per-period** Sharpe ratios, not
annualised. Use :func:`annualised_sharpe` to bridge between the two
representations explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

EULER_MASCHERONI = 0.5772156649015329


# -----------------------------------------------------------------------------
# Standard-normal helpers (avoid hard scipy dependency)
# -----------------------------------------------------------------------------

def _phi(z: float) -> float:
    """Standard-normal CDF Φ(z) via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _inv_phi(p: float) -> float:
    """Inverse standard-normal CDF Φ⁻¹(p), Beasley-Springer-Moro approximation.

    Accuracy ~1e-9 over the open interval (0, 1). Calls outside that interval
    raise ValueError — callers must clip.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"_inv_phi domain error: p={p}")

    # Coefficients
    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# -----------------------------------------------------------------------------
# Annualisation
# -----------------------------------------------------------------------------

def annualised_sharpe(returns: Sequence[float], periods_per_year: float = 252.0) -> float:
    """Annualised Sharpe (zero risk-free).

    Defined as μ̂/σ̂ · √P where μ̂ and σ̂ are sample mean and std (ddof=1) of
    per-period returns and P is `periods_per_year`. Returns 0.0 on degenerate
    inputs (T<2 or σ̂=0).
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    mean = float(r.mean())
    std = float(r.std(ddof=1))
    # Float arithmetic can leave a sub-1e-15 std on constant inputs. Treat
    # anything below 1e-12 as "no signal" — it can't be a meaningful Sharpe.
    if std < 1e-12:
        return 0.0
    return mean / std * math.sqrt(periods_per_year)


# -----------------------------------------------------------------------------
# Sample moments (kept here so PSR is fully self-contained)
# -----------------------------------------------------------------------------

def _sample_skew_kurt(r: np.ndarray) -> tuple[float, float]:
    """Sample skewness (γ₃) and *non-excess* kurtosis (γ₄, normal → 3).

    Uses the population estimators (bias not corrected). The PSR derivation
    expects these conventions; using bias-corrected estimators is an
    asymptotic improvement only.
    """
    if r.size < 2:
        return 0.0, 3.0
    mu = r.mean()
    centered = r - mu
    m2 = float((centered ** 2).mean())
    if m2 == 0.0:
        return 0.0, 3.0
    m3 = float((centered ** 3).mean())
    m4 = float((centered ** 4).mean())
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2)  # NOT excess
    return skew, kurt


# -----------------------------------------------------------------------------
# Probabilistic Sharpe Ratio (PSR)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ProbabilisticSharpeResult:
    """Output of :func:`probabilistic_sharpe_ratio`.

    Attributes
    ----------
    sr_period_observed:
        Per-period observed Sharpe (μ̂/σ̂).
    sr_period_benchmark:
        Per-period benchmark Sharpe (SR★).
    sigma_sr:
        Standard error of SR̂ (Mertens 2002).
    psr:
        Φ((SR̂ − SR★) / σ(SR̂)). Interpreted as "probability the true Sharpe
        exceeds the benchmark".
    n_obs:
        Number of return observations T used.
    skew:
        Sample skewness γ₃.
    kurt:
        Sample non-excess kurtosis γ₄ (= 3 under normality).
    """

    sr_period_observed: float
    sr_period_benchmark: float
    sigma_sr: float
    psr: float
    n_obs: int
    skew: float
    kurt: float


def probabilistic_sharpe_ratio(
    returns: Sequence[float],
    *,
    benchmark_sharpe_annual: float = 0.0,
    periods_per_year: float = 252.0,
) -> ProbabilisticSharpeResult:
    """Probabilistic Sharpe Ratio of observed returns against a benchmark.

    The benchmark is supplied as an *annualised* Sharpe for ergonomics; it is
    converted to per-period internally so the math stays consistent.
    """
    r = np.asarray(returns, dtype=float)
    T = int(r.size)  # noqa: N806
    if T < 3:
        return ProbabilisticSharpeResult(0.0, 0.0, 0.0, 0.0, T, 0.0, 3.0)

    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma < 1e-12:
        return ProbabilisticSharpeResult(0.0, 0.0, 0.0, 0.0, T, 0.0, 3.0)

    sr_hat = mu / sigma  # per-period
    sr_star = benchmark_sharpe_annual / math.sqrt(periods_per_year)

    skew, kurt = _sample_skew_kurt(r)

    # Mertens variance: (1 − γ₃·SR̂ + (γ₄ − 1)/4·SR̂²) / (T − 1)
    variance_term = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2
    variance_term = max(variance_term, 1e-12)
    sigma_sr = math.sqrt(variance_term / (T - 1))

    psr = _phi((sr_hat - sr_star) / sigma_sr)
    return ProbabilisticSharpeResult(
        sr_period_observed=sr_hat,
        sr_period_benchmark=sr_star,
        sigma_sr=sigma_sr,
        psr=psr,
        n_obs=T,
        skew=skew,
        kurt=kurt,
    )


# -----------------------------------------------------------------------------
# Deflated Sharpe Ratio (DSR)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    """Output of :func:`deflated_sharpe_ratio`.

    Attributes
    ----------
    expected_max_sharpe_period:
        SR★₀ — the expected maximum per-period Sharpe under the null
        across `n_trials` independent attempts.
    psr:
        Underlying PSR computed against SR★₀.
    dsr:
        Identical to `psr` here; named separately to make intent explicit in
        downstream code.
    n_trials:
        N (count of competing strategies in the selection set).
    cross_trial_sharpe_variance:
        V[SR] across trials, in per-period Sharpe units. Required input.
    psr_components:
        The inner :class:`ProbabilisticSharpeResult` for transparency.
    """

    expected_max_sharpe_period: float
    psr: float
    dsr: float
    n_trials: int
    cross_trial_sharpe_variance: float
    psr_components: ProbabilisticSharpeResult


def _expected_max_sharpe_period(
    n_trials: int,
    cross_trial_sharpe_variance: float,
) -> float:
    """SR★₀ — Bailey & López de Prado (2014) closed-form approximation."""
    if n_trials < 2:
        return 0.0
    v = max(cross_trial_sharpe_variance, 0.0)
    if v == 0.0:
        return 0.0
    sqrt_v = math.sqrt(v)
    p1 = 1.0 - 1.0 / n_trials
    p2 = 1.0 - 1.0 / (n_trials * math.e)
    # clip to safe domain of _inv_phi
    p1 = min(max(p1, 1e-12), 1.0 - 1e-12)
    p2 = min(max(p2, 1e-12), 1.0 - 1e-12)
    return sqrt_v * ((1.0 - EULER_MASCHERONI) * _inv_phi(p1) + EULER_MASCHERONI * _inv_phi(p2))


def deflated_sharpe_ratio(
    returns: Sequence[float],
    *,
    n_trials: int,
    cross_trial_sharpe_variance: float,
    periods_per_year: float = 252.0,
) -> DeflatedSharpeResult:
    """Deflated Sharpe Ratio.

    Parameters
    ----------
    returns:
        Per-period returns of the *selected* (winning) strategy.
    n_trials:
        Number of independent strategies that were compared. For a sweep with
        K configurations, N = K.
    cross_trial_sharpe_variance:
        V[SR] across all `n_trials` trials, **in per-period Sharpe units**.
        Compute it once over the sweep before calling this function.
    periods_per_year:
        Used to convert any annualised quantities; defaults to 252 trading
        days. The returned `expected_max_sharpe_period` is per-period.

    Notes
    -----
    DSR is undefined for a single backtest. If you have no sweep data, do not
    call this function — use the underlying PSR alone and disclose that the
    deflation has not been performed.
    """
    if n_trials < 2:
        raise ValueError("n_trials must be ≥ 2 — DSR is undefined for a single trial")

    sr_star_period = _expected_max_sharpe_period(n_trials, cross_trial_sharpe_variance)
    sr_star_annual = sr_star_period * math.sqrt(periods_per_year)

    psr_result = probabilistic_sharpe_ratio(
        returns,
        benchmark_sharpe_annual=sr_star_annual,
        periods_per_year=periods_per_year,
    )

    return DeflatedSharpeResult(
        expected_max_sharpe_period=sr_star_period,
        psr=psr_result.psr,
        dsr=psr_result.psr,  # explicit alias for downstream clarity
        n_trials=n_trials,
        cross_trial_sharpe_variance=cross_trial_sharpe_variance,
        psr_components=psr_result,
    )
