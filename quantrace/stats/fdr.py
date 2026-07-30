"""False-Discovery-Rate control for backtest selection sets.

A parameter sweep tests many strategy configurations at once; a walk-forward
tests one configuration on several out-of-sample windows. Either way, the
per-test question is "is this Sharpe distinguishable from zero?" — and asking
it m times inflates the number of false positives unless the *family* of tests
is controlled.

This module implements the Benjamini–Hochberg (1995) step-up procedure, which
controls the expected fraction of false discoveries among the tests declared
significant (the FDR) at level α.

Validity under dependence
-------------------------
BH controls the FDR exactly under independence and, by Benjamini–Yekutieli
(2001), under *positive regression dependence* (PRDS). Sweep combinations are
positively correlated — neighbouring parameter values produce overlapping
positions and therefore positively correlated return streams — so BH is the
appropriate (and not overly conservative) choice here. The BY log-factor
correction would be needed only for arbitrary/negative dependence, which a
parameter grid does not exhibit.

p-values
--------
Per-test p-values come from the Mertens (2002) standard error of the Sharpe
estimate, the same machinery behind the Probabilistic Sharpe Ratio:

    p = 1 − Φ(SR̂ / σ(SR̂)),   σ(SR̂)² = (1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) / (T−1)

i.e. a one-sided test of H0: true SR ≤ 0 that accounts for skewness and fat
tails of the underlying return stream when those moments are supplied.

References
----------
Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate."
    Journal of the Royal Statistical Society B.
Benjamini, Y. & Yekutieli, D. (2001). "The Control of the False Discovery Rate
    in Multiple Testing under Dependency." Annals of Statistics.
Harvey, C. & Liu, Y. (2015). "Backtesting." Journal of Portfolio Management —
    argues for exactly this kind of multiple-testing haircut on trading signals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from quantrace.stats.sharpe import probabilistic_sharpe_from_summary

# Screening-level FDR. 5% is the conventional, defensible default; loosen to
# 0.10 deliberately (and say so in the note) if the research phase needs recall.
DEFAULT_FDR_ALPHA = 0.05


def sharpe_p_value(
    *,
    sharpe_period: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """One-sided p-value for H0: true Sharpe ≤ 0.

    Uses the Mertens standard error via the PSR machinery, so supplied
    skew/kurtosis widen the error bars for fat-tailed return streams. Inputs
    are **per-period** (divide an annualised Sharpe by √periods_per_year).
    Degenerate inputs (T < 3) return 1.0 — no observations, no evidence.
    """
    res = probabilistic_sharpe_from_summary(
        observed_sharpe_period=sharpe_period,
        n_obs=n_obs,
        benchmark_sharpe_period=0.0,
        skew=skew,
        kurt=kurt,
    )
    if res.sigma_sr == 0.0:
        return 1.0
    return float(min(max(1.0 - res.psr, 0.0), 1.0))


@dataclass(frozen=True, slots=True)
class FdrResult:
    """Output of :func:`benjamini_hochberg`.

    Attributes
    ----------
    q_values:
        BH-adjusted p-values in the input order. ``q_values[i] ≤ alpha`` is
        exactly the BH rejection rule; a q-value reads as "the smallest FDR
        level at which this test would still be declared a discovery".
    significant:
        Rejection flags in the input order (``q ≤ alpha``).
    alpha:
        The FDR level the procedure was run at.
    n_tests / n_significant:
        Size of the family and how many survived.
    """

    q_values: list[float]
    significant: list[bool]
    alpha: float
    n_tests: int
    n_significant: int


def benjamini_hochberg(
    p_values: Sequence[float],
    *,
    alpha: float = DEFAULT_FDR_ALPHA,
) -> FdrResult:
    """Benjamini–Hochberg step-up procedure.

    Sort the m p-values ascending, find the largest k with
    p_(k) ≤ k/m·α, and reject hypotheses 1..k. The returned q-values are the
    standard monotone adjustment q_(i) = min_{j ≥ i}( p_(j) · m / j ), so the
    rejection set is exactly ``{i : q_i ≤ alpha}``.

    NaN p-values are treated as 1.0 (no evidence) so a failed trial can never
    become a discovery but still counts toward the family size m.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    p = np.asarray(list(p_values), dtype=float)
    m = int(p.size)
    if m == 0:
        return FdrResult([], [], alpha, 0, 0)

    p = np.where(np.isnan(p), 1.0, np.clip(p, 0.0, 1.0))

    order = np.argsort(p, kind="stable")
    ranked = p[order]
    q_ranked = ranked * m / np.arange(1, m + 1, dtype=float)
    # Step-up monotonicity: q_(i) = min over j ≥ i of p_(j)·m/j.
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q = np.empty(m, dtype=float)
    q[order] = q_ranked
    significant = q <= alpha

    return FdrResult(
        q_values=[float(x) for x in q],
        significant=[bool(x) for x in significant],
        alpha=float(alpha),
        n_tests=m,
        n_significant=int(significant.sum()),
    )


__all__ = [
    "DEFAULT_FDR_ALPHA",
    "FdrResult",
    "benjamini_hochberg",
    "sharpe_p_value",
]
