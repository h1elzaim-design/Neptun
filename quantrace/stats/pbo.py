"""Probability of Backtest Overfitting via CSCV.

Reference
---------
Bailey, D. H., Borwein, J. M., López de Prado, M. & Zhu, Q. J. (2017).
    "The Probability of Backtest Overfitting."
    Journal of Computational Finance, 20(4).

Idea
----
A parameter sweep selects the in-sample (IS) best of N configurations. If
that selection carries no information, the winner's *out-of-sample* (OOS)
rank among its N peers is uniform — a coin flip whether it lands in the top
or bottom half. CSCV (Combinatorially Symmetric Cross-Validation) measures
exactly that:

1. Split the T×N matrix of per-period trial returns into S equal, contiguous,
   chronological blocks.
2. For every combination of S/2 blocks as IS (the complement is OOS),
   compute each trial's Sharpe on IS and on OOS.
3. Pick the IS winner n*; record its relative OOS rank ω ∈ (0, 1) and the
   logit λ = ln(ω / (1 − ω)).

**PBO** is the fraction of combinations where the IS winner lands in the
bottom half OOS (λ ≤ 0). PBO ≈ 0.5 means selection is pure noise; a serious
research process wants it well below ~0.2 before believing a sweep winner.

Because every block appears equally often in IS and OOS, the estimate is
symmetric and uses each observation T·C(S−2, S/2−1) times — far more sample
efficiency than a single train/test split, without look-ahead: blocks stay
chronologically intact, only their *assignment* is combinatorial.

Complexity
----------
Per-block sums and sums-of-squares are precomputed once (S×N), so each of
the C(S, S/2) combinations costs two (C×S) @ (S×N) matmuls instead of a pass
over the raw T×N matrix — S=16 (12 870 combinations) with hundreds of trials
runs in well under a second. Combinations beyond ``max_combinations`` are
subsampled deterministically (seeded RNG).

Determinism
-----------
Fully deterministic for C(S, S/2) ≤ max_combinations; otherwise deterministic
given the seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

DEFAULT_N_BLOCKS = 16
DEFAULT_MAX_COMBINATIONS = 20_000
DEFAULT_SEED = 7

# Guard against meaningless estimates: with fewer than 2 trials there is no
# selection, and tiny blocks make the per-block Sharpe estimates pure noise.
MIN_TRIALS = 2
MIN_OBS_PER_BLOCK = 4


@dataclass(frozen=True, slots=True)
class PboResult:
    """Output of :func:`probability_of_backtest_overfitting`.

    Attributes
    ----------
    pbo:
        P[IS winner ranks in the bottom OOS half] across combinations.
    n_trials, n_obs:
        Dimensions of the evaluated matrix (after trimming T to S·⌊T/S⌋).
    n_blocks, n_combinations:
        CSCV configuration; ``n_combinations`` is the number actually
        evaluated (subsampled if C(S, S/2) exceeded the cap).
    prob_oos_loss:
        Fraction of combinations where the IS winner's OOS Sharpe is < 0 —
        "how often would the selected config have *lost money* out of
        sample?"
    logit_mean, logit_median:
        Location of the λ distribution (positive = selection skill).
    degradation_slope, degradation_intercept:
        OLS fit OOS ≈ a + b·IS over the winner's (IS, OOS) Sharpe pairs.
        Slopes near 0 (or negative) mean IS performance does not carry OOS.
    oos_sharpe_mean:
        Mean OOS Sharpe (per period) of the IS winner across combinations.
    """

    pbo: float
    n_trials: int
    n_obs: int
    n_blocks: int
    n_combinations: int
    prob_oos_loss: float
    logit_mean: float
    logit_median: float
    degradation_slope: float
    degradation_intercept: float
    oos_sharpe_mean: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "pbo": self.pbo,
            "n_trials": self.n_trials,
            "n_obs": self.n_obs,
            "n_blocks": self.n_blocks,
            "n_combinations": self.n_combinations,
            "prob_oos_loss": self.prob_oos_loss,
            "logit_mean": self.logit_mean,
            "logit_median": self.logit_median,
            "degradation_slope": self.degradation_slope,
            "degradation_intercept": self.degradation_intercept,
            "oos_sharpe_mean": self.oos_sharpe_mean,
            "method": "cscv",
        }


def _combination_masks(
    n_blocks: int,
    max_combinations: int,
    seed: int,
) -> np.ndarray:
    """(C × S) 0/1 matrix — one row per IS block selection of size S/2."""
    half = n_blocks // 2
    total = math.comb(n_blocks, half)
    if total <= max_combinations:
        masks = np.zeros((total, n_blocks), dtype=float)
        for i, combo in enumerate(combinations(range(n_blocks), half)):
            masks[i, list(combo)] = 1.0
        return masks
    # Deterministic subsample of the combination space.
    rng = np.random.default_rng(seed)
    masks = np.zeros((max_combinations, n_blocks), dtype=float)
    for i in range(max_combinations):
        masks[i, rng.choice(n_blocks, size=half, replace=False)] = 1.0
    return masks


def _sharpe_from_block_sums(
    masks: np.ndarray,
    block_sum: np.ndarray,
    block_sumsq: np.ndarray,
    obs_per_block: int,
) -> np.ndarray:
    """Per-period Sharpe (C × N) of each trial over the blocks a mask selects."""
    n = masks.sum(axis=1, keepdims=True) * obs_per_block  # (C, 1)
    s = masks @ block_sum  # (C, N)
    ss = masks @ block_sumsq  # (C, N)
    mean = s / n
    var = (ss - n * mean**2) / (n - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(var > 1e-24, mean / np.sqrt(np.maximum(var, 1e-24)), 0.0)
    return sr


def probability_of_backtest_overfitting(
    trial_returns: np.ndarray,
    *,
    n_blocks: int = DEFAULT_N_BLOCKS,
    max_combinations: int = DEFAULT_MAX_COMBINATIONS,
    seed: int = DEFAULT_SEED,
) -> PboResult:
    """CSCV Probability of Backtest Overfitting for a sweep's trial matrix.

    Parameters
    ----------
    trial_returns:
        (T × N) matrix — per-period returns of the N competing configurations
        over the *same* T periods (columns must be time-aligned; that is what
        makes the block assignment leakage-free).
    n_blocks:
        S — number of chronological blocks. Must be even and ≥ 4. Bailey et
        al. use 16; the effective sample of the estimate is C(S, S/2)
        combinations. Automatically reduced (in steps of 2) when T is too
        short for ``MIN_OBS_PER_BLOCK`` observations per block.
    max_combinations:
        Cap on evaluated combinations (deterministic subsample beyond it).
    seed:
        RNG seed for the (rare) subsampling path.

    Raises
    ------
    ValueError
        Fewer than 2 trials, non-2D input, or T too short for even the
        minimal S=4 split.
    """
    m = np.asarray(trial_returns, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"trial_returns must be 2-D (T × N), got shape {m.shape}")
    t_obs, n_trials = m.shape
    if n_trials < MIN_TRIALS:
        raise ValueError(f"need ≥ {MIN_TRIALS} trials, got {n_trials}")
    if n_blocks < 4 or n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even and ≥ 4")
    if not np.isfinite(m).all():
        raise ValueError("trial_returns contains non-finite values")

    # Shrink S until each block holds at least MIN_OBS_PER_BLOCK observations.
    s = n_blocks
    while s > 4 and t_obs // s < MIN_OBS_PER_BLOCK:
        s -= 2
    obs_per_block = t_obs // s
    if obs_per_block < MIN_OBS_PER_BLOCK:
        raise ValueError(
            f"T={t_obs} too short for CSCV: needs ≥ {4 * MIN_OBS_PER_BLOCK} observations"
        )

    # Trim the tail remainder so blocks are equal-sized, then aggregate.
    trimmed = m[: s * obs_per_block]
    blocks = trimmed.reshape(s, obs_per_block, n_trials)
    block_sum = blocks.sum(axis=1)  # (S, N)
    block_sumsq = (blocks**2).sum(axis=1)  # (S, N)

    masks = _combination_masks(s, max_combinations, seed)
    sr_is = _sharpe_from_block_sums(masks, block_sum, block_sumsq, obs_per_block)
    sr_oos = _sharpe_from_block_sums(1.0 - masks, block_sum, block_sumsq, obs_per_block)

    winner = sr_is.argmax(axis=1)  # (C,)
    rows = np.arange(masks.shape[0])
    winner_is = sr_is[rows, winner]
    winner_oos = sr_oos[rows, winner]

    # Relative OOS rank of the IS winner among all N trials, in (0, 1).
    ranks = (sr_oos <= winner_oos[:, None]).sum(axis=1) / (n_trials + 1.0)
    ranks = np.clip(ranks, 1e-9, 1.0 - 1e-9)
    logits = np.log(ranks / (1.0 - ranks))

    # OLS degradation fit: OOS ≈ a + b·IS across combinations.
    var_is = float(winner_is.var())
    if var_is > 1e-18:
        slope = float(np.cov(winner_is, winner_oos, ddof=0)[0, 1] / var_is)
        intercept = float(winner_oos.mean() - slope * winner_is.mean())
    else:
        slope, intercept = 0.0, float(winner_oos.mean())

    return PboResult(
        pbo=float((logits <= 0.0).mean()),
        n_trials=int(n_trials),
        n_obs=int(s * obs_per_block),
        n_blocks=int(s),
        n_combinations=int(masks.shape[0]),
        prob_oos_loss=float((winner_oos < 0.0).mean()),
        logit_mean=float(logits.mean()),
        logit_median=float(np.median(logits)),
        degradation_slope=slope,
        degradation_intercept=intercept,
        oos_sharpe_mean=float(winner_oos.mean()),
    )
