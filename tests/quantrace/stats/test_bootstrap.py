"""Stationary bootstrap — resampling core, Sharpe CI, drawdown distribution."""

from __future__ import annotations

import numpy as np
import pytest

from quantrace.stats.bootstrap import (
    bootstrap_drawdown_distribution,
    bootstrap_sharpe_ci,
    default_block_length,
    stationary_bootstrap_indices,
)


def _returns(mu: float, sigma: float, n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n)


# -----------------------------------------------------------------------------
# Resampling core
# -----------------------------------------------------------------------------


def test_indices_shape_and_range():
    idx = stationary_bootstrap_indices(100, n_resamples=50, avg_block_len=5)
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_indices_deterministic_given_seed():
    a = stationary_bootstrap_indices(
        200, n_resamples=10, avg_block_len=8, rng=np.random.default_rng(42)
    )
    b = stationary_bootstrap_indices(
        200, n_resamples=10, avg_block_len=8, rng=np.random.default_rng(42)
    )
    np.testing.assert_array_equal(a, b)


def test_indices_preserve_consecutive_runs():
    # With a huge expected block length, most steps continue the previous
    # block: index[t+1] == (index[t] + 1) % n almost everywhere.
    n = 500
    idx = stationary_bootstrap_indices(
        n, n_resamples=20, avg_block_len=250, rng=np.random.default_rng(1)
    )
    consecutive = (np.diff(idx, axis=1) % n) == 1
    assert consecutive.mean() > 0.95


def test_indices_block_starts_are_uniformish():
    # With block length 1 the scheme degenerates to i.i.d. resampling —
    # every index should appear with roughly equal frequency.
    idx = stationary_bootstrap_indices(
        50, n_resamples=2000, avg_block_len=1, rng=np.random.default_rng(2)
    )
    counts = np.bincount(idx.ravel(), minlength=50)
    freq = counts / counts.sum()
    assert abs(freq.max() - freq.min()) < 0.01


def test_default_block_length_scales_with_t():
    assert default_block_length(1000) == pytest.approx(10.0)
    assert default_block_length(8) == 2.0
    assert default_block_length(2) >= 1.0


def test_indices_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(1, n_resamples=10)
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(100, n_resamples=0)
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(100, n_resamples=10, avg_block_len=0.5)


# -----------------------------------------------------------------------------
# Sharpe CI
# -----------------------------------------------------------------------------


def test_sharpe_ci_brackets_point_estimate():
    r = _returns(0.0006, 0.01, 1500)
    res = bootstrap_sharpe_ci(r, n_resamples=500)
    assert res.ci_low < res.sharpe_annual < res.ci_high
    assert res.confidence == 0.95
    assert res.n_obs == 1500


def test_sharpe_ci_zero_drift_straddles_zero():
    r = _returns(0.0, 0.01, 2000, seed=11)
    res = bootstrap_sharpe_ci(r, n_resamples=800)
    assert res.ci_low < 0.0 < res.ci_high
    assert res.p_value > 0.05  # cannot reject SR ≤ 0


def test_sharpe_ci_strong_edge_rejects_null():
    # SR ≈ (0.002/0.01)·√252 ≈ 3.2 over 6 years — unambiguous.
    r = _returns(0.002, 0.01, 1500, seed=5)
    res = bootstrap_sharpe_ci(r, n_resamples=800)
    assert res.ci_low > 0.0
    assert res.p_value < 0.01


def test_sharpe_ci_deterministic():
    r = _returns(0.0005, 0.012, 600)
    a = bootstrap_sharpe_ci(r, n_resamples=300, seed=9)
    b = bootstrap_sharpe_ci(r, n_resamples=300, seed=9)
    assert a == b


def test_sharpe_ci_rejects_short_series():
    with pytest.raises(ValueError):
        bootstrap_sharpe_ci([0.01] * 5)


def test_sharpe_ci_rejects_bad_confidence():
    with pytest.raises(ValueError):
        bootstrap_sharpe_ci(_returns(0.0, 0.01, 100), confidence=0.4)


def test_sharpe_ci_to_dict_roundtrip():
    res = bootstrap_sharpe_ci(_returns(0.0005, 0.01, 300), n_resamples=200)
    d = res.to_dict()
    assert d["method"] == "stationary_bootstrap"
    assert d["ci_low"] == res.ci_low
    assert d["n_resamples"] == 200


# -----------------------------------------------------------------------------
# Drawdown distribution
# -----------------------------------------------------------------------------


def test_drawdown_quantiles_ordered_and_negative():
    r = _returns(0.0004, 0.012, 1000)
    res = bootstrap_drawdown_distribution(r, n_resamples=400)
    qs = sorted(res.quantiles)
    vals = [res.quantiles[q] for q in qs]
    # Higher quantile → shallower (less negative) drawdown.
    assert vals == sorted(vals)
    assert all(v <= 0.0 for v in vals)
    assert res.max_dd_observed <= 0.0
    assert 0.0 <= res.prob_worse_than_observed <= 1.0


def test_drawdown_observed_matches_direct_computation():
    r = np.array([0.10, -0.20, 0.05, -0.10, 0.15, 0.02, -0.03, 0.01])
    res = bootstrap_drawdown_distribution(r, n_resamples=50)
    equity = np.cumprod(1.0 + r)
    dd = (equity / np.maximum.accumulate(equity) - 1.0).min()
    assert res.max_dd_observed == pytest.approx(float(dd))


def test_drawdown_deterministic():
    r = _returns(0.0002, 0.015, 500)
    a = bootstrap_drawdown_distribution(r, n_resamples=200, seed=4)
    b = bootstrap_drawdown_distribution(r, n_resamples=200, seed=4)
    assert a.quantiles == b.quantiles
    assert a.prob_worse_than_observed == b.prob_worse_than_observed
