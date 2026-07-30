"""CSCV Probability of Backtest Overfitting — noise vs genuine edge."""

from __future__ import annotations

import numpy as np
import pytest

from quantrace.stats.pbo import probability_of_backtest_overfitting


def _noise_matrix(t: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, (t, n))


def test_pure_noise_pbo_near_half():
    # No trial has an edge — the IS winner's OOS rank is a coin flip.
    m = _noise_matrix(1024, 40, seed=1)
    res = probability_of_backtest_overfitting(m, n_blocks=8)
    assert 0.3 <= res.pbo <= 0.7
    assert res.n_trials == 40


def test_genuine_edge_pbo_low():
    # One trial with a big real edge dominates IS *and* OOS.
    m = _noise_matrix(1024, 20, seed=2)
    m[:, 0] += 0.004  # ≈ Sharpe 6 annualised — unambiguous skill
    res = probability_of_backtest_overfitting(m, n_blocks=8)
    assert res.pbo < 0.1
    assert res.prob_oos_loss < 0.1
    assert res.oos_sharpe_mean > 0.0
    assert res.logit_median > 0.0


def test_noise_winner_oos_sharpe_near_zero():
    m = _noise_matrix(2048, 50, seed=3)
    res = probability_of_backtest_overfitting(m, n_blocks=8)
    # Selection over noise: the winner's OOS Sharpe collapses toward 0.
    assert abs(res.oos_sharpe_mean) < 0.05  # per-period units


def test_deterministic():
    m = _noise_matrix(512, 12, seed=4)
    a = probability_of_backtest_overfitting(m)
    b = probability_of_backtest_overfitting(m)
    assert a == b


def test_block_count_auto_shrinks_for_short_t():
    # T=40 cannot host 16 blocks of ≥4 obs → S shrinks (but stays even ≥ 4).
    m = _noise_matrix(40, 6, seed=5)
    res = probability_of_backtest_overfitting(m, n_blocks=16)
    assert res.n_blocks < 16
    assert res.n_blocks % 2 == 0
    assert res.n_blocks >= 4
    assert res.n_obs <= 40


def test_combination_cap_subsamples_deterministically():
    m = _noise_matrix(512, 10, seed=6)
    a = probability_of_backtest_overfitting(m, n_blocks=16, max_combinations=500)
    b = probability_of_backtest_overfitting(m, n_blocks=16, max_combinations=500)
    assert a.n_combinations == 500
    assert a == b


def test_rejects_single_trial():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(_noise_matrix(256, 1))


def test_rejects_short_series():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(_noise_matrix(12, 5))


def test_rejects_non_finite():
    m = _noise_matrix(256, 5)
    m[3, 2] = np.nan
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(m)


def test_rejects_odd_blocks():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(_noise_matrix(256, 5), n_blocks=7)


def test_to_dict_is_json_friendly():
    res = probability_of_backtest_overfitting(_noise_matrix(256, 8))
    d = res.to_dict()
    assert d["method"] == "cscv"
    assert set(d) >= {"pbo", "prob_oos_loss", "n_combinations", "degradation_slope"}
    assert all(isinstance(v, (int, float, str)) for v in d.values())
