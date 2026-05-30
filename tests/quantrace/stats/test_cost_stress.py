"""Unit tests for quantrace.stats.cost_stress."""

from __future__ import annotations

import numpy as np
import pytest

from quantrace.stats.cost_stress import apply_cost_stress


class TestInvariance:
    def test_unit_multipliers_leave_returns_unchanged(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.0005, 0.012, size=500)
        res = apply_cost_stress(r, slippage_multiplier=1.0, fee_multiplier=1.0)
        assert np.allclose(np.asarray(res.stressed_returns), r)

    def test_zero_trades_per_period_leaves_returns_unchanged(self):
        rng = np.random.default_rng(43)
        r = rng.normal(0.0005, 0.012, size=300)
        res = apply_cost_stress(
            r, slippage_multiplier=5.0, fee_multiplier=10.0, trades_per_period=0.0,
        )
        assert np.allclose(np.asarray(res.stressed_returns), r)


class TestMonotonicity:
    def test_higher_multipliers_reduce_returns(self):
        rng = np.random.default_rng(44)
        r = rng.normal(0.0008, 0.012, size=1000)
        s1 = apply_cost_stress(r, slippage_multiplier=2.0).stressed_returns
        s5 = apply_cost_stress(r, slippage_multiplier=5.0).stressed_returns
        # element-wise: every period gets a larger drag at 5×
        assert (np.asarray(s1) > np.asarray(s5)).all()

    def test_sharpe_decreases_monotonically(self):
        rng = np.random.default_rng(45)
        r = rng.normal(0.0012, 0.010, size=2000)
        sharpes = []
        for mult in (1.0, 2.0, 3.0, 5.0):
            res = apply_cost_stress(r, slippage_multiplier=mult, fee_multiplier=mult)
            sharpes.append(res.stressed_sharpe_annual)
        assert sharpes == sorted(sharpes, reverse=True)


class TestSurvivalFlag:
    def test_marginal_strategy_collapses(self):
        # Construct a return series with positive but low Sharpe (~0.6)
        rng = np.random.default_rng(46)
        r = rng.normal(0.0003, 0.008, size=2000)
        baseline = apply_cost_stress(
            r, slippage_multiplier=1.0, baseline_slippage_bps=3.0, trades_per_period=1.0,
        )
        stressed = apply_cost_stress(
            r, slippage_multiplier=10.0, baseline_slippage_bps=3.0, trades_per_period=1.0,
        )
        assert baseline.stressed_sharpe_annual > stressed.stressed_sharpe_annual
        # baseline survives, stress kills it
        assert baseline.survives
        assert not stressed.survives

    def test_robust_strategy_survives_high_stress(self):
        rng = np.random.default_rng(47)
        # Very strong returns: huge drift, low vol → easy survive
        r = rng.normal(0.003, 0.006, size=1000)
        stressed = apply_cost_stress(r, slippage_multiplier=10.0)
        assert stressed.survives


class TestDeterminism:
    def test_identical_inputs_produce_identical_outputs(self):
        rng = np.random.default_rng(48)
        r = rng.normal(0.0005, 0.011, size=300)
        a = apply_cost_stress(r, slippage_multiplier=2.0)
        b = apply_cost_stress(r, slippage_multiplier=2.0)
        assert a.stressed_sharpe_annual == b.stressed_sharpe_annual
        assert a.stressed_cagr == b.stressed_cagr
        assert a.stressed_max_drawdown == b.stressed_max_drawdown


class TestValidation:
    def test_rejects_negative_multipliers(self):
        with pytest.raises(ValueError):
            apply_cost_stress([0.01], slippage_multiplier=-0.5)
        with pytest.raises(ValueError):
            apply_cost_stress([0.01], fee_multiplier=-0.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            apply_cost_stress(
                [0.01, 0.02, 0.03],
                trades_per_period=[1.0, 1.0],  # too short
            )

    def test_empty_returns_yield_empty_result(self):
        res = apply_cost_stress([])
        assert res.stressed_returns == []
        assert res.stressed_sharpe_annual == 0.0
        assert not res.survives
