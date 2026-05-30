"""Tests für quantrace.sweep — Parameter-Sweep über Strategie-Parameterkombinationen."""

from __future__ import annotations

import pytest

from quantrace.models import BacktestConfig, StrategySpec, Timeframe
from quantrace.sweep import SweepResult, _generate_param_grid, sweep

# ---------------------------------------------------------------------------
# Unit: _generate_param_grid
# ---------------------------------------------------------------------------


class TestParamGrid:
    def test_single_param(self):
        grid = _generate_param_grid({"fast": [10, 20, 50]})
        assert len(grid) == 3
        assert grid[0] == {"fast": 10}
        assert grid[2] == {"fast": 50}

    def test_two_params_cartesian(self):
        grid = _generate_param_grid({"fast": [10, 20], "slow": [100, 200]})
        assert len(grid) == 4
        # Sorted keys → fast before slow
        assert {"fast": 10, "slow": 100} in grid
        assert {"fast": 20, "slow": 200} in grid

    def test_empty_param_space(self):
        grid = _generate_param_grid({})
        assert grid == [{}]

    def test_single_value_per_param(self):
        grid = _generate_param_grid({"fast": [20], "slow": [100]})
        assert len(grid) == 1
        assert grid[0] == {"fast": 20, "slow": 100}

    def test_three_params(self):
        grid = _generate_param_grid({"a": [1, 2], "b": [3, 4], "c": [5]})
        assert len(grid) == 4  # 2 × 2 × 1


# ---------------------------------------------------------------------------
# Integration: sweep() mit synthetischen Daten
# ---------------------------------------------------------------------------


class TestSweep:
    def _make_spec(self, param_space: dict) -> StrategySpec:
        return StrategySpec(
            strategy_id="sma_sweep_test",
            name="SMA Sweep Test",
            class_path="strategies.templates.sma_crossover:SmaCrossover",
            strategy_class="trend_following",
            universe="synthetic",
            timeframe=Timeframe.DAILY,
            params={"fast": 20, "slow": 100},
            param_space=param_space,
        )

    def test_sweep_basic(self, synthetic_md):
        """Sweep mit 2×2 Kombinationen soll 4 Runs liefern."""
        spec = self._make_spec({"fast": [10, 20], "slow": [100, 200]})
        result = sweep(spec, synthetic_md)

        assert isinstance(result, SweepResult)
        assert result.total_combinations == 4
        assert result.completed == 4
        assert result.failed == 0
        assert len(result.runs) == 4
        assert result.best_params  # Nicht leer
        assert result.rank_by == "sharpe"

    def test_sweep_sorted_descending(self, synthetic_md):
        """Runs sollen nach Sharpe absteigend sortiert sein."""
        spec = self._make_spec({"fast": [10, 20, 50], "slow": [100]})
        result = sweep(spec, synthetic_md, rank_by="sharpe")

        sharpes = [r.result.sharpe for r in result.runs]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_sweep_rank_by_cagr(self, synthetic_md):
        """Sweep kann auch nach CAGR ranken."""
        spec = self._make_spec({"fast": [10, 50], "slow": [100]})
        result = sweep(spec, synthetic_md, rank_by="cagr")

        assert result.rank_by == "cagr"
        cagrs = [r.result.cagr for r in result.runs]
        assert cagrs == sorted(cagrs, reverse=True)

    def test_sweep_rank_by_max_drawdown(self, synthetic_md):
        """max_drawdown: niedrigerer Wert (näher 0) ist besser → ascending."""
        spec = self._make_spec({"fast": [10, 50], "slow": [100]})
        result = sweep(spec, synthetic_md, rank_by="max_drawdown")

        dds = [r.result.max_drawdown for r in result.runs]
        # Lower-is-better → aufsteigend sortiert (weniger negativ = besser)
        assert dds == sorted(dds)

    def test_sweep_empty_param_space_raises(self, synthetic_md):
        """Leerer param_space muss ValueError werfen."""
        spec = self._make_spec({})
        with pytest.raises(ValueError, match="param_space"):
            sweep(spec, synthetic_md)

    def test_sweep_single_combination(self, synthetic_md):
        """Ein einzelner Punkt im Grid funktioniert."""
        spec = self._make_spec({"fast": [20], "slow": [100]})
        result = sweep(spec, synthetic_md)

        assert result.total_combinations == 1
        assert result.completed == 1
        assert result.best_params == {"fast": 20, "slow": 100}

    def test_best_run_property(self, synthetic_md):
        """best_run soll den ersten Run zurückgeben."""
        spec = self._make_spec({"fast": [10, 20], "slow": [100]})
        result = sweep(spec, synthetic_md)

        assert result.best_run is not None
        assert result.best_run.params == result.best_params

    def test_sweep_preserves_backtest_config(self, synthetic_md):
        """Custom BacktestConfig wird an alle Runs weitergegeben."""
        spec = self._make_spec({"fast": [10, 20], "slow": [100]})
        custom_config = BacktestConfig(cash=50_000, fees_bps=5.0)
        result = sweep(spec, synthetic_md, config=custom_config)

        for run in result.runs:
            assert run.result.config.cash == 50_000
            assert run.result.config.fees_bps == 5.0
