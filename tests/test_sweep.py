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


# ---------------------------------------------------------------------------
# Statistische Disziplin: DSR + FDR am Sweep-Ergebnis
# ---------------------------------------------------------------------------


class TestSweepDiscipline:
    def _make_spec(self, param_space: dict) -> StrategySpec:
        return StrategySpec(
            strategy_id="sma_discipline_test",
            name="SMA Discipline Test",
            class_path="strategies.templates.sma_crossover:SmaCrossover",
            strategy_class="trend_following",
            universe="synthetic",
            timeframe=Timeframe.DAILY,
            params={"fast": 20, "slow": 100},
            param_space=param_space,
        )

    def test_winner_dsr_and_fdr_populated(self, synthetic_md):
        """Ein Multi-Combo-Sweep trägt DSR, E[max] und BH-FDR am Ergebnis."""
        spec = self._make_spec({"fast": [10, 20, 50], "slow": [100, 200]})
        result = sweep(spec, synthetic_md)

        assert result.best_n_obs is not None and result.best_n_obs > 100
        assert result.best_dsr is not None and 0.0 <= result.best_dsr <= 1.0
        assert result.best_psr is not None and 0.0 <= result.best_psr <= 1.0
        assert result.expected_max_sharpe_annual is not None
        assert result.expected_max_sharpe_annual >= 0.0
        assert result.best_skew is not None and result.best_kurt is not None

        fdr = result.fdr
        assert fdr is not None
        assert fdr["method"] == "benjamini_hochberg"
        assert fdr["n_tests"] == result.completed == 6
        assert fdr["n_untested"] == 0  # synthetische Daten: jede Combo testbar
        assert 0 <= fdr["n_significant"] <= fdr["n_tests"]
        assert fdr["winner_q_value"] == result.runs[0].q_value

        for run in result.runs:
            assert run.p_value is not None and 0.0 <= run.p_value <= 1.0
            assert run.q_value is not None
            assert run.q_value >= run.p_value - 1e-12  # BH verkleinert nie
            assert run.fdr_significant == (run.q_value <= fdr["alpha"])

    def test_selection_stats_opt_out_for_inner_sweeps(self, synthetic_md):
        """selection_stats=False (Walk-Forward-Inner-Sweeps): Ranking läuft,
        aber keine verworfene DSR/FDR-Rechnung."""
        spec = self._make_spec({"fast": [10, 20], "slow": [100]})
        result = sweep(spec, synthetic_md, selection_stats=False)

        assert result.completed == 2 and result.best_params
        assert result.best_dsr is None and result.fdr is None
        assert result.best_n_obs is None
        assert result.pbo is None and result.best_bootstrap is None
        assert all(r.p_value is None for r in result.runs)

    def test_pbo_and_bootstrap_populated(self, synthetic_md):
        """≥4 Combos: CSCV-PBO + Stationary-Bootstrap des Winners am Ergebnis."""
        spec = self._make_spec({"fast": [10, 20, 50], "slow": [100, 200]})
        result = sweep(spec, synthetic_md)

        pbo = result.pbo
        assert pbo is not None
        assert pbo["method"] == "cscv"
        assert 0.0 <= pbo["pbo"] <= 1.0
        assert pbo["n_trials"] + pbo["n_excluded_trials"] == result.completed
        assert pbo["n_blocks"] % 2 == 0 and pbo["n_blocks"] >= 4
        assert pbo["n_combinations"] >= 2
        assert 0.0 <= pbo["prob_oos_loss"] <= 1.0

        boot = result.best_bootstrap
        assert boot is not None
        sharpe = boot["sharpe"]
        assert sharpe["method"] == "stationary_bootstrap"
        assert sharpe["ci_low"] <= sharpe["sharpe_annual"] <= sharpe["ci_high"]
        assert 0.0 < sharpe["p_value"] <= 1.0
        dd = boot["max_drawdown"]
        assert dd["max_dd_observed"] <= 0.0
        assert 0.0 <= dd["prob_worse_than_observed"] <= 1.0
        # Quantile monoton: tieferes Quantil = tieferer Drawdown.
        qs = sorted(dd["quantiles"], key=float)
        vals = [dd["quantiles"][q] for q in qs]
        assert vals == sorted(vals)

    def test_pbo_needs_at_least_four_combos(self, synthetic_md):
        """N=2: Bootstrap des Winners ja, PBO (Rang-Statistik) nein."""
        spec = self._make_spec({"fast": [10, 20], "slow": [100]})
        result = sweep(spec, synthetic_md)

        assert result.pbo is None
        assert result.best_bootstrap is not None

    def test_winner_equity_path_persisted(self, synthetic_md):
        """Der Winner behält seinen Equity-Pfad ([{date, value}]) im JSON —
        die Basis für Portfolio-Uniqueness; die übrigen Runs bleiben schlank."""
        import json

        spec = self._make_spec({"fast": [10, 20], "slow": [100]})
        result = sweep(spec, synthetic_md)

        eq = result.best_equity
        assert eq is not None and len(eq) > 100
        dates = [p["date"] for p in eq]
        assert dates == sorted(dates)
        assert all(p["value"] > 0 for p in eq)

        payload = json.loads(result.model_dump_json())
        assert len(payload["best_equity"]) == len(eq)
        assert "equity_curve" not in payload["runs"][0]["result"]

    def test_selection_stats_off_skips_equity_path(self, synthetic_md):
        spec = self._make_spec({"fast": [10, 20], "slow": [100]})
        result = sweep(spec, synthetic_md, selection_stats=False)
        assert result.best_equity is None

    def test_single_combo_has_no_multiplicity_stats(self, synthetic_md):
        """N=1: kein Selection-Set → DSR/FDR bleiben None (undefiniert, nicht 0)."""
        spec = self._make_spec({"fast": [20], "slow": [100]})
        result = sweep(spec, synthetic_md)

        assert result.best_dsr is None
        assert result.fdr is None
        assert result.runs[0].p_value is None

    def test_discipline_fields_survive_json_roundtrip(self, synthetic_md):
        """Persistiertes JSON trägt die neuen Skalare (Equity bleibt draußen)."""
        import json

        spec = self._make_spec({"fast": [10, 20], "slow": [100]})
        result = sweep(spec, synthetic_md)
        payload = json.loads(result.model_dump_json())

        assert payload["best_dsr"] == result.best_dsr
        assert payload["best_n_obs"] == result.best_n_obs
        assert payload["fdr"]["n_tests"] == 2
        assert payload["best_bootstrap"]["sharpe"]["p_value"] > 0.0
        assert "equity_curve" not in payload["runs"][0]["result"]
        assert payload["runs"][0]["p_value"] is not None

    def test_old_json_without_discipline_fields_still_parses(self, synthetic_md):
        """Backward-Compat: alte SweepResult-JSONs (ohne die neuen Felder) laden."""
        import json

        spec = self._make_spec({"fast": [10, 20], "slow": [100]})
        result = sweep(spec, synthetic_md)
        payload = json.loads(result.model_dump_json())
        for key in ("best_n_obs", "best_dsr", "best_psr", "expected_max_sharpe_annual", "fdr"):
            payload.pop(key)
        for run in payload["runs"]:
            for key in ("p_value", "q_value", "fdr_significant"):
                run.pop(key)

        old = SweepResult.model_validate(payload)
        assert old.best_dsr is None
        assert old.fdr is None
        assert old.runs[0].p_value is None
