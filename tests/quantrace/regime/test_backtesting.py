"""Tests for quantrace.regime.backtesting — regime-conditioned performance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantrace.regime.backtesting import (
    _dd_stats,
    _risk_metrics,
    regime_conditioned_metrics,
)


def _make_prices(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    n_syms = 3
    log_rets = rng.normal(0.0004, 0.01, (n, n_syms))
    prices = np.exp(np.cumsum(log_rets, axis=0)) * 100
    return pd.DataFrame(prices, index=dates, columns=["A", "B", "C"])


def _make_equity(n: int = 600, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = rng.normal(0.0003, 0.009, n)
    equity = 100_000 * np.exp(np.cumsum(rets))
    return pd.Series(equity, index=dates)


class TestRiskMetrics:
    def test_positive_return_positive_sharpe(self) -> None:
        rng = np.random.default_rng(1)
        # Biased positive returns so Sharpe > 0.
        rets = pd.Series(rng.normal(0.001, 0.005, 200))
        sharpe, sortino, cagr, vol = _risk_metrics(rets)
        assert sharpe > 0
        assert cagr > 0
        assert vol >= 0

    def test_zero_returns_zero_sharpe(self) -> None:
        rets = pd.Series([0.0] * 50)
        sharpe, sortino, cagr, vol = _risk_metrics(rets)
        assert sharpe == 0.0

    def test_all_finite(self) -> None:
        rng = np.random.default_rng(7)
        rets = pd.Series(rng.normal(0, 0.01, 200))
        for v in _risk_metrics(rets):
            assert np.isfinite(v)

    def test_empty_series(self) -> None:
        result = _risk_metrics(pd.Series([], dtype=float))
        assert result == (0.0, 0.0, 0.0, 0.0)


class TestDdStats:
    def test_flat_equity_no_drawdown(self) -> None:
        eq = pd.Series([100.0] * 50)
        max_dd, avg_dd, uw_pct = _dd_stats(eq)
        assert max_dd == pytest.approx(0.0, abs=1e-9)

    def test_known_drawdown(self) -> None:
        eq = pd.Series([100.0, 110.0, 88.0, 99.0])
        max_dd, _avg, _uw = _dd_stats(eq)
        expected = 88 / 110 - 1  # -0.2
        assert max_dd == pytest.approx(expected, abs=1e-6)

    def test_short_series(self) -> None:
        result = _dd_stats(pd.Series([100.0]))
        assert result == (0.0, 0.0, 0.0)


class TestRegimeConditionedMetrics:
    def test_returns_dict_with_rows(self) -> None:
        equity = _make_equity(600)
        prices = _make_prices(600)
        result = regime_conditioned_metrics(equity, prices, n_states=3, feature_window=21)
        assert result is not None
        assert "rows" in result
        assert len(result["rows"]) > 0

    def test_rows_cover_all_states(self) -> None:
        equity = _make_equity(600)
        prices = _make_prices(600)
        result = regime_conditioned_metrics(equity, prices, n_states=3, feature_window=21)
        assert result is not None
        # Should have at most n_states rows (≥5 days each required)
        assert len(result["rows"]) <= 3

    def test_pct_time_sums_to_approx_one(self) -> None:
        equity = _make_equity(700)
        prices = _make_prices(700)
        result = regime_conditioned_metrics(equity, prices, n_states=3, feature_window=21)
        assert result is not None
        total = sum(r["pct_time"] for r in result["rows"])
        assert total == pytest.approx(1.0, abs=0.02)

    def test_all_metrics_finite(self) -> None:
        equity = _make_equity(600)
        prices = _make_prices(600)
        result = regime_conditioned_metrics(equity, prices, n_states=3, feature_window=21)
        assert result is not None
        for row in result["rows"]:
            for k in ("sharpe", "sortino", "cagr", "vol", "max_drawdown"):
                assert np.isfinite(row[k]), f"{k} is not finite"

    def test_too_little_data_returns_none(self) -> None:
        equity = _make_equity(30)
        prices = _make_prices(30)
        result = regime_conditioned_metrics(equity, prices, n_states=3, feature_window=21)
        assert result is None

    def test_n_states_stored(self) -> None:
        equity = _make_equity(500)
        prices = _make_prices(500)
        result = regime_conditioned_metrics(equity, prices, n_states=2, feature_window=21)
        assert result is not None
        assert result["n_states"] == 2

    def test_metadata_fields_present(self) -> None:
        equity = _make_equity(600)
        prices = _make_prices(600)
        result = regime_conditioned_metrics(equity, prices)
        assert result is not None
        assert "n_states" in result
        assert "feature_window" in result
        for row in result["rows"]:
            assert "label" in row
            assert "n_days" in row
            assert "period_start" in row
            assert "period_end" in row
