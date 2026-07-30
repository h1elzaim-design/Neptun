"""Execution-lag (same-bar look-ahead) guard for the backtest runner.

A signal derived from bar t's close must not fill at close[t]; the runner
shifts signals forward so the fill lands on the next bar. These tests cover the
pure shift helper and the config contract — neither needs vectorbt.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from quantrace.backtest_runner import _lag_signals
from quantrace.models import BacktestConfig


def test_lag_shifts_signals_forward_one_bar():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    entries = pd.DataFrame({"A": [True, False, False, True, False]}, index=idx)
    exits = pd.DataFrame({"A": [False, True, False, False, True]}, index=idx)

    e2, x2 = _lag_signals(entries, exits, 1)

    # The very first bar can never carry a (lagged) signal — nothing precedes it.
    assert not e2.iloc[0]["A"]
    assert not x2.iloc[0]["A"]
    # A signal generated on bar t fills on bar t+1.
    assert e2["A"].tolist() == [False, True, False, False, True]
    assert x2["A"].tolist() == [False, False, True, False, False]
    # Stays boolean (no object downcast that would break vectorbt).
    assert e2["A"].dtype == bool
    assert x2["A"].dtype == bool


def test_lag_zero_is_passthrough():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    entries = pd.DataFrame({"A": [True, False, True]}, index=idx)
    exits = pd.DataFrame({"A": [False, True, False]}, index=idx)

    e2, x2 = _lag_signals(entries, exits, 0)

    assert e2 is entries
    assert x2 is exits


def test_execution_lag_default_is_one():
    assert BacktestConfig().execution_lag == 1


def test_execution_lag_rejects_negative():
    with pytest.raises(ValidationError):
        BacktestConfig(execution_lag=-1)
