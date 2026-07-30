"""The sweep persists the winning run's return moments so the Deflated Sharpe
can drop the Gaussian-null assumption. Covers the extraction helper directly
(no vectorbt needed — the equity curve is supplied)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from quantrace.sweep import _winner_moments


def _result_with_equity(returns: np.ndarray):
    equity = pd.Series(100.0 * np.cumprod(1.0 + returns))
    return SimpleNamespace(equity_curve=equity)


def test_returns_none_without_equity():
    assert _winner_moments(SimpleNamespace(equity_curve=None)) == (None, None)


def test_returns_none_for_too_short_curve():
    assert _winner_moments(_result_with_equity(np.array([0.01, 0.02]))) == (None, None)


def test_recovers_near_gaussian_moments():
    rng = np.random.default_rng(0)
    skew, kurt = _winner_moments(_result_with_equity(rng.normal(0.0005, 0.01, 2000)))
    assert skew is not None and kurt is not None
    assert abs(skew) < 0.3          # ~0 for normal
    assert 2.5 < kurt < 3.6         # ~3 (non-excess) for normal


def test_detects_fat_tails():
    rng = np.random.default_rng(1)
    # Student-t-ish: occasional large shocks → high kurtosis.
    base = rng.normal(0.0004, 0.008, 2000)
    base[::100] -= 0.06  # negative shocks → negative skew, fat tails
    skew, kurt = _winner_moments(_result_with_equity(base))
    assert kurt > 4.0
    assert skew < 0.0
