"""Feature engineering for regime detection.

A regime is characterised by *trend* (are returns drifting up or down?) and
*risk* (how violent are the moves?). We feed the HMM two trailing, fully causal
features so the resulting regimes are persistent and interpretable rather than
flickering on daily noise:

    trend  — annualised mean log-return over a trailing window
    vol    — annualised realised volatility over the same window

Both are computed from a single benchmark price series. For a multi-asset
universe the benchmark is the equal-weight average price (see
:func:`benchmark_series`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252.0


def benchmark_series(prices: pd.Series | pd.DataFrame) -> pd.Series:
    """Collapse prices to a single benchmark series.

    A ``Series`` is returned as-is. A ``DataFrame`` (columns = symbols) becomes
    the equal-weight mean across symbols — a cheap market proxy that is robust
    to individual-symbol gaps.
    """
    if isinstance(prices, pd.Series):
        return prices.astype(float)
    return prices.astype(float).mean(axis=1)


def regime_features(
    prices: pd.Series | pd.DataFrame,
    *,
    window: int = 21,
) -> pd.DataFrame:
    """Trailing trend + realised-vol features, indexed by date.

    The first ``window`` rows are dropped (insufficient history). Both columns
    are annualised so their scale is intuitive (e.g. vol ≈ 0.15 is a calm 15 %).
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    bench = benchmark_series(prices)
    log_ret = np.log(bench).diff()

    trend = log_ret.rolling(window).mean() * _TRADING_DAYS
    vol = log_ret.rolling(window).std(ddof=1) * np.sqrt(_TRADING_DAYS)

    feats = pd.DataFrame({"trend": trend, "vol": vol})
    return feats.replace([np.inf, -np.inf], np.nan).dropna()
