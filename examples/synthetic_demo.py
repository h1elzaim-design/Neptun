#!/usr/bin/env python3
"""Neptun 60-second demo — runs the full pipeline on synthetic data.

No API keys, no network, no real market data. This script fabricates a small
multi-asset price panel in memory, runs an SMA-crossover backtest through the
*real* engine, and prints the resulting metrics. It exists to prove that a
fresh clone works end to end with zero configuration.

    python examples/synthetic_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantrace.models import BacktestConfig, MarketData, Timeframe
from strategies.templates.sma_crossover import SmaCrossover


def make_synthetic_market(seed: int = 42) -> MarketData:
    """Build a MarketData panel with the layout the engine expects.

    Columns are a MultiIndex of (symbol, field) with field in
    {open, high, low, close, volume}; the index is a daily DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", "2024-12-31")
    symbols = ["AAA", "BBB", "CCC"]

    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        # Geometric random walk with a small positive drift.
        daily_ret = rng.normal(0.0003, 0.011, len(idx))
        close = 100 * np.exp(np.cumsum(daily_ret))
        frames[sym] = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.001, len(idx))),
                "high": close * 1.004,
                "low": close * 0.996,
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, len(idx)).astype(float),
            },
            index=idx,
        )

    combined = pd.concat(frames, axis=1)
    combined.columns.names = ["symbol", "field"]

    return MarketData(
        universe="synthetic_demo",
        symbols=symbols,
        timeframe=Timeframe.DAILY,
        start=idx[0].date(),
        end=idx[-1].date(),
        provider="synthetic",
        frame=combined,
    )


def main() -> None:
    try:
        from quantrace.backtest_runner import run_inline
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Could not import the backtest runner. Install the framework first:\n"
            '    pip install -e ".[dev]"\n'
            f"(original error: {exc})"
        ) from exc

    market = make_synthetic_market()
    strategy = SmaCrossover(fast=20, slow=100)

    result = run_inline(
        strategy_id="demo_sma_20_100",
        strategy=strategy,
        data=market,
        config=BacktestConfig(),
    )

    print("Neptun synthetic demo — SMA crossover (20/100) on 3 random-walk assets")
    print(f"  period      : {result.start} .. {result.end}")
    print(f"  total return: {result.total_return:>8.2%}")
    print(f"  CAGR        : {result.cagr:>8.2%}")
    print(f"  Sharpe      : {result.sharpe:>8.2f}")
    print(f"  max drawdown: {result.max_drawdown:>8.2%}")
    print(f"  trades      : {result.trades.n_trades:>8d}")
    print(f"  win rate    : {result.trades.win_rate:>8.2%}")
    print(
        "\nNote: random-walk data has no real edge, so the numbers are not "
        "meaningful — the point is that the pipeline runs."
    )


if __name__ == "__main__":
    main()
