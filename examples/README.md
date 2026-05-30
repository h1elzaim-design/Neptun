# Examples

Runnable, self-contained examples for the Neptun framework. None of them need
an API key, a network connection, or real market data — they fabricate
synthetic data in memory so a fresh clone works immediately.

## `synthetic_demo.py`

Generates a small multi-asset random-walk price panel, runs an SMA-crossover
backtest through the real engine, and prints the metrics.

```bash
python examples/synthetic_demo.py
```

Use it to:
- confirm your install works end to end,
- see the shape of `MarketData` the engine expects, and
- copy the pattern for writing your own offline experiments.

Because the underlying data is a random walk with no real signal, the
performance numbers are not meaningful. The purpose is to demonstrate the
pipeline, not a strategy.
