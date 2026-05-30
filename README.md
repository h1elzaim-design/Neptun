# Neptun

**A framework for building a semi-autonomous quant trading research lab.**

Neptun is the open-source engine behind a research workflow that *researches
autonomously but never trades autonomously*. It gives you the reusable
machinery — data contracts, a backtest runner, a six-dimensional evaluation
score, walk-forward validation, and robust performance statistics — so you can
focus on your own ideas and your own edge.

> Neptun ships the **tools**, not the **recipe**. It contains textbook strategy
> templates and synthetic-data examples. The proprietary strategies, signal
> definitions, scoring thresholds, and research notes that make a lab
> competitive are intentionally *not* part of this repository — they belong in
> your own private space.

---

## What's inside

| Layer | What it does |
|-------|--------------|
| **Data contracts** (`quantrace/models.py`) | Typed, hashable objects every component speaks: `MarketData`, `StrategySpec`, `BacktestConfig`, `BacktestResult`, `EvaluationReport`. |
| **Data loader** (`quantrace/data_agent.py`) | Loads OHLCV via [OpenBB](https://openbb.co) (yfinance by default — no API key needed) and caches to Parquet. |
| **Strategy interface** (`quantrace/strategy.py`) | One small base class. A strategy returns `(entries, exits)` and nothing else, so the engine stays swappable. |
| **Backtest runner** (`quantrace/backtest_runner.py`) | A thin, consistent [vectorbt](https://vectorbt.dev) wrapper. Identical cost/slippage assumptions for every strategy, so comparisons are fair. |
| **Evaluation** (`quantrace/evaluation.py`) | A weighted score across performance, risk, stability, realism, generalization, and simplicity — plus hard guardrails. Weights/targets live in a config you own. |
| **Walk-forward & sweeps** (`quantrace/walk_forward.py`, `quantrace/sweep.py`) | Out-of-sample validation and parameter grids. |
| **Statistics** (`quantrace/stats/`) | Probabilistic & Deflated Sharpe (Bailey & López de Prado), survivorship and cost-stress checks. |
| **Strategy templates** (`strategies/templates/`) | SMA/EMA crossover, MACD, RSI(2), Bollinger, mean reversion, Donchian & ATR breakout, 12-1 momentum, buy & hold. All textbook, all ~20–40 lines. |
| **CLI** (`quantrace/cli.py`) | `fetch`, `backtest`, `sweep`, `walkforward`, `compare`, `report`. |

## Install

```bash
git clone https://github.com/<you>/Neptun.git
cd Neptun

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras: `.[data]` for live OpenBB data fetching, `.[ibkr]` for the
Interactive Brokers adapter.

## 60-second demo (no API keys, no real data)

```bash
python examples/synthetic_demo.py
```

This generates synthetic price data in-memory, runs an SMA-crossover backtest
through the real engine, and prints the metrics. It proves the whole pipeline
works on a fresh clone with zero configuration.

## Real data quickstart

```bash
# Load a public ETF universe via yfinance (no key required)
quantrace fetch --universe us_core_etfs --start 2018-01-01 --end 2024-12-31

# Backtest an SMA crossover
quantrace backtest --strategy sma_crossover --universe us_core_etfs \
    --start 2018-01-01 --end 2024-12-31 --fast 20 --slow 100
```

## Write your own strategy

```python
import pandas as pd
from quantrace.models import MarketData
from quantrace.strategy import Strategy


class MyIdea(Strategy):
    defaults = {"lookback": 50}

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = self.close(data)
        ma = close.rolling(self.params["lookback"]).mean()
        long_now = (close > ma).fillna(False)
        long_prev = long_now.shift(1, fill_value=False)
        entries = long_now & ~long_prev
        exits = ~long_now & long_prev
        return entries, exits
```

That's the whole contract. Hand it to `quantrace.backtest_runner.run_inline`
with some `MarketData` and you get a full `BacktestResult`.

## The guardrails (non-negotiable in this framework's design)

- The engine **researches**; it does not place live orders on its own.
- Every candidate must survive an **out-of-sample** test before it counts.
- Backtests **must** include fees and slippage.
- Performance is **risk-adjusted**, not raw return.

## Tests

```bash
pytest -q
```

The bundled tests run offline against synthetic data.

## License

[MIT](LICENSE). See [SECURITY.md](SECURITY.md) for the public/private boundary
and how to handle secrets, and [CONTRIBUTING.md](CONTRIBUTING.md) to get
involved.
