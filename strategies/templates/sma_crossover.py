"""Klassischer SMA-Crossover. Trend Following, einfach und robust als Baseline."""

from __future__ import annotations

import pandas as pd

from quantrace.models import MarketData
from quantrace.strategy import Strategy


class SmaCrossover(Strategy):
    defaults = {"fast": 20, "slow": 100}

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = self.close(data)
        fast = close.rolling(self.params["fast"]).mean()
        slow = close.rolling(self.params["slow"]).mean()
        long_now = (fast > slow).fillna(False).astype(bool)
        long_prev = long_now.shift(1, fill_value=False).astype(bool)
        entries = long_now & ~long_prev
        exits = ~long_now & long_prev
        return entries, exits
