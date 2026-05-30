"""EMA-Crossover.

Wie SMA-Crossover, aber gewichtet jüngere Preise stärker (geringerer Lag).
Reagiert schneller auf Trendwechsel, neigt aber zu mehr Whipsaws bei kleinen
Fast-Perioden.
"""

from __future__ import annotations

import pandas as pd

from quantrace.models import MarketData
from quantrace.strategy import Strategy


class EmaCrossover(Strategy):
    defaults = {"fast": 12, "slow": 26}

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = self.close(data)
        fast = close.ewm(span=int(self.params["fast"]), adjust=False).mean()
        slow = close.ewm(span=int(self.params["slow"]), adjust=False).mean()
        long_now = (fast > slow).fillna(False).astype(bool)
        long_prev = long_now.shift(1, fill_value=False).astype(bool)
        entries = long_now & ~long_prev
        exits = ~long_now & long_prev
        return entries, exits
