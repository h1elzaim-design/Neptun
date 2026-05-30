"""MACD Signal-Line-Crossover.

MACD = EMA(fast) - EMA(slow). Signal = EMA(MACD, signal). Long wenn MACD über
Signal kreuzt, Exit beim Kreuz nach unten. Standard-Setup 12/26/9.
"""

from __future__ import annotations

import pandas as pd

from quantrace.models import MarketData
from quantrace.strategy import Strategy


class Macd(Strategy):
    defaults = {"fast": 12, "slow": 26, "signal": 9}

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = self.close(data)
        ema_fast = close.ewm(span=int(self.params["fast"]), adjust=False).mean()
        ema_slow = close.ewm(span=int(self.params["slow"]), adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=int(self.params["signal"]), adjust=False).mean()

        above = (macd > signal).fillna(False).astype(bool)
        above_prev = above.shift(1, fill_value=False).astype(bool)
        entries = above & ~above_prev
        exits = ~above & above_prev
        return entries, exits
