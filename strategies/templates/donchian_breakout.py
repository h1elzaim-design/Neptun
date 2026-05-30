"""Donchian-Breakout (Turtle-Style).

Long bei Ausbruch über das N-Tage-Hoch, Exit bei Bruch unter das M-Tage-Tief.
Klassisches Trend-Following — performt in starken Trends, viele kleine Verluste
in Seitwärtsphasen.
"""

from __future__ import annotations

import pandas as pd

from quantrace.data_agent import close_prices
from quantrace.models import MarketData
from quantrace.strategy import Strategy


def _field(data: MarketData, name: str) -> pd.DataFrame:
    """Extract a single OHLCV field across all symbols as wide DataFrame."""
    return data.frame.xs(name, level="field", axis=1)


class DonchianBreakout(Strategy):
    defaults = {"entry_period": 20, "exit_period": 10}

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = close_prices(data)
        try:
            high = _field(data, "high")
            low = _field(data, "low")
        except KeyError:
            high = close
            low = close

        entry_n = int(self.params["entry_period"])
        exit_n = int(self.params["exit_period"])

        upper = high.rolling(entry_n).max().shift(1)
        lower = low.rolling(exit_n).min().shift(1)

        entries = (close > upper).fillna(False)
        exits = (close < lower).fillna(False)
        return entries, exits
