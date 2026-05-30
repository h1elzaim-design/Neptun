"""Bollinger-Bands Mean-Reversion.

Long bei Close < unteres Band (SMA - k·StdDev), Exit beim Touch der Mitte.
Wie MeanReversion, aber explizit in Band-Sprache und mit konfigurierbarem k.
"""

from __future__ import annotations

import pandas as pd

from quantrace.models import MarketData
from quantrace.strategy import Strategy


class BollingerBands(Strategy):
    defaults = {"lookback": 20, "k": 2.0}

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = self.close(data)
        n = int(self.params["lookback"])
        k = float(self.params["k"])
        mean = close.rolling(n).mean()
        std = close.rolling(n).std(ddof=0)
        lower = mean - k * std
        entries = (close < lower).fillna(False)
        exits = (close >= mean).fillna(False)
        return entries.astype(bool), exits.astype(bool)
