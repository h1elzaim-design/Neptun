"""Mean Reversion auf z-Score des Close gegen rollendes Mittel.

Long bei z < -entry_z, Exit wenn z >= exit_z. Bewusst symmetrisch einfach gehalten,
damit es als Template für Variation taugt (Bollinger, RSI, etc.).
"""

from __future__ import annotations

import pandas as pd

from quantrace.models import MarketData
from quantrace.strategy import Strategy


class MeanReversion(Strategy):
    defaults = {"lookback": 20, "entry_z": 2.0, "exit_z": 0.0}

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = self.close(data)
        mean = close.rolling(self.params["lookback"]).mean()
        std = close.rolling(self.params["lookback"]).std(ddof=0)
        z = (close - mean) / std.replace(0, pd.NA)

        entries = (z < -self.params["entry_z"]).fillna(False)
        exits = (z >= self.params["exit_z"]).fillna(False)
        return entries, exits
