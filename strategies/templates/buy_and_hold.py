"""Buy-and-Hold Baseline.

Triviale Baseline: kauft am ersten Tag, hält bis zum Ende. Existiert primär,
damit Sharpe / CAGR / MaxDD jeder anderen Strategie gegen ein realistisches
Benchmark verglichen werden kann. Wenn deine "Edge-Strategie" Buy-and-Hold nicht
schlägt, ist sie keine Edge.
"""

from __future__ import annotations

import pandas as pd

from quantrace.models import MarketData
from quantrace.strategy import Strategy


class BuyAndHold(Strategy):
    defaults: dict[str, object] = {}

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = self.close(data)
        entries = pd.DataFrame(False, index=close.index, columns=close.columns)
        exits = pd.DataFrame(False, index=close.index, columns=close.columns)
        # erste valide Zeile pro Symbol → Entry
        first_valid = close.notna().idxmax()
        for sym, ts in first_valid.items():
            entries.loc[ts, sym] = True
        return entries.astype(bool), exits.astype(bool)
