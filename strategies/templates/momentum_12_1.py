"""Cross-Sectional-Momentum (12-1).

Klassisches Jegadeesh-Titman-Momentum: Rangliste der Assets nach Return über
die letzten `lookback` Tage minus den letzten `skip` Tagen (12-1 = 12 Monate
ohne den jüngsten Monat). Long auf das Top-Quantil, Exit wenn nicht mehr im
Top-Quantil.

Erzeugt Signale auf täglicher Frequenz; in der Praxis wird das oft monatlich
rebalanced, hier signalisiert es kontinuierlich.
"""

from __future__ import annotations

import pandas as pd

from quantrace.models import MarketData
from quantrace.strategy import Strategy


class CrossSectionalMomentum(Strategy):
    defaults = {"lookback": 252, "skip": 21, "top_quantile": 0.3}

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = self.close(data)
        lookback = int(self.params["lookback"])
        skip = int(self.params["skip"])
        top_q = float(self.params["top_quantile"])

        momentum = close.shift(skip) / close.shift(lookback) - 1.0
        # rank cross-sectionally per date (axis=1)
        ranks = momentum.rank(axis=1, pct=True)
        in_top = (ranks >= (1.0 - top_q)).fillna(False)
        in_top_prev = in_top.shift(1, fill_value=False).astype(bool)
        entries = in_top & ~in_top_prev
        exits = ~in_top & in_top_prev
        return entries.astype(bool), exits.astype(bool)
