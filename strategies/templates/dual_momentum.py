"""Dual Momentum (Gary Antonacci).

Kombiniert zwei orthogonale Momentum-Signale:

  * **Relativ** — ranke alle Assets nach Return über `lookback` Tage und halte
    nur das Top-Quantil (cross-sectional, wie klassisches Momentum).
  * **Absolut** — halte ein Asset nur, wenn sein *eigener* Lookback-Return über
    `abs_threshold` liegt (sonst Cash). Der absolute Filter ist der Risk-off-
    Schalter, der Dual Momentum in breiten Bärenmärkten ganz aus dem Markt nimmt
    — auch dann, wenn ein Asset *relativ* noch top ist.

Abgrenzung zu `momentum_12_1`: jenes nutzt ein Skip-Window (Reversal-Filter) und
kennt keinen absoluten Cash-Schalter. Dual Momentum tauscht das Skip gegen den
absoluten Trend-Filter — empirisch der größere Hebel für die Drawdown-Kontrolle.
"""

from __future__ import annotations

import pandas as pd

from quantrace.models import MarketData
from quantrace.strategy import Strategy


class DualMomentum(Strategy):
    defaults = {"lookback": 252, "top_quantile": 0.5, "abs_threshold": 0.0}

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = self.close(data)
        lookback = int(self.params["lookback"])
        top_q = float(self.params["top_quantile"])
        abs_thr = float(self.params["abs_threshold"])

        momentum = close / close.shift(lookback) - 1.0
        ranks = momentum.rank(axis=1, pct=True)

        in_top = (ranks >= (1.0 - top_q)).fillna(False)
        abs_ok = (momentum > abs_thr).fillna(False)
        hold = in_top & abs_ok

        hold_prev = hold.shift(1, fill_value=False).astype(bool)
        entries = hold & ~hold_prev
        exits = ~hold & hold_prev
        return entries.astype(bool), exits.astype(bool)
