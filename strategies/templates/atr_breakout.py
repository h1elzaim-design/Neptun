"""ATR-Channel-Breakout.

Long bei Close > SMA(n) + k·ATR(n), Exit bei Close < SMA(n) - k·ATR(n).
Robusterer Breakout als Donchian, weil er Volatilität dynamisch berücksichtigt.
"""

from __future__ import annotations

import pandas as pd

from quantrace.models import MarketData
from quantrace.strategy import Strategy


def _field(data: MarketData, name: str) -> pd.DataFrame:
    return data.frame.xs(name, level="field", axis=1)


def _atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).stack(future_stack=True),
            (high - prev_close).abs().stack(future_stack=True),
            (low - prev_close).abs().stack(future_stack=True),
        ],
        axis=1,
    ).max(axis=1)
    tr = tr.unstack()
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


class AtrBreakout(Strategy):
    defaults = {"lookback": 20, "k": 2.0}

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = self.close(data)
        try:
            high = _field(data, "high")
            low = _field(data, "low")
        except KeyError:
            high = close
            low = close
        n = int(self.params["lookback"])
        k = float(self.params["k"])

        sma = close.rolling(n).mean()
        atr = _atr(high, low, close, n)
        upper = sma + k * atr
        lower = sma - k * atr

        entries = (close > upper).fillna(False)
        exits = (close < lower).fillna(False)
        return entries.astype(bool), exits.astype(bool)
