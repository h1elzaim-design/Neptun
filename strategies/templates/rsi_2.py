"""RSI-2 Mean-Reversion (Larry Connors).

Kurzfristiges Mean-Reversion-Setup auf Tagesbasis:
  - Long bei RSI(2) < entry_rsi (oversold)
  - Exit bei RSI(2) > exit_rsi
Optionaler Trend-Filter: Long nur, wenn Close über SMA(trend_sma).
"""

from __future__ import annotations

import pandas as pd

from quantrace.models import MarketData
from quantrace.strategy import Strategy


def _rsi(close: pd.DataFrame, period: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


class Rsi2(Strategy):
    defaults = {"period": 2, "entry_rsi": 10.0, "exit_rsi": 70.0, "trend_sma": 200}

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = self.close(data)
        rsi = _rsi(close, int(self.params["period"]))

        oversold = (rsi < float(self.params["entry_rsi"])).fillna(False)
        if int(self.params["trend_sma"]) > 0:
            trend_ok = (close > close.rolling(int(self.params["trend_sma"])).mean()).fillna(False)
            entries = oversold & trend_ok
        else:
            entries = oversold

        exits = (rsi > float(self.params["exit_rsi"])).fillna(False)
        return entries.astype(bool), exits.astype(bool)
