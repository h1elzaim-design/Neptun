"""Smoke-Tests für alle Standard-Strategie-Templates.

Jede Strategie muss:
  - mit Defaults instanziieren
  - generate_signals() mit (entries, exits) zurückgeben
  - bool DataFrames in Form (n_rows × n_symbols) liefern
  - mindestens irgendwo entries/exits produzieren (auf 5 Jahren synthetischer Daten)
"""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.templates.atr_breakout import AtrBreakout
from strategies.templates.bollinger_bands import BollingerBands
from strategies.templates.buy_and_hold import BuyAndHold
from strategies.templates.donchian_breakout import DonchianBreakout
from strategies.templates.dual_momentum import DualMomentum
from strategies.templates.ema_crossover import EmaCrossover
from strategies.templates.kalman_trend import KalmanTrend
from strategies.templates.macd import Macd
from strategies.templates.momentum_12_1 import CrossSectionalMomentum
from strategies.templates.regime_filter import RegimeFilter
from strategies.templates.rsi_2 import Rsi2

STRATEGIES = [
    DonchianBreakout(),
    EmaCrossover(),
    Macd(),
    Rsi2(),
    BollingerBands(),
    CrossSectionalMomentum(lookback=60, skip=5, top_quantile=0.5),
    DualMomentum(lookback=60, top_quantile=0.5),
    KalmanTrend(),
    RegimeFilter(trend_lookback=50),
    AtrBreakout(),
    BuyAndHold(),
]


@pytest.mark.parametrize("strat", STRATEGIES, ids=lambda s: type(s).__name__)
def test_template_signal_contract(strat, synthetic_md):
    entries, exits = strat.generate_signals(synthetic_md)
    assert isinstance(entries, pd.DataFrame)
    assert isinstance(exits, pd.DataFrame)
    assert entries.shape == exits.shape
    assert set(entries.columns) == set(synthetic_md.symbols)
    assert entries.dtypes.eq(bool).all()
    assert exits.dtypes.eq(bool).all()


@pytest.mark.parametrize(
    "strat", [s for s in STRATEGIES if not isinstance(s, BuyAndHold)],
    ids=lambda s: type(s).__name__,
)
def test_template_generates_at_least_one_signal(strat, synthetic_md):
    entries, exits = strat.generate_signals(synthetic_md)
    assert entries.any().any() or exits.any().any(), (
        f"{type(strat).__name__} produziert keine Signale auf 5 Jahren synthetischer Daten"
    )


def test_buy_and_hold_has_exactly_one_entry_per_symbol(synthetic_md):
    entries, exits = BuyAndHold().generate_signals(synthetic_md)
    per_symbol = entries.sum(axis=0)
    assert (per_symbol == 1).all(), f"Buy-and-Hold soll je Symbol genau 1 Entry haben, war: {per_symbol.to_dict()}"
    assert not exits.any().any(), "Buy-and-Hold soll keine Exits haben"
