from __future__ import annotations

from quantrace.models import StrategySpec, Timeframe
from quantrace.strategy import load_strategy
from strategies.templates.mean_reversion import MeanReversion
from strategies.templates.sma_crossover import SmaCrossover


def test_sma_crossover_signal_shape(synthetic_md):
    strat = SmaCrossover(fast=10, slow=30)
    entries, exits = strat.generate_signals(synthetic_md)
    assert entries.shape == exits.shape
    assert set(entries.columns) == set(synthetic_md.symbols)
    assert entries.dtypes.eq(bool).all()
    assert exits.dtypes.eq(bool).all()


def test_mean_reversion_signal_shape(synthetic_md):
    strat = MeanReversion(lookback=20, entry_z=2.0, exit_z=0.0)
    entries, exits = strat.generate_signals(synthetic_md)
    assert entries.shape == synthetic_md.frame.xs("close", level="field", axis=1).shape
    # Es sollte mindestens ein Signal in 5 Jahren synthetischer Daten geben
    assert entries.any().any() or exits.any().any()


def test_load_strategy_from_spec():
    spec = StrategySpec(
        strategy_id="sma_5_20",
        name="x",
        class_path="strategies.templates.sma_crossover:SmaCrossover",
        strategy_class="trend_following",
        universe="u",
        timeframe=Timeframe.DAILY,
        params={"fast": 5, "slow": 20},
    )
    strat = load_strategy(spec)
    assert isinstance(strat, SmaCrossover)
    assert strat.params["fast"] == 5
    assert strat.params["slow"] == 20
