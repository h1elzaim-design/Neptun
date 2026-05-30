"""Tests für die Walk-Forward-Validation."""

from __future__ import annotations

import pandas as pd
import pytest

from quantrace.models import StrategySpec, Timeframe
from quantrace.walk_forward import WalkForwardResult, _slice_market_data, walk_forward


def test_slice_market_data(synthetic_md):
    """Testet das Herausschneiden eines Zeitraums aus MarketData."""
    start = pd.Timestamp("2019-06-01")
    end = pd.Timestamp("2019-12-31")

    sliced = _slice_market_data(synthetic_md, start, end)

    assert sliced.start >= start.date()
    assert sliced.end <= end.date()
    assert len(sliced.frame) < len(synthetic_md.frame)
    assert sliced.symbols == synthetic_md.symbols


def test_slice_market_data_empty(synthetic_md):
    """Sollte ValueError werfen, wenn der Slice leer ist."""
    start = pd.Timestamp("2018-01-01")
    end = pd.Timestamp("2018-12-31")

    with pytest.raises(ValueError, match="ist leer"):
        _slice_market_data(synthetic_md, start, end)


def test_walk_forward_basic(synthetic_md):
    """Testet einen vollständigen Walk-Forward-Durchlauf."""
    spec = StrategySpec(
        strategy_id="wf_test",
        name="WF Test",
        class_path="strategies.templates.sma_crossover:SmaCrossover",
        strategy_class="trend_following",
        universe="synthetic",
        timeframe=Timeframe.DAILY,
        params={"fast": 10, "slow": 50},
        param_space={"fast": [10, 20], "slow": [50]},
    )

    # synthetic_md hat 5 Jahre Daten, 3 Folds sollten gut passen
    result = walk_forward(spec, synthetic_md, n_folds=3, train_ratio=0.5)

    assert isinstance(result, WalkForwardResult)
    assert result.n_folds == 3
    assert len(result.folds) == 3
    assert result.rank_by == "sharpe"

    # Metriken sollten gesetzt sein
    assert result.is_sharpe_mean != 0.0
    assert result.oos_sharpe_mean != 0.0

    # Prüfe Fold-Inhalte
    for i, fold in enumerate(result.folds, 1):
        assert fold.fold_index == i
        if i > 1:
            assert fold.train_sharpe != 0.0
        assert "fast" in fold.chosen_params
        assert "slow" in fold.chosen_params


def test_walk_forward_not_enough_data(synthetic_md):
    """split_walk_forward wirft ValueError bei zu wenig Daten."""
    spec = StrategySpec(
        strategy_id="wf_test",
        name="WF Test",
        class_path="strategies.templates.sma_crossover:SmaCrossover",
        strategy_class="trend_following",
        universe="synthetic",
        timeframe=Timeframe.DAILY,
        param_space={"fast": [10], "slow": [50]},
    )

    # Wir schneiden künstlich auf 50 Tage runter
    small_md = _slice_market_data(
        synthetic_md, synthetic_md.frame.index[0], synthetic_md.frame.index[49]
    )

    with pytest.raises(ValueError, match="Zu wenig Daten"):
        walk_forward(spec, small_md, n_folds=3)
