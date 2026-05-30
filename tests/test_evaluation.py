from __future__ import annotations

from datetime import date

import pandas as pd

from quantrace.evaluation import evaluate, split_walk_forward
from quantrace.models import BacktestConfig, BacktestResult, StrategySpec, Timeframe, TradeMetrics


def _mock_bt(strategy_id: str, sharpe: float, dd: float, trades: int = 50) -> BacktestResult:
    return BacktestResult(
        strategy_id=strategy_id,
        data_hash="deadbeef",
        config=BacktestConfig(),
        start=date(2020, 1, 1),
        end=date(2022, 12, 31),
        total_return=0.30,
        cagr=0.10,
        sharpe=sharpe,
        sortino=sharpe * 1.2,
        calmar=0.5,
        max_drawdown=dd,
        avg_drawdown=dd / 3,
        ulcer_index=0.05,
        trades=TradeMetrics(
            n_trades=trades,
            win_rate=0.55,
            avg_trade_return=0.01,
            avg_winner=0.03,
            avg_loser=-0.02,
            profit_factor=1.8,
            expectancy=0.005,
        ),
    )


def _spec() -> StrategySpec:
    return StrategySpec(
        strategy_id="test",
        name="test",
        class_path="strategies.templates.sma_crossover:SmaCrossover",
        strategy_class="trend_following",
        universe="u",
        timeframe=Timeframe.DAILY,
        params={"fast": 20, "slow": 100},
    )


def test_evaluate_passes_when_metrics_good():
    bts = [_mock_bt("test", sharpe=1.6, dd=-0.18), _mock_bt("test", sharpe=1.4, dd=-0.20)]
    rep = evaluate(_spec(), bts)
    assert rep.passed_guardrails is True
    assert rep.score_total > 0


def test_evaluate_rejects_low_sharpe():
    bts = [_mock_bt("test", sharpe=0.2, dd=-0.10), _mock_bt("test", sharpe=0.1, dd=-0.10)]
    rep = evaluate(_spec(), bts)
    assert rep.passed_guardrails is False
    assert any("sharpe" in r for r in rep.rejection_reasons)


def test_evaluate_rejects_when_no_oos():
    bts = [_mock_bt("test", sharpe=1.8, dd=-0.10)]
    rep = evaluate(_spec(), bts)
    assert rep.passed_guardrails is False
    assert any("Out-of-sample" in r for r in rep.rejection_reasons)


def test_walk_forward_split():
    idx = pd.bdate_range("2018-01-01", "2023-12-31")
    folds = split_walk_forward(idx, n_folds=4)
    assert len(folds) == 4
    for train_start, train_end, test_end in folds:
        assert train_start <= train_end <= test_end
