"""Unit tests for the paper-trading rebalance planner (no execution)."""

from __future__ import annotations

from quantrace.brokers.base import OrderSide, Position
from quantrace.paper import RiskLimits, plan_rebalance

LIMITS = RiskLimits(
    max_notional_pct=0.10,
    max_notional_abs=25_000.0,
    max_orders=50,
    max_turnover_pct=1.0,
    gross_leverage=1.0,
    max_drawdown_kill_switch=-0.10,
)


def test_rebalance_from_cash_buys_targets():
    plan = plan_rebalance(
        target_weights={"AAA": 0.10, "BBB": 0.10},
        positions=[],
        prices={"AAA": 100.0, "BBB": 50.0},
        account_value=100_000.0,
        limits=LIMITS,
    )
    assert not plan.blocked
    by = {o.symbol: o for o in plan.orders}
    # 10% of 100k = 10k → 100 shares @100, 200 shares @50.
    assert by["AAA"].side is OrderSide.BUY and by["AAA"].quantity == 100.0
    assert by["BBB"].side is OrderSide.BUY and by["BBB"].quantity == 200.0
    assert plan.turnover == 0.20  # 20k / 100k


def test_symbol_absent_from_target_is_liquidated():
    plan = plan_rebalance(
        target_weights={"AAA": 0.10},
        positions=[Position(symbol="OLD", quantity=10, avg_cost=100, market_value=1000)],
        prices={"AAA": 100.0, "OLD": 100.0},
        account_value=100_000.0,
        limits=LIMITS,
    )
    old = next(o for o in plan.orders if o.symbol == "OLD")
    assert old.side is OrderSide.SELL and old.quantity == 10.0
    # Sells are ordered before buys.
    assert plan.orders[0].side is OrderSide.SELL


def test_per_position_cap_clamps_weight():
    plan = plan_rebalance(
        target_weights={"AAA": 0.50},  # asks 50%, cap is 10%
        positions=[],
        prices={"AAA": 100.0},
        account_value=100_000.0,
        limits=LIMITS,
    )
    order = plan.orders[0]
    assert order.quantity == 100.0  # capped to 10% → 10k → 100 shares
    assert any("capped" in w for w in plan.warnings)


def test_abs_cap_binds_when_tighter_than_pct():
    # NAV 1M: 10% = 100k, but max_notional_abs = 25k binds → 250 shares.
    plan = plan_rebalance(
        target_weights={"AAA": 0.10},
        positions=[],
        prices={"AAA": 100.0},
        account_value=1_000_000.0,
        limits=LIMITS,
    )
    assert plan.orders[0].quantity == 250.0
    assert any("capped" in w for w in plan.warnings)


def test_kill_switch_blocks_all_orders():
    plan = plan_rebalance(
        target_weights={"AAA": 0.10},
        positions=[],
        prices={"AAA": 100.0},
        account_value=100_000.0,
        limits=LIMITS,
        drawdown=-0.15,
    )
    assert plan.blocked
    assert plan.orders == []
    assert any("kill-switch" in w for w in plan.warnings)


def test_dust_trades_dropped():
    # Already on target → tiny/no delta, no orders.
    plan = plan_rebalance(
        target_weights={"AAA": 0.10},
        positions=[Position(symbol="AAA", quantity=100, avg_cost=100, market_value=10_000)],
        prices={"AAA": 100.0},
        account_value=100_000.0,
        limits=LIMITS,
        min_trade_notional=1.0,
    )
    assert plan.orders == []


def test_missing_price_warns_and_skips():
    plan = plan_rebalance(
        target_weights={"AAA": 0.10, "NOPX": 0.10},
        positions=[],
        prices={"AAA": 100.0},
        account_value=100_000.0,
        limits=LIMITS,
    )
    assert {o.symbol for o in plan.orders} == {"AAA"}
    assert any("NOPX" in w for w in plan.warnings)


def test_turnover_warning_when_over_daily_limit():
    tight = RiskLimits(max_turnover_pct=0.05, max_notional_pct=1.0, max_notional_abs=1e12)
    plan = plan_rebalance(
        target_weights={"AAA": 0.50},
        positions=[],
        prices={"AAA": 100.0},
        account_value=100_000.0,
        limits=tight,
    )
    assert any("Turnover" in w for w in plan.warnings)
