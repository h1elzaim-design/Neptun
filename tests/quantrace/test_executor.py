"""Unit tests for the gated rebalance executor (fake broker, no network)."""

from __future__ import annotations

import pytest

from quantrace.brokers.base import Broker, Order, Position
from quantrace.paper import (
    LiveExecutionBlocked,
    RiskLimits,
    execute_plan,
    plan_rebalance,
)

LIMITS = RiskLimits()


class FakeBroker(Broker):
    """Records submitted orders; can be told to fail a given symbol."""

    def __init__(self, fail_symbol: str | None = None) -> None:
        self.orders: list[Order] = []
        self.fail_symbol = fail_symbol
        self._n = 0

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool:
        return True

    def account_value(self) -> float:
        return 100_000.0

    def positions(self) -> list[Position]:
        return []

    def submit(self, order: Order) -> str:
        if order.symbol == self.fail_symbol:
            raise RuntimeError("broker rejected order")
        self.orders.append(order)
        self._n += 1
        return f"oid-{self._n}"

    def cancel(self, order_id: str) -> None: ...


def _plan(targets, positions=None):
    return plan_rebalance(
        target_weights=targets,
        positions=positions or [],
        prices={"AAA": 100.0, "BBB": 50.0, "OLD": 100.0},
        account_value=100_000.0,
        limits=LIMITS,
    )


def test_paper_execution_submits_all_orders_with_client_ids():
    plan = _plan({"AAA": 0.10, "BBB": 0.10})
    broker = FakeBroker()
    report = execute_plan(plan, broker, live=False, client_tag="2026-06-14")
    assert report.n_submitted == 2 and report.n_failed == 0
    assert {o.client_id for o in broker.orders} == {
        "2026-06-14:AAA:BUY",
        "2026-06-14:BBB:BUY",
    }


def test_sells_submitted_before_buys():
    plan = _plan(
        {"AAA": 0.10},
        positions=[Position("OLD", 10, 100, 1000)],
    )
    broker = FakeBroker()
    execute_plan(plan, broker, live=False)
    assert broker.orders[0].side.value == "SELL"  # liquidate OLD first


def test_blocked_plan_submits_nothing():
    plan = plan_rebalance(
        {"AAA": 0.10}, [], {"AAA": 100.0}, 100_000.0, LIMITS, drawdown=-0.20
    )
    broker = FakeBroker()
    report = execute_plan(plan, broker, live=False)
    assert report.blocked and report.submitted == [] and broker.orders == []


def test_live_refused_without_gates():
    plan = _plan({"AAA": 0.10})
    with pytest.raises(LiveExecutionBlocked):
        execute_plan(plan, FakeBroker(), live=True, allow_live=False, live_enabled=False)
    with pytest.raises(LiveExecutionBlocked):
        execute_plan(plan, FakeBroker(), live=True, allow_live=True, live_enabled=False)


def test_live_allowed_with_both_gates():
    plan = _plan({"AAA": 0.10})
    broker = FakeBroker()
    report = execute_plan(plan, broker, live=True, allow_live=True, live_enabled=True)
    assert report.live and report.n_submitted == 1


def test_broker_error_is_captured_not_raised():
    plan = _plan({"AAA": 0.10, "BBB": 0.10})
    broker = FakeBroker(fail_symbol="AAA")
    report = execute_plan(plan, broker, live=False)
    assert report.n_failed == 1 and report.n_submitted == 1
    failed = next(o for o in report.submitted if o.error)
    assert failed.symbol == "AAA"
