"""Execute a rebalance plan against a broker — the gated, outward-facing step.

Separated from planning so the risky part is small and reviewable. Guarantees:

- **Paper by default.** Live execution is refused unless *both* ``allow_live``
  and ``live_enabled`` (config/risk_limits/limits.yaml → live_trading.enabled)
  are true. This is the hard governance gate ("handelt nie autonom").
- **Kill-switch honoured.** A blocked plan (drawdown kill-switch, non-positive
  NAV) submits nothing.
- **Idempotent.** Each order carries a deterministic ``client_id`` derived from
  ``client_tag`` (default: today) + symbol + side, so re-running the same day
  is rejected as a duplicate by the broker instead of double-trading.
- **Sells before buys.** The plan is already sorted that way; we preserve it to
  free buying power first.

The broker is injected (any :class:`quantrace.brokers.base.Broker`), so this is
unit-tested with a fake broker — no network, no real orders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from quantrace.brokers.base import Broker
from quantrace.paper.rebalance import RebalancePlan

log = logging.getLogger(__name__)


# Named without the ruff-preferred "Error" suffix on purpose: this is a safety
# guard whose name reads as a state ("live execution blocked") at every call and
# import site. Renaming would churn the gated-execution API for a cosmetic rule.
class LiveExecutionBlocked(PermissionError):  # noqa: N818
    """Raised when live execution is requested without the required gates."""


@dataclass(slots=True)
class OrderResult:
    symbol: str
    side: str
    quantity: float
    client_id: str
    order_id: str | None = None
    error: str | None = None


@dataclass(slots=True)
class ExecutionReport:
    live: bool
    blocked: bool = False
    submitted: list[OrderResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_submitted(self) -> int:
        return sum(1 for o in self.submitted if o.order_id is not None)

    @property
    def n_failed(self) -> int:
        return sum(1 for o in self.submitted if o.error is not None)


def execute_plan(
    plan: RebalancePlan,
    broker: Broker,
    *,
    live: bool,
    allow_live: bool = False,
    live_enabled: bool = False,
    client_tag: str | None = None,
) -> ExecutionReport:
    """Submit ``plan``'s orders through ``broker``. Returns a per-order report.

    Parameters
    ----------
    live
        True if this targets a live (real-money) account, False for paper.
    allow_live, live_enabled
        Both must be true for live execution; otherwise it is refused.
    client_tag
        Idempotency namespace for client order ids. Defaults to today's date,
        so a same-day re-run is deduplicated by the broker.
    """
    report = ExecutionReport(live=live, warnings=list(plan.warnings))

    if live and not (allow_live and live_enabled):
        raise LiveExecutionBlocked(
            "Live execution requires allow_live=True AND live_trading.enabled. "
            "Refusing — run against paper instead."
        )

    if plan.blocked:
        report.blocked = True
        log.warning("execute_plan: plan blocked, no orders submitted")
        return report

    tag = client_tag or date.today().isoformat()
    for po in plan.orders:
        client_id = f"{tag}:{po.symbol}:{po.side.value}"
        order = po.to_order()
        order.client_id = client_id
        try:
            order_id = broker.submit(order)
            report.submitted.append(
                OrderResult(
                    symbol=po.symbol,
                    side=po.side.value,
                    quantity=po.quantity,
                    client_id=client_id,
                    order_id=str(order_id),
                )
            )
        except Exception as exc:  # broker/network errors must not abort the batch
            log.warning("execute_plan: submit failed for %s: %s", po.symbol, exc)
            report.submitted.append(
                OrderResult(
                    symbol=po.symbol,
                    side=po.side.value,
                    quantity=po.quantity,
                    client_id=client_id,
                    error=str(exc),
                )
            )

    log.info(
        "execute_plan: %s submitted=%d failed=%d (live=%s)",
        tag, report.n_submitted, report.n_failed, live,
    )
    return report
