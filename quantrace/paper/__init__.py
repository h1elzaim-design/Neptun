"""Paper-trading scaffolding (Phase 4) — planning only, no execution here.

This package computes *what would be traded* to move a paper portfolio to its
target weights. It deliberately submits nothing: turning a plan into broker
orders is gated behind explicit human approval (see ADR-005 in
docs/ARCHITECTURE.md and CLAUDE.md "handelt nie autonom"). The planner is pure
and unit-tested so the
risky part — execution — stays a thin, separately-reviewed adapter.
"""

from quantrace.paper.daily_plan import (
    DailyPlanReport,
    build_daily_plan,
    render_plan_note,
)
from quantrace.paper.executor import (
    ExecutionReport,
    LiveExecutionBlocked,
    OrderResult,
    execute_plan,
)
from quantrace.paper.rebalance import (
    PlannedOrder,
    RebalancePlan,
    RiskLimits,
    plan_rebalance,
)
from quantrace.paper.registry import (
    ApprovedCandidate,
    PortfolioRegistry,
    load_registry,
)

__all__ = [
    "ApprovedCandidate",
    "DailyPlanReport",
    "ExecutionReport",
    "LiveExecutionBlocked",
    "OrderResult",
    "PlannedOrder",
    "PortfolioRegistry",
    "RebalancePlan",
    "RiskLimits",
    "build_daily_plan",
    "execute_plan",
    "load_registry",
    "plan_rebalance",
    "render_plan_note",
]
