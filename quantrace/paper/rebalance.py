"""Pure rebalance planner: target weights → proposed orders.

Given a set of target portfolio weights, the current broker positions, the
account value and the latest prices, compute the orders that would move the
portfolio onto its target — and check them against the risk limits in
`config/risk_limits/limits.yaml`. **Nothing is submitted.** The output is a
:class:`RebalancePlan` that a human reviews before any execution adapter acts.

Design choices (quant-grade, intentionally conservative):
- Per-position caps (`max_notional_pct`, `max_notional_abs`) *clamp* the target
  weight rather than reject the rebalance — an oversized target silently becomes
  a compliant one, with a warning. This keeps the planner total.
- A drawdown beyond the kill-switch *blocks* the whole rebalance (no orders) —
  this is the one hard stop, mirroring `max_drawdown_kill_switch`.
- Dust trades below `min_trade_notional` are dropped so daily rebalances don't
  churn the book on rounding noise.
- Turnover / order-count breaches *warn* (they're soft, governance-level limits)
  rather than mutate the plan; the human decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quantrace.brokers.base import Order, OrderSide, OrderType, Position


@dataclass(slots=True)
class RiskLimits:
    """Subset of config/risk_limits/limits.yaml the planner enforces."""

    max_notional_pct: float = 0.10
    max_notional_abs: float = 25_000.0
    max_orders: int = 50
    max_turnover_pct: float = 1.0
    gross_leverage: float = 1.0
    max_drawdown_kill_switch: float = -0.10

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> RiskLimits:
        pos = cfg.get("per_position") or {}
        day = cfg.get("per_day") or {}
        port = cfg.get("portfolio") or {}
        return cls(
            max_notional_pct=float(pos.get("max_notional_pct", 0.10)),
            max_notional_abs=float(pos.get("max_notional_abs", 25_000.0)),
            max_orders=int(day.get("max_orders", 50)),
            max_turnover_pct=float(day.get("max_turnover_pct", 1.0)),
            gross_leverage=float(port.get("gross_leverage", 1.0)),
            max_drawdown_kill_switch=float(port.get("max_drawdown_kill_switch", -0.10)),
        )


@dataclass(slots=True)
class PlannedOrder:
    symbol: str
    side: OrderSide
    quantity: float          # shares, always positive
    current_shares: float
    target_shares: float
    price: float
    notional: float          # |quantity * price|, the traded value

    def to_order(self) -> Order:
        """Materialise a broker Order — called only by a gated execution layer."""
        return Order(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            order_type=OrderType.MARKET,
        )


@dataclass(slots=True)
class RebalancePlan:
    account_value: float
    orders: list[PlannedOrder] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocked: bool = False
    gross_after: float = 0.0
    turnover: float = 0.0

    @property
    def n_orders(self) -> int:
        return len(self.orders)


def _positions_by_symbol(positions: list[Position]) -> dict[str, float]:
    return {p.symbol: float(p.quantity) for p in positions}


def plan_rebalance(
    target_weights: dict[str, float],
    positions: list[Position],
    prices: dict[str, float],
    account_value: float,
    limits: RiskLimits,
    *,
    drawdown: float = 0.0,
    min_trade_notional: float = 1.0,
) -> RebalancePlan:
    """Compute the orders that move the portfolio onto ``target_weights``.

    Parameters
    ----------
    target_weights
        symbol → fraction of NAV (longs only for now; 0 ≤ w). Symbols held but
        absent from the target are liquidated (target 0).
    positions
        Current broker positions.
    prices
        symbol → latest price. A symbol without a price is skipped with a warning.
    account_value
        Portfolio NAV.
    limits
        Risk limits to enforce / warn on.
    drawdown
        Current equity drawdown as a negative fraction (e.g. -0.12 for -12%).
        Beyond ``max_drawdown_kill_switch`` the whole rebalance is blocked.
    min_trade_notional
        Trades whose value is below this are dropped as dust.
    """
    plan = RebalancePlan(account_value=account_value)

    if account_value <= 0:
        plan.blocked = True
        plan.warnings.append(f"Non-positive account value ({account_value}); no orders.")
        return plan

    # Hard stop: kill-switch on drawdown.
    if drawdown <= limits.max_drawdown_kill_switch:
        plan.blocked = True
        plan.warnings.append(
            f"Drawdown {drawdown:.2%} ≤ kill-switch {limits.max_drawdown_kill_switch:.2%}: "
            "rebalance blocked, no orders."
        )
        return plan

    current = _positions_by_symbol(positions)
    symbols = sorted(set(target_weights) | set(current))

    per_pos_cap = min(limits.max_notional_pct, limits.max_notional_abs / account_value)

    gross_notional = 0.0
    turnover_notional = 0.0
    for sym in symbols:
        price = prices.get(sym)
        if price is None or price <= 0:
            if target_weights.get(sym, 0.0) > 0 or current.get(sym, 0.0):
                plan.warnings.append(f"No price for {sym}; skipped.")
            continue

        raw_w = max(0.0, float(target_weights.get(sym, 0.0)))
        capped_w = min(raw_w, per_pos_cap)
        if capped_w < raw_w:
            plan.warnings.append(
                f"{sym}: target {raw_w:.1%} capped to {capped_w:.1%} (per-position limit)."
            )

        target_notional = capped_w * account_value
        target_shares = target_notional / price
        current_shares = current.get(sym, 0.0)
        delta = target_shares - current_shares
        trade_notional = abs(delta) * price

        gross_notional += abs(target_shares) * price

        if trade_notional < min_trade_notional:
            continue

        turnover_notional += trade_notional
        plan.orders.append(
            PlannedOrder(
                symbol=sym,
                side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                quantity=abs(delta),
                current_shares=current_shares,
                target_shares=target_shares,
                price=price,
                notional=trade_notional,
            )
        )

    # Sells first: free buying power before buys (matters for a real broker).
    plan.orders.sort(key=lambda o: (o.side != OrderSide.SELL, o.symbol))

    plan.gross_after = gross_notional / account_value
    plan.turnover = turnover_notional / account_value

    if plan.gross_after > limits.gross_leverage + 1e-9:
        plan.warnings.append(
            f"Gross leverage {plan.gross_after:.2f} exceeds limit {limits.gross_leverage:.2f}."
        )
    if plan.turnover > limits.max_turnover_pct + 1e-9:
        plan.warnings.append(
            f"Turnover {plan.turnover:.1%} exceeds daily limit {limits.max_turnover_pct:.1%}."
        )
    if plan.n_orders > limits.max_orders:
        plan.warnings.append(
            f"{plan.n_orders} orders exceed daily cap {limits.max_orders}."
        )

    return plan
