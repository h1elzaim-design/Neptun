"""IBKR-Adapter über ib_insync.

Stub für Phase 4. Wir wollen die Schnittstelle JETZT, damit Strategien
broker-agnostisch entwickelt werden. Aber Live-Order-Submission bleibt
explizit hinter einer Guardrail (`allow_live=False` per default).

Pflichtlektüre vor Aktivierung: IBKR Gateway/TWS muss laufen, Paper Port
ist 7497, Live ist 7496. Niemals Live ohne menschliche Freigabe.
"""

from __future__ import annotations

import logging
import os

from quantrace.brokers.base import Broker, Order, OrderSide, OrderType, Position

log = logging.getLogger(__name__)


class IBKRBroker(Broker):
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        account: str | None = None,
        allow_live: bool = False,
    ) -> None:
        self.host = host or os.environ.get("IBKR_HOST", "127.0.0.1")
        self.port = int(port or os.environ.get("IBKR_PORT", "7497"))
        self.client_id = int(client_id or os.environ.get("IBKR_CLIENT_ID", "17"))
        self.account = account or os.environ.get("IBKR_ACCOUNT", "")
        self.allow_live = allow_live
        self._ib = None

        if self.port == 7496 and not allow_live:
            raise RuntimeError("IBKR Port 7496 ist LIVE. Setze allow_live=True nur nach Freigabe.")

    def _require_ib(self):
        if self._ib is None:
            try:
                from ib_insync import IB
            except ImportError as e:
                raise ImportError("ib_insync nicht installiert. `pip install -e .[ibkr]`.") from e
            self._ib = IB()
        return self._ib

    def connect(self) -> None:
        ib = self._require_ib()
        ib.connect(self.host, self.port, clientId=self.client_id, readonly=not self.allow_live)
        log.info(
            "IBKR connected: %s:%s clientId=%s readonly=%s",
            self.host,
            self.port,
            self.client_id,
            not self.allow_live,
        )

    def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    def account_value(self) -> float:
        ib = self._require_ib()
        values = ib.accountValues(self.account or "")
        for v in values:
            if v.tag == "NetLiquidation" and v.currency == "USD":
                return float(v.value)
        return 0.0

    def positions(self) -> list[Position]:
        ib = self._require_ib()
        out: list[Position] = []
        for p in ib.positions(self.account or ""):
            out.append(
                Position(
                    symbol=p.contract.symbol,
                    quantity=float(p.position),
                    avg_cost=float(p.avgCost),
                    market_value=0.0,
                )
            )
        return out

    def submit(self, order: Order) -> str:
        if not self.allow_live and self.port == 7496:
            raise PermissionError("Live-Order auf Port 7496 nicht erlaubt. Freigabe fehlt.")
        from ib_insync import LimitOrder, MarketOrder, Stock, StopOrder

        ib = self._require_ib()
        contract = Stock(order.symbol, "SMART", "USD")
        ib.qualifyContracts(contract)

        action = "BUY" if order.side is OrderSide.BUY else "SELL"
        if order.order_type is OrderType.MARKET:
            ib_order = MarketOrder(action, order.quantity, tif=order.tif)
        elif order.order_type is OrderType.LIMIT:
            assert order.limit_price is not None
            ib_order = LimitOrder(action, order.quantity, order.limit_price, tif=order.tif)
        elif order.order_type is OrderType.STOP:
            assert order.stop_price is not None
            ib_order = StopOrder(action, order.quantity, order.stop_price, tif=order.tif)
        else:
            raise ValueError(f"Unbekannter OrderType: {order.order_type}")

        trade = ib.placeOrder(contract, ib_order)
        return str(trade.order.orderId)

    def cancel(self, order_id: str) -> None:
        ib = self._require_ib()
        for trade in ib.openTrades():
            if str(trade.order.orderId) == order_id:
                ib.cancelOrder(trade.order)
                return
        log.warning("Order %s nicht in openTrades", order_id)
