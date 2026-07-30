"""Alpaca-Adapter über alpaca-py.

Alpaca ist Open Source, Free Tier, und Paper-Trading-fähig ohne KYC. Perfekt
für Phase 2 (Strategie gegen Echtzeit-Markt testen) bevor IBKR live geht.

Endpoints:
    https://paper-api.alpaca.markets       Paper Trading (default)
    https://api.alpaca.markets              Live Trading (allow_live=True nötig)

Env-Vars:
    ALPACA_API_KEY        Paper-Key (oder Live, wenn allow_live=True)
    ALPACA_SECRET_KEY     Paper-Secret (analog)
    ALPACA_PAPER          "true" (default) oder "false"
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from quantrace.brokers.base import Broker, Order, OrderSide, OrderType, Position

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class AlpacaBroker(Broker):
    """Alpaca Paper/Live-Broker. Default: Paper.

    Phase-2-Komfort: ein einziger Account erlaubt sowohl Paper-Strategien
    als auch (später, mit explizitem `allow_live=True`) Live-Execution.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool | None = None,
        allow_live: bool = False,
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        env_paper = os.environ.get("ALPACA_PAPER", "true").strip().lower()
        self.paper = paper if paper is not None else env_paper != "false"
        self.allow_live = allow_live

        if not self.paper and not self.allow_live:
            raise RuntimeError(
                "Alpaca Live-Mode angefordert ohne allow_live=True. "
                "Setze allow_live=True nur nach expliziter menschlicher Freigabe."
            )
        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY und ALPACA_SECRET_KEY müssen gesetzt sein."
            )

        self._client: Any = None

    def _require_client(self) -> Any:
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient
            except ImportError as e:
                raise ImportError(
                    "alpaca-py nicht installiert. `pip install -e .[alpaca]`."
                ) from e
            self._client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper,
            )
        return self._client

    def connect(self) -> None:
        client = self._require_client()
        acct = client.get_account()
        log.info(
            "Alpaca connected: paper=%s status=%s equity=%s",
            self.paper,
            getattr(acct, "status", "?"),
            getattr(acct, "equity", "?"),
        )

    def disconnect(self) -> None:
        self._client = None

    def is_connected(self) -> bool:
        return self._client is not None

    def account_value(self) -> float:
        client = self._require_client()
        acct = client.get_account()
        return float(acct.equity)

    def positions(self) -> list[Position]:
        client = self._require_client()
        out: list[Position] = []
        for p in client.get_all_positions():
            out.append(
                Position(
                    symbol=p.symbol,
                    quantity=float(p.qty),
                    avg_cost=float(p.avg_entry_price),
                    market_value=float(p.market_value),
                )
            )
        return out

    def submit(self, order: Order) -> str:
        if not self.paper and not self.allow_live:
            raise PermissionError("Live-Order ohne allow_live=True nicht erlaubt.")
        client = self._require_client()
        request = _to_alpaca_request(order)
        submitted = client.submit_order(order_data=request)
        return str(submitted.id)

    def cancel(self, order_id: str) -> None:
        client = self._require_client()
        client.cancel_order_by_id(order_id)


def _to_alpaca_request(order: Order) -> Any:
    """Mappt unser Order-Modell auf alpaca-py-Requests."""
    from alpaca.trading.enums import OrderSide as AlpacaSide
    from alpaca.trading.requests import (
        LimitOrderRequest,
        MarketOrderRequest,
        StopOrderRequest,
    )

    side = AlpacaSide.BUY if order.side is OrderSide.BUY else AlpacaSide.SELL
    tif = _to_tif(order.tif)

    if order.order_type is OrderType.MARKET:
        return MarketOrderRequest(
            symbol=order.symbol,
            qty=order.quantity,
            side=side,
            time_in_force=tif,
            client_order_id=order.client_id,
        )
    if order.order_type is OrderType.LIMIT:
        if order.limit_price is None:
            raise ValueError("LIMIT-Order braucht limit_price.")
        return LimitOrderRequest(
            symbol=order.symbol,
            qty=order.quantity,
            side=side,
            time_in_force=tif,
            limit_price=order.limit_price,
            client_order_id=order.client_id,
        )
    if order.order_type is OrderType.STOP:
        if order.stop_price is None:
            raise ValueError("STOP-Order braucht stop_price.")
        return StopOrderRequest(
            symbol=order.symbol,
            qty=order.quantity,
            side=side,
            time_in_force=tif,
            stop_price=order.stop_price,
            client_order_id=order.client_id,
        )
    raise ValueError(f"Unbekannter OrderType: {order.order_type}")


def _to_tif(tif: str) -> Any:
    from alpaca.trading.enums import TimeInForce

    mapping = {
        "DAY": TimeInForce.DAY,
        "GTC": TimeInForce.GTC,
        "IOC": TimeInForce.IOC,
        "FOK": TimeInForce.FOK,
        "OPG": TimeInForce.OPG,
        "CLS": TimeInForce.CLS,
    }
    return mapping.get(tif.upper(), TimeInForce.DAY)
