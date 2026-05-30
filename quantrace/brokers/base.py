"""Broker-Interface — broker-agnostisch, damit Paper und Live austauschbar bleiben."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"


@dataclass(slots=True)
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    tif: str = "DAY"
    client_id: str | None = None


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float


class Broker(ABC):
    """Minimaler Vertrag. Erweitern, wenn Phase 4 (Paper Trading) startet."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def account_value(self) -> float: ...

    @abstractmethod
    def positions(self) -> list[Position]: ...

    @abstractmethod
    def submit(self, order: Order) -> str:
        """Reicht eine Order ein. Gibt eine Broker-Order-ID zurück."""

    @abstractmethod
    def cancel(self, order_id: str) -> None: ...
