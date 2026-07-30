"""Live-Marktdaten-Stream über Alpaca.

Phase 2: Echtzeit-Bars/Trades/Quotes für Paper-Strategien. Open Source SDK,
kein Broker-Account nötig (Alpaca Paper-Konto reicht).

Schnittstelle bewusst klein gehalten — die Agent-Loop registriert Handler,
ruft `run()`, und Alpaca pusht Updates per WebSocket. Wenn Phase 4 (IBKR)
kommt, bauen wir denselben Vertrag mit `IBKRDataStream`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)

# Handler-Signaturen: async function nimmt das jeweilige Alpaca-Datenobjekt entgegen.
BarHandler = Callable[[Any], Awaitable[None]]
QuoteHandler = Callable[[Any], Awaitable[None]]
TradeHandler = Callable[[Any], Awaitable[None]]


class AlpacaDataStream:
    """WebSocket-Stream für Echtzeit-Marktdaten via Alpaca.

    Free-Tier liefert IEX-Daten (Echtzeit, aber nur IEX-Trades). Für Volle-
    SIP-Konsolidierung braucht's den Bezahltarif — gleicher Code, anderer Feed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        feed: str = "iex",
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.feed = feed

        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY und ALPACA_SECRET_KEY müssen gesetzt sein."
            )

        self._client: Any = None
        self._bar_symbols: list[str] = []
        self._quote_symbols: list[str] = []
        self._trade_symbols: list[str] = []
        self._bar_handler: BarHandler | None = None
        self._quote_handler: QuoteHandler | None = None
        self._trade_handler: TradeHandler | None = None

    def _require_client(self) -> Any:
        if self._client is None:
            try:
                from alpaca.data.enums import DataFeed
                from alpaca.data.live import StockDataStream
            except ImportError as e:
                raise ImportError(
                    "alpaca-py nicht installiert. `pip install -e .[alpaca]`."
                ) from e

            feed_map = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}
            feed = feed_map.get(self.feed.lower(), DataFeed.IEX)
            self._client = StockDataStream(
                api_key=self.api_key,
                secret_key=self.secret_key,
                feed=feed,
            )
        return self._client

    def on_bar(self, handler: BarHandler, *symbols: str) -> None:
        """Registriert einen Bar-Handler für die übergebenen Symbole."""
        if not symbols:
            raise ValueError("Mindestens ein Symbol nötig.")
        client = self._require_client()
        self._bar_handler = handler
        self._bar_symbols = list(symbols)
        client.subscribe_bars(handler, *symbols)

    def on_quote(self, handler: QuoteHandler, *symbols: str) -> None:
        """Registriert einen Quote-Handler (Bid/Ask)."""
        if not symbols:
            raise ValueError("Mindestens ein Symbol nötig.")
        client = self._require_client()
        self._quote_handler = handler
        self._quote_symbols = list(symbols)
        client.subscribe_quotes(handler, *symbols)

    def on_trade(self, handler: TradeHandler, *symbols: str) -> None:
        """Registriert einen Trade-Handler (Last Trade)."""
        if not symbols:
            raise ValueError("Mindestens ein Symbol nötig.")
        client = self._require_client()
        self._trade_handler = handler
        self._trade_symbols = list(symbols)
        client.subscribe_trades(handler, *symbols)

    def run(self) -> None:
        """Blockiert und liefert Stream-Events an die Handler. Strg-C beendet sauber."""
        if not (self._bar_handler or self._quote_handler or self._trade_handler):
            raise RuntimeError("Keine Handler registriert.")
        client = self._require_client()
        log.info(
            "AlpacaDataStream startet: bars=%s quotes=%s trades=%s feed=%s",
            self._bar_symbols,
            self._quote_symbols,
            self._trade_symbols,
            self.feed,
        )
        client.run()

    def stop(self) -> None:
        if self._client is not None:
            self._client.stop()
