"""Tiingo-Crypto-Provider (#184, Schritt C) — Roh-OHLCV für 24/7-Märkte.

Gleiche Signaturen wie `quantrace.providers.tiingo`, damit `data_agent` beide
austauschbar benutzen kann. Der Endpunkt unterscheidet sich in vier Punkten,
und jeder davon ist eine Fehlerquelle, wenn man ihn übersieht:

1. **Anderer Pfad und Batch-Form.** ``/tiingo/crypto/prices?tickers=btcusd``
   statt ``/tiingo/daily/<sym>/prices`` — die Ticker sind ein Query-Parameter,
   kein Pfadsegment.
2. **Verschachtelte Antwort.** Die Kurse liegen unter ``priceData`` in einem
   Objekt pro Ticker, nicht als flache Liste.
3. **``resampleFreq=1day``**, nicht ``daily``.
4. **Keine Corporate Actions.** Crypto kennt weder Dividenden noch Splits;
   ``divCash``/``splitFactor`` werden mit den neutralen Werten 0.0 und 1.0
   geschrieben, damit Lake-Layout und Read-Time-Adjustierung dieselben bleiben
   wie bei Aktien. Ein eigenes Speicherformat für Crypto wäre die naheliegende
   Alternative und würde jede Leseschicht zwingen, zwei Fälle zu kennen.

**Was ich hier nicht verifizieren konnte:** die echte API. Die Tests fahren
gegen aufgezeichnete Antwortformen; ob Tiingo sich exakt so verhält, zeigt
erst der erste Fetch mit Token. Die Parser sind deshalb defensiv — eine
unerwartete Form ergibt einen leeren Frame (und damit einen sichtbaren
Lake-Miss), keine halb geparsten Kurse.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import httpx
import pandas as pd

# Token-Beschaffung und Retry/Backoff sind identisch — geteilt statt kopiert,
# damit eine Änderung am Rate-Limit-Verhalten nicht an einer Stelle vergessen
# wird.
from quantrace.providers.tiingo import _RAW_FIELDS, _get_with_retry, _token

log = logging.getLogger(__name__)

_BASE = "https://api.tiingo.com/tiingo/crypto/prices"

#: Wie viele Ticker pro Request. Der Endpunkt kann mehrere gleichzeitig, was
#: das 50-req/h-Limit von Tiingo-Free schont — aber eine zu lange URL wird
#: abgewiesen, und ein Fehler beträfe dann den ganzen Block.
_MAX_TICKERS_PER_REQUEST = 20


def _price_rows(payload: Any, ticker: str) -> list[dict[str, Any]]:
    """Die ``priceData``-Liste eines Tickers aus der Antwort holen.

    Defensiv: Tiingo liefert eine Liste von Ticker-Objekten, aber Groß-/
    Kleinschreibung und Reihenfolge sind nicht garantiert. Fehlt der Ticker,
    ist das kein Fehler — der Aufrufer behandelt einen leeren Frame als
    Lake-Miss.
    """
    if not isinstance(payload, list):
        return []
    wanted = ticker.lower()
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("ticker", "")).lower() != wanted:
            continue
        rows = entry.get("priceData")
        return rows if isinstance(rows, list) else []
    return []


def _to_raw_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Crypto-``priceData`` → DataFrame im selben Layout wie der EOD-Provider."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed").dt.tz_localize(None)
    # Tiingo liefert Crypto-Tagesstempel mit Uhrzeit (00:00:00+00:00). Für den
    # Lake zählt der Tag — ohne das Normalisieren stünden zwei Läufe mit
    # verschiedenen Zeitzonen als verschiedene Zeilen im Parquet.
    df["date"] = df["date"].dt.normalize()
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if not {"open", "high", "low", "close", "volume"} <= set(df.columns):
        return pd.DataFrame()

    # Crypto kennt keine Corporate Actions — neutrale Werte, damit das
    # Lake-Layout und quantrace.adjust unverändert bleiben.
    df["divCash"] = 0.0
    df["splitFactor"] = 1.0
    return df[_RAW_FIELDS].astype(float)


def fetch_symbol_raw(
    symbol: str,
    start: date,
    end: date,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """ROH-OHLCV für ein Crypto-Paar (z.B. ``BTCUSD``)."""
    frames = fetch_symbols_raw([symbol], start, end, client=client)
    return frames.get(symbol, pd.DataFrame())


def fetch_symbols_raw(
    symbols: list[str],
    start: date,
    end: date,
    client: httpx.Client | None = None,
) -> dict[str, pd.DataFrame]:
    """Mehrere Paare in einem Request — der Endpunkt kann das, EOD nicht."""
    if not symbols:
        return {}

    auth_headers = {"Authorization": f"Token {_token()}"}
    owns = client is None
    client = client or httpx.Client(timeout=30.0)
    out: dict[str, pd.DataFrame] = {}
    try:
        for i in range(0, len(symbols), _MAX_TICKERS_PER_REQUEST):
            chunk = symbols[i : i + _MAX_TICKERS_PER_REQUEST]
            params = {
                "tickers": ",".join(s.lower() for s in chunk),
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "resampleFreq": "1day",
            }
            payload = _get_with_retry(client, _BASE, params, auth_headers)
            for sym in chunk:
                frame = _to_raw_frame(_price_rows(payload, sym))
                if not frame.empty:
                    out[sym] = frame
    finally:
        if owns:
            client.close()
    return out


def fetch_universe_raw_ranges(
    ranges: dict[str, tuple[date, date]],
) -> dict[str, pd.DataFrame]:
    """ROH-OHLCV pro Symbol über den jeweils angeforderten Bereich.

    Gleiche Semantik wie im EOD-Provider: fehlende Symbole werden übersprungen
    und geloggt, nicht fatal.

    Anders als dort werden Symbole mit **identischem** Zeitfenster gebündelt —
    im Normalfall (alle Symbole eines Universums werden gemeinsam
    aktualisiert) ist das ein einziger Request statt einem pro Paar, was bei
    50 req/h spürbar ist.
    """
    if not ranges:
        return {}

    by_window: dict[tuple[date, date], list[str]] = {}
    for sym, window in ranges.items():
        by_window.setdefault(window, []).append(sym)

    out: dict[str, pd.DataFrame] = {}
    with httpx.Client(timeout=30.0) as client:
        for (start, end), syms in by_window.items():
            try:
                out.update(fetch_symbols_raw(sorted(syms), start, end, client=client))
            except Exception as exc:  # noqa: BLE001 — ein Block darf den Rest nicht kippen
                log.warning(
                    "Tiingo-Crypto: %s konnte nicht geladen werden (%s..%s): %s",
                    sorted(syms),
                    start,
                    end,
                    exc,
                )
    missing = sorted(set(ranges) - set(out))
    if missing:
        log.warning("Tiingo-Crypto: keine Daten für %s", missing)
    return out
