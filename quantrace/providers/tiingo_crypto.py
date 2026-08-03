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

**Gegen die echte API geprüft am 2026-08-03.** Alle vier Punkte oben haben
gehalten. Zwei Dinge, die dabei *nicht* gehalten haben, stehen als Warnung hier:

- ``_MAX_TICKERS_PER_REQUEST`` stand auf 20 — geraten, mit einer erfundenen
  Begründung („URL zu lang"). Tiingo erlaubt 5 und sagt das im Fehler-Body.
- Ein gescheiterter Block ließ **alle** seine Ticker leer ausgehen, obwohl
  jeder einzelne abrufbar war. Deshalb jetzt der Rückfall auf Einzelabrufe.

Die Parser bleiben defensiv: eine unerwartete Form ergibt einen leeren Frame
(und damit einen sichtbaren Lake-Miss), keine halb geparsten Kurse.
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

#: Wie viele Ticker pro Request — **von Tiingo vorgegeben**, nicht geschätzt.
#:
#: Der erste echte Lauf am 2026-08-03 mit neun Tickern ergab:
#:
#:     400  {"detail":"Error: A limit of 5 tickers may be requested at a time"}
#:
#: Hier stand vorher 20, mit der Begründung „eine zu lange URL wird abgewiesen".
#: Beides war geraten und beides war falsch. Die Zahl gehört zur API und nicht
#: zu unserer Vorstellung von ihr — wer sie erhöht, muss die Antwort oben
#: widerlegen können.
_MAX_TICKERS_PER_REQUEST = 5


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


def _params(tickers: list[str], start: date, end: date) -> dict[str, str]:
    return {
        "tickers": ",".join(s.lower() for s in tickers),
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "resampleFreq": "1day",
    }


def _harvest(
    payload: Any, tickers: list[str], out: dict[str, pd.DataFrame]
) -> None:
    for sym in tickers:
        frame = _to_raw_frame(_price_rows(payload, sym))
        if not frame.empty:
            out[sym] = frame


def _one_by_one(
    client: httpx.Client,
    tickers: list[str],
    start: date,
    end: date,
    headers: dict[str, str],
    out: dict[str, pd.DataFrame],
) -> None:
    """Rückfall nach einem gescheiterten Block: jeden Ticker einzeln.

    Fällt selbst **nicht** weiter zurück — ein einzelner Ticker, der scheitert,
    ist das Ende der Fahnenstange und wird übersprungen.
    """
    for sym in tickers:
        try:
            payload = _get_with_retry(client, _BASE, _params([sym], start, end), headers)
        except Exception as exc:  # noqa: BLE001 — ein Symbol darf den Rest nicht kippen
            log.warning("Tiingo-Crypto: %s einzeln fehlgeschlagen: %s", sym, exc)
            continue
        _harvest(payload, [sym], out)


def fetch_symbols_raw(
    symbols: list[str],
    start: date,
    end: date,
    client: httpx.Client | None = None,
) -> dict[str, pd.DataFrame]:
    """Mehrere Paare pro Request — der Endpunkt kann das, EOD nicht.

    Scheitert ein Block, werden seine Ticker **einzeln** nachgeholt. Das kostet
    im Fehlerfall mehr Requests, verhindert aber den Ausfall, den der erste
    echte Lauf gezeigt hat: ein Verstoß gegen ein Limit ließ alle neun Symbole
    leer ausgehen, obwohl jedes einzelne abrufbar gewesen wäre. Ein Batch ist
    eine Optimierung — er darf nicht die Zuverlässigkeit bestimmen.
    """
    if not symbols:
        return {}

    auth_headers = {"Authorization": f"Token {_token()}"}
    owns = client is None
    client = client or httpx.Client(timeout=30.0)
    out: dict[str, pd.DataFrame] = {}
    try:
        for i in range(0, len(symbols), _MAX_TICKERS_PER_REQUEST):
            chunk = symbols[i : i + _MAX_TICKERS_PER_REQUEST]
            try:
                payload = _get_with_retry(client, _BASE, _params(chunk, start, end), auth_headers)
            except Exception as exc:  # noqa: BLE001 — Block-Fehler → Einzelabruf
                log.warning(
                    "Tiingo-Crypto: Block %s fehlgeschlagen (%s) — jetzt einzeln",
                    chunk,
                    exc,
                )
                _one_by_one(client, chunk, start, end, auth_headers, out)
                continue
            _harvest(payload, chunk, out)
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
