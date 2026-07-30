"""Tiingo EOD-Provider — Roh-OHLCV + Corporate Actions, direkt per httpx.

Wir holen und speichern ROH-Kurse plus `divCash`/`splitFactor`. Die Adjustierung
passiert beim Lesen (quantrace.adjust) — so bleiben gecachte Daten point-in-time
korrekt, auch wenn später Dividenden/Splits dazukommen.

Retry/Backoff: Tiingo-Free erlaubt 50 req/h. 429 (Rate-Limit) und transiente
5xx/Netzfehler werden mit exponentiellem Backoff wiederholt.

Doku: https://www.tiingo.com/documentation/end-of-day
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date
from typing import Any

import httpx
import pandas as pd

log = logging.getLogger(__name__)

_BASE = "https://api.tiingo.com/tiingo/daily"
_RAW_FIELDS = ["open", "high", "low", "close", "volume", "divCash", "splitFactor"]
_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0  # Sekunden: 2, 4, 8, 16


def _token() -> str:
    tok = os.environ.get("TIINGO_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("TIINGO_TOKEN fehlt — für provider='tiingo' erforderlich.")
    return tok


def _to_raw_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Tiingo-JSON → DataFrame[DatetimeIndex, ROH-OHLCV + divCash/splitFactor]."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.set_index("date").sort_index()

    if not {"open", "high", "low", "close", "volume"} <= set(df.columns):
        return pd.DataFrame()

    # Corporate Actions sind bei vielen Zeilen nicht gesetzt → Defaults.
    if "divCash" not in df.columns:
        df["divCash"] = 0.0
    if "splitFactor" not in df.columns:
        df["splitFactor"] = 1.0
    df["divCash"] = df["divCash"].fillna(0.0)
    df["splitFactor"] = df["splitFactor"].fillna(1.0)

    return df[_RAW_FIELDS].astype(float)


def _get_with_retry(
    client: httpx.Client, url: str, params: dict, headers: dict | None = None
) -> list[dict[str, Any]]:
    """GET mit exponentiellem Backoff bei 429/5xx/Netzfehler."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.get(url, params=params, headers=headers or {})
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = _BACKOFF_BASE ** (attempt + 1)
                log.warning("Tiingo %s (status %s) — retry in %.0fs", url, resp.status_code, wait)
                time.sleep(wait)
                last_exc = httpx.HTTPStatusError(
                    f"status {resp.status_code}", request=resp.request, response=resp
                )
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            wait = _BACKOFF_BASE ** (attempt + 1)
            log.warning("Tiingo %s Netzfehler (%s) — retry in %.0fs", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Tiingo-Request endgültig fehlgeschlagen: {url}") from last_exc


def fetch_symbol_raw(
    symbol: str,
    start: date,
    end: date,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """ROH-OHLCV + Corporate Actions für ein Symbol."""
    params = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "format": "json",
        "resampleFreq": "daily",
    }
    # Token per Header statt URL-Query-Param — verhindert, dass der Token in
    # Server-Logs, Proxy-Traces und Fehlermeldungen auftaucht.
    auth_headers = {"Authorization": f"Token {_token()}"}
    owns = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        rows = _get_with_retry(client, f"{_BASE}/{symbol}/prices", params, auth_headers)
    finally:
        if owns:
            client.close()
    return _to_raw_frame(rows)


def fetch_universe_raw_ranges(
    ranges: dict[str, tuple[date, date]],
) -> dict[str, pd.DataFrame]:
    """ROH-OHLCV pro Symbol, jeweils nur über den **angeforderten** Datumsbereich
    (die tatsächliche Lücke) — ein geteilter httpx.Client für alle Symbole.

    Fehlende Symbole werden übersprungen (geloggt), nicht fatal — der Caller
    entscheidet. Damit fetcht der inkrementelle Lake-Pfad wirklich nur die Gaps
    statt jedes Mal die volle Range."""
    out: dict[str, pd.DataFrame] = {}
    with httpx.Client(timeout=30.0, headers={"Authorization": f"Token {_token()}"}) as client:
        for sym, (s, e) in ranges.items():
            try:
                frame = fetch_symbol_raw(sym, s, e, client=client)
            except Exception as exc:
                log.warning("Tiingo: Symbol %s konnte nicht geladen werden: %s", sym, exc)
                continue
            if not frame.empty:
                out[sym] = frame
    return out


def fetch_universe_raw(
    symbols: list[str],
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """ROH-OHLCV pro Symbol als dict[symbol -> DataFrame], alle über denselben
    Bereich. Convenience-Wrapper um :func:`fetch_universe_raw_ranges`."""
    return fetch_universe_raw_ranges({sym: (start, end) for sym in symbols})
