"""FRED-Provider — Makro-Zeitreihen der St. Louis Fed, direkt per httpx.

Warum das hier steht: die Regime-Engine (``quantrace/regime/``) läuft heute
ausschließlich auf Kursen. Zinsstruktur und Credit-Spreads sind aber die
klassischen Regime-Treiber — eine inverse Zinskurve und ausweitende Spreads
sagen mehr über den Zustand des Marktes als die Vola des S&P allein.

**Point-in-time-Disziplin.** Makro-Reihen werden revidiert: das BIP für Q1
sieht im Juli anders aus als im April. Ein Backtest, der die *heutige* Fassung
einer Reihe verwendet, kennt Zahlen, die es zum Handelszeitpunkt noch nicht
gab — Look-ahead durch die Hintertür. Deshalb:

* ``fetch_series`` liefert die **aktuelle** Fassung und ist für Reihen gedacht,
  die nicht revidiert werden (Marktpreise wie DGS10, BAMLH0A0HYM2, VIXCLS).
* ``fetch_series_vintage`` nutzt FREDs ALFRED-Schnittstelle (``realtime_start``)
  und liefert den Stand, wie er an einem bestimmten Tag bekannt war. Für alles
  Revidierte (GDP, CPI, UNRATE) ist das der einzig zulässige Weg.

``REVISED_SERIES`` markiert die bekannten Fallen; ``fetch_series`` warnt, wenn
eine davon ohne Vintage geholt wird. Die Entscheidung bleibt beim Aufrufer —
für explorative Analysen ist die aktuelle Fassung legitim, im Backtest nicht.

API-Key: kostenlos unter https://fred.stlouisfed.org/docs/api/api_key.html,
Env-Var ``FRED_API_KEY``. Rate-Limit 120 req/min.

Doku: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
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

_BASE = "https://api.stlouisfed.org/fred"
_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0  # Sekunden: 2, 4, 8, 16

#: Reihen, die nachträglich revidiert werden. Ohne Vintage im Backtest = Look-ahead.
REVISED_SERIES = frozenset(
    {
        "GDP", "GDPC1", "GDPPOT",           # BIP
        "CPIAUCSL", "CPILFESL", "PCEPI",    # Inflation
        "UNRATE", "PAYEMS", "ICSA",         # Arbeitsmarkt
        "INDPRO", "RSAFS", "HOUST",         # Aktivität
        "M2SL", "M1SL",                     # Geldmenge
    }
)

#: Kuratierte Regime-Features. Alles Marktpreise — werden nicht revidiert und
#: sind damit ohne Vintage backtest-tauglich.
REGIME_SERIES: dict[str, str] = {
    "DGS10": "10Y Treasury Constant Maturity",
    "DGS2": "2Y Treasury Constant Maturity",
    "DGS3MO": "3M Treasury Constant Maturity",
    "T10Y2Y": "10Y minus 2Y Spread (Kurvensteilheit)",
    "T10Y3M": "10Y minus 3M Spread (Rezessionsindikator)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Option-Adjusted Spread",
    "BAMLC0A0CM": "ICE BofA US Corporate Option-Adjusted Spread",
    "VIXCLS": "CBOE Volatility Index",
    "DTWEXBGS": "Trade Weighted US Dollar Index (Broad)",
    "DFF": "Effective Federal Funds Rate",
}


def _api_key() -> str:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FRED_API_KEY fehlt — kostenlos unter "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key


def _get_with_retry(client: httpx.Client, url: str, params: dict) -> dict[str, Any]:
    """GET mit exponentiellem Backoff bei 429/5xx/Netzfehler — wie im Tiingo-Client."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.get(url, params=params)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = _BACKOFF_BASE ** (attempt + 1)
                log.warning("FRED %s (status %s) — retry in %.0fs", url, resp.status_code, wait)
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
            log.warning("FRED %s Netzfehler (%s) — retry in %.0fs", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"FRED-Request endgültig fehlgeschlagen: {url}") from last_exc


def _to_series(payload: dict[str, Any], series_id: str) -> pd.Series:
    """FRED-JSON → float-Series mit DatetimeIndex.

    FRED kodiert fehlende Werte als ``"."`` — die werden zu NaN und **nicht**
    interpoliert. Eine Reihe mit Lücken ist ehrlicher als eine erfundene.
    """
    obs = payload.get("observations") or []
    if not obs:
        return pd.Series(dtype=float, name=series_id)

    df = pd.DataFrame(obs)
    if "date" not in df.columns or "value" not in df.columns:
        return pd.Series(dtype=float, name=series_id)

    df["date"] = pd.to_datetime(df["date"])
    values = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    return pd.Series(values.to_numpy(), index=df["date"], name=series_id).sort_index()


def fetch_series(
    series_id: str,
    start: date,
    end: date,
    client: httpx.Client | None = None,
) -> pd.Series:
    """Aktuelle Fassung einer FRED-Reihe.

    Für Marktpreise (Zinsen, Spreads, VIX) korrekt. Für revidierte Reihen warnt
    die Funktion — dort gehört ``fetch_series_vintage`` hin.
    """
    if series_id.upper() in REVISED_SERIES:
        log.warning(
            "FRED-Reihe %s wird nachträglich revidiert. Im Backtest ist das "
            "Look-ahead — nutze fetch_series_vintage(). Für Exploration ok.",
            series_id,
        )
    return _fetch(series_id, start, end, client=client)


def fetch_series_vintage(
    series_id: str,
    start: date,
    end: date,
    as_of: date,
    client: httpx.Client | None = None,
) -> pd.Series:
    """Die Reihe so, wie sie am ``as_of``-Tag bekannt war (ALFRED-Vintage).

    Das ist der point-in-time-korrekte Weg für alles, was revidiert wird.
    """
    return _fetch(series_id, start, end, as_of=as_of, client=client)


def _fetch(
    series_id: str,
    start: date,
    end: date,
    as_of: date | None = None,
    client: httpx.Client | None = None,
) -> pd.Series:
    params: dict[str, Any] = {
        "series_id": series_id,
        "api_key": _api_key(),
        "file_type": "json",
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
    }
    if as_of is not None:
        # realtime_start == realtime_end == as_of ⇒ genau die damals gültige Fassung
        params["realtime_start"] = as_of.isoformat()
        params["realtime_end"] = as_of.isoformat()

    owns = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        payload = _get_with_retry(client, f"{_BASE}/series/observations", params)
    finally:
        if owns:
            client.close()
    return _to_series(payload, series_id)


def fetch_many(
    series_ids: list[str],
    start: date,
    end: date,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Mehrere Reihen als DataFrame — ein geteilter Client, Spalten = series_id.

    Reihen, die scheitern, werden übersprungen und geloggt (nicht fatal) —
    dasselbe Muster wie ``tiingo.fetch_universe_raw_ranges``. Der Index ist die
    Vereinigung aller Beobachtungstage; unterschiedliche Frequenzen (täglich vs.
    monatlich) erzeugen also NaN-Lücken. Bewusst nicht geforwardfillt: wer
    tägliche Features braucht, muss das explizit und kausal tun.
    """
    out: dict[str, pd.Series] = {}
    with httpx.Client(timeout=30.0) as client:
        for sid in series_ids:
            try:
                series = (
                    fetch_series_vintage(sid, start, end, as_of, client=client)
                    if as_of is not None
                    else fetch_series(sid, start, end, client=client)
                )
            except Exception as exc:
                log.warning("FRED: Reihe %s konnte nicht geladen werden: %s", sid, exc)
                continue
            if not series.empty:
                out[sid] = series
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_index()


def fetch_regime_features(
    start: date,
    end: date,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Die kuratierten Regime-Reihen aus ``REGIME_SERIES``.

    Alle darin sind Marktpreise und werden nicht revidiert — ``as_of`` ist
    deshalb optional und normalerweise unnötig.
    """
    return fetch_many(list(REGIME_SERIES), start, end, as_of=as_of)
