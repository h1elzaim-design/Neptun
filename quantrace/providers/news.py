"""News-Provider — normalisierte Schlagzeilen als *Kontext* für den Research-Agenten.

Bewusst **kein** numerisches Backtest-Feature
---------------------------------------------
News fließen in die Hypothesen-Generierung, nicht in die Signalberechnung. Der
Grund ist Look-ahead: ein Sentiment-Feature bräuchte strikte Point-in-time-
Alignment gegen die Kursreihen und ein eigenes Bias-Audit (Publikationszeit vs.
Handelszeit, Revisionen, Backfill der Provider-Historie). Das ist ein eigener,
größerer Track — hier geht es um Kontext für einen Menschen-plus-LLM-Prozess,
der ohnehin unter Governance steht.

Trotzdem gilt die Point-in-time-Disziplin: jedes :class:`NewsItem` trägt
``published_at``, und :func:`filter_point_in_time` ist der harte Filter, durch
den alles muss, bevor es in einen Prompt darf. Ein Prompt, der Schlagzeilen von
*nach* dem Analysedatum sieht, erzeugt Hypothesen mit eingebautem Hindsight —
und die sähen im Backtest hervorragend aus.

Provider
--------
``gdelt``
    **Keyless.** GDELT DOC 2.0 — riesige Abdeckung, rauschig, kein Sentiment.
    Der Default für „ich will Schlagzeilen ohne Account". Scoring läuft dann
    über ``agents._llm`` (siehe :mod:`agents.news_context`).
``alpha_vantage``
    Braucht ``ALPHA_VANTAGE_API_KEY``. Liefert Ticker-Level-Sentiment mit
    Richtung **und** Magnitude. Free-Tier: 25 Requests/Tag — genug für einen
    täglichen Research-Zyklus, nichts für Realtime.
``off``
    Kein Fetch. Default, damit sich ohne bewusstes Opt-in nichts ändert.

Credentials
-----------
Alpha Vantage akzeptiert den Key **nur als Query-Parameter** (kein Header-Auth,
anders als Tiingo). Deshalb wird hier nie eine volle URL geloggt oder in eine
Fehlermeldung gehoben — :func:`_redact` ersetzt den Key durch ``***``, und
Log-Zeilen nennen nur Host und Pfad.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

log = logging.getLogger(__name__)

#: Unterstützte Provider. ``off`` ist der Default — Opt-in per Env.
NEWS_PROVIDERS = ("off", "gdelt", "alpha_vantage")

_GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
_ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"

_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0  # Sekunden: 2, 4, 8, 16
_DEFAULT_LIMIT = 50


@dataclass(frozen=True, slots=True)
class NewsItem:
    """Eine normalisierte Schlagzeile — providerunabhängig.

    Attributes
    ----------
    published_at:
        **Timezone-aware UTC.** Das ist das Feld, an dem die
        Point-in-time-Disziplin hängt; naive Datetimes werden abgelehnt.
    symbols:
        Getaggte Ticker (leer, wenn der Provider nicht taggt — GDELT tut das nicht).
    sentiment:
        −1…+1, falls der Provider eines liefert. ``None`` heißt „nicht bewertet",
        **nicht** „neutral" — der Unterschied entscheidet, ob das LLM-Scoring
        drüberlaufen muss.
    """

    published_at: datetime
    headline: str
    source: str
    url: str
    provider: str
    symbols: tuple[str, ...] = ()
    summary: str = ""
    sentiment: float | None = None
    sentiment_label: str | None = None

    def __post_init__(self) -> None:
        if self.published_at.tzinfo is None:
            raise ValueError(
                "published_at muss timezone-aware sein (UTC) — naive Zeitstempel "
                "machen den Point-in-time-Filter unzuverlässig"
            )
        if not self.headline.strip():
            raise ValueError("headline darf nicht leer sein")

    @property
    def is_scored(self) -> bool:
        return self.sentiment is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "published_at": self.published_at.astimezone(UTC).isoformat(),
            "headline": self.headline,
            "source": self.source,
            "url": self.url,
            "provider": self.provider,
            "symbols": list(self.symbols),
            "summary": self.summary,
            "sentiment": self.sentiment,
            "sentiment_label": self.sentiment_label,
        }


@dataclass(frozen=True, slots=True)
class NewsFetchResult:
    """Fetch-Ergebnis samt Herkunft — was gefiltert wurde, bleibt sichtbar."""

    items: list[NewsItem]
    provider: str
    as_of: datetime | None = None
    dropped_future: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "provider": self.provider,
            "as_of": self.as_of.astimezone(UTC).isoformat() if self.as_of else None,
            "n_items": len(self.items),
            "dropped_future": self.dropped_future,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------


def configured_provider() -> str:
    """Provider aus ``NEWS_PROVIDER``. Unbekannt oder leer → ``off``."""
    raw = os.environ.get("NEWS_PROVIDER", "").strip().lower()
    if raw not in NEWS_PROVIDERS:
        if raw:
            log.warning("NEWS_PROVIDER=%r unbekannt — News-Layer bleibt aus.", raw)
        return "off"
    return raw


def _alpha_vantage_key() -> str:
    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ALPHA_VANTAGE_API_KEY fehlt — für NEWS_PROVIDER='alpha_vantage' "
            "erforderlich. Keyless-Alternative: NEWS_PROVIDER='gdelt'."
        )
    return key


def _redact(text: str) -> str:
    """API-Keys aus Text entfernen, bevor er in ein Log oder eine Exception geht."""
    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    return text.replace(key, "***") if key else text


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _get_with_retry(
    client: httpx.Client, url: str, params: dict[str, Any], *, label: str
) -> Any:
    """GET mit exponentiellem Backoff bei 429/5xx/Netzfehler.

    Loggt nur ``label`` — nie die volle URL, damit ein Query-Param-Key
    (Alpha Vantage) nicht in Logs landet.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.get(url, params=params)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = _BACKOFF_BASE ** (attempt + 1)
                log.warning("%s (status %s) — retry in %.0fs", label, resp.status_code, wait)
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
            log.warning("%s Netzfehler (%s) — retry in %.0fs", label, _redact(str(exc)), wait)
            time.sleep(wait)
    raise RuntimeError(f"News-Request endgültig fehlgeschlagen: {label}") from last_exc


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse_gdelt_datetime(raw: str) -> datetime | None:
    """GDELT liefert ``YYYYMMDDTHHMMSSZ``."""
    try:
        return datetime.strptime(raw.strip(), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except (ValueError, AttributeError):
        return None


def _parse_alpha_vantage_datetime(raw: str) -> datetime | None:
    """Alpha Vantage liefert ``YYYYMMDDTHHMMSS`` (UTC, ohne Suffix)."""
    try:
        return datetime.strptime(raw.strip(), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    except (ValueError, AttributeError):
        return None


def parse_gdelt(payload: Any, *, symbols: Sequence[str] = ()) -> list[NewsItem]:
    """GDELT-DOC-Antwort → NewsItems. Kaputte Einträge werden übersprungen.

    GDELT taggt keine Ticker; die abgefragten Symbole werden übernommen, damit
    der Kontext weiß, wonach gesucht wurde — das ist eine *Query*-Zuordnung,
    keine Entitätserkennung, und wird im Prompt auch so benannt.
    """
    articles = (payload or {}).get("articles") if isinstance(payload, dict) else None
    if not isinstance(articles, list):
        return []

    tagged = tuple(s.strip().upper() for s in symbols if s and s.strip())
    items: list[NewsItem] = []
    for art in articles:
        if not isinstance(art, dict):
            continue
        published = _parse_gdelt_datetime(str(art.get("seendate", "")))
        title = str(art.get("title") or "").strip()
        if published is None or not title:
            continue
        items.append(
            NewsItem(
                published_at=published,
                headline=title,
                source=str(art.get("domain") or "unknown"),
                url=str(art.get("url") or ""),
                provider="gdelt",
                symbols=tagged,
            )
        )
    return items


def _alpha_vantage_label(score: float) -> str:
    """Alpha-Vantage-Schwellen (Doku: Bearish < −0.35 … Bullish > 0.35)."""
    if score <= -0.35:
        return "bearish"
    if score <= -0.15:
        return "somewhat_bearish"
    if score < 0.15:
        return "neutral"
    if score < 0.35:
        return "somewhat_bullish"
    return "bullish"


def parse_alpha_vantage(payload: Any, *, symbols: Sequence[str] = ()) -> list[NewsItem]:
    """Alpha-Vantage-NEWS_SENTIMENT-Antwort → NewsItems.

    Wenn ein Artikel Ticker-Level-Sentiment für ein abgefragtes Symbol trägt,
    gewinnt das gegen das Artikel-Gesamtsentiment — die Frage ist „was heißt
    das für *dieses* Symbol", nicht „wie ist die Stimmung im Artikel".
    """
    if not isinstance(payload, dict):
        return []
    feed = payload.get("feed")
    if not isinstance(feed, list):
        return []

    wanted = {s.strip().upper() for s in symbols if s and s.strip()}
    items: list[NewsItem] = []
    for art in feed:
        if not isinstance(art, dict):
            continue
        published = _parse_alpha_vantage_datetime(str(art.get("time_published", "")))
        title = str(art.get("title") or "").strip()
        if published is None or not title:
            continue

        ticker_scores: dict[str, float] = {}
        for entry in art.get("ticker_sentiment") or []:
            if not isinstance(entry, dict):
                continue
            sym = str(entry.get("ticker") or "").strip().upper()
            try:
                ticker_scores[sym] = float(entry.get("ticker_sentiment_score"))
            except (TypeError, ValueError):
                continue

        relevant = sorted(ticker_scores.keys() & wanted) if wanted else sorted(ticker_scores)
        if relevant:
            score = sum(ticker_scores[s] for s in relevant) / len(relevant)
        else:
            try:
                score = float(art.get("overall_sentiment_score"))
            except (TypeError, ValueError):
                score = None  # type: ignore[assignment]

        items.append(
            NewsItem(
                published_at=published,
                headline=title,
                source=str(art.get("source") or "unknown"),
                url=str(art.get("url") or ""),
                provider="alpha_vantage",
                symbols=tuple(relevant) or tuple(sorted(wanted)),
                summary=str(art.get("summary") or "")[:500],
                sentiment=score,
                sentiment_label=_alpha_vantage_label(score) if score is not None else None,
            )
        )
    return items


# ---------------------------------------------------------------------------
# Point-in-time
# ---------------------------------------------------------------------------


def filter_point_in_time(
    items: Iterable[NewsItem], as_of: datetime
) -> tuple[list[NewsItem], int]:
    """(Items **vor** ``as_of``, Anzahl verworfener Zukunfts-Items).

    Der harte Look-ahead-Guard. Strikt ``<``, nicht ``<=``: eine Schlagzeile,
    die exakt auf dem Analysezeitpunkt liegt, ist zu diesem Zeitpunkt nicht
    verlässlich verfügbar gewesen.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of muss timezone-aware sein (UTC)")
    kept, dropped = [], 0
    for item in items:
        if item.published_at < as_of:
            kept.append(item)
        else:
            dropped += 1
    kept.sort(key=lambda i: i.published_at, reverse=True)
    return kept, dropped


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_news(
    symbols: Sequence[str],
    *,
    provider: str | None = None,
    as_of: datetime | None = None,
    limit: int = _DEFAULT_LIMIT,
    client: httpx.Client | None = None,
) -> NewsFetchResult:
    """Schlagzeilen zu ``symbols``, point-in-time gefiltert.

    Parameters
    ----------
    provider:
        ``gdelt`` | ``alpha_vantage`` | ``off``. ``None`` → ``NEWS_PROVIDER``.
    as_of:
        Analysezeitpunkt (UTC, aware). Alles ab diesem Zeitpunkt fliegt raus.
        ``None`` → jetzt.

    Raises
    ------
    ValueError
        Unbekannter Provider, leere Symbolliste bei aktivem Provider.
    RuntimeError
        Fehlende Credentials oder endgültig fehlgeschlagener Request.
    """
    chosen = (provider or configured_provider()).strip().lower()
    if chosen not in NEWS_PROVIDERS:
        raise ValueError(f"provider muss eine aus {NEWS_PROVIDERS} sein, war {chosen!r}")

    cutoff = as_of or datetime.now(UTC)
    if cutoff.tzinfo is None:
        raise ValueError("as_of muss timezone-aware sein (UTC)")

    if chosen == "off":
        return NewsFetchResult(
            items=[],
            provider="off",
            as_of=cutoff,
            warnings=["News-Layer ist aus (NEWS_PROVIDER=off)."],
        )

    tickers = [s.strip().upper() for s in symbols if s and s.strip()]
    if not tickers:
        raise ValueError("symbols ist leer — ohne Symbole gibt es nichts zu suchen")

    owns = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        if chosen == "gdelt":
            raw = _fetch_gdelt(client, tickers, cutoff, limit)
            items = parse_gdelt(raw, symbols=tickers)
        else:
            raw = _fetch_alpha_vantage(client, tickers, cutoff, limit)
            items = parse_alpha_vantage(raw, symbols=tickers)
    finally:
        if owns:
            client.close()

    kept, dropped = filter_point_in_time(items, cutoff)
    warnings: list[str] = []
    if dropped:
        warnings.append(
            f"{dropped} Schlagzeile(n) nach {cutoff.date()} verworfen (Point-in-time-Filter)."
        )
    if not kept:
        warnings.append("Keine Schlagzeilen im Fenster — der Kontext bleibt leer.")

    return NewsFetchResult(
        items=kept[:limit],
        provider=chosen,
        as_of=cutoff,
        dropped_future=dropped,
        warnings=warnings,
    )


def _fetch_gdelt(
    client: httpx.Client, symbols: Sequence[str], as_of: datetime, limit: int
) -> Any:
    """GDELT DOC 2.0, keyless. Suchfenster: 7 Tage vor ``as_of``."""
    start = as_of.timestamp() - 7 * 24 * 3600
    params = {
        "query": " OR ".join(f'"{s}"' for s in symbols) + " sourcelang:english",
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(min(max(limit, 1), 250)),
        "startdatetime": datetime.fromtimestamp(start, UTC).strftime("%Y%m%d%H%M%S"),
        "enddatetime": as_of.strftime("%Y%m%d%H%M%S"),
        "sort": "datedesc",
    }
    return _get_with_retry(client, _GDELT_BASE, params, label="gdelt/doc")


def _fetch_alpha_vantage(
    client: httpx.Client, symbols: Sequence[str], as_of: datetime, limit: int
) -> Any:
    """Alpha Vantage NEWS_SENTIMENT. Key nur im Query-Param (Provider-Limit)."""
    window_start = datetime.fromtimestamp(as_of.timestamp() - 7 * 24 * 3600, UTC)
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ",".join(symbols),
        "time_from": window_start.strftime("%Y%m%dT%H%M"),
        "time_to": as_of.strftime("%Y%m%dT%H%M"),
        "limit": str(min(max(limit, 1), 1000)),
        "sort": "LATEST",
        "apikey": _alpha_vantage_key(),
    }
    payload = _get_with_retry(client, _ALPHA_VANTAGE_BASE, params, label="alpha_vantage/news")

    # Alpha Vantage antwortet auf Rate-Limit und Fehler mit HTTP 200 und einem
    # Hinweis-Feld — ohne diese Prüfung sähe das wie „keine News" aus.
    if isinstance(payload, dict):
        for key in ("Note", "Information", "Error Message"):
            if payload.get(key):
                raise RuntimeError(
                    f"Alpha Vantage: {_redact(str(payload[key]))[:300]}"
                )
    return payload


__all__ = [
    "NEWS_PROVIDERS",
    "NewsFetchResult",
    "NewsItem",
    "configured_provider",
    "fetch_news",
    "filter_point_in_time",
    "parse_alpha_vantage",
    "parse_gdelt",
]
