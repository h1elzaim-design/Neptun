"""News-Provider — Parsing, Point-in-time-Guard, Credential-Hygiene.

Kein Netz: die Provider-Antworten sind Fixtures, der HTTP-Client wird
gemockt. Der wichtigste Test der Datei ist der **Look-ahead-Guard** — eine
Schlagzeile von nach dem Analysezeitpunkt darf niemals in einen Prompt.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from quantrace.providers.news import (
    NEWS_PROVIDERS,
    NewsItem,
    configured_provider,
    fetch_news,
    filter_point_in_time,
    parse_alpha_vantage,
    parse_gdelt,
)

AS_OF = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


def _item(days_before: float, headline: str = "Schlagzeile", **kw) -> NewsItem:
    return NewsItem(
        published_at=AS_OF - timedelta(days=days_before),
        headline=headline,
        source="example.com",
        url="https://example.com/a",
        provider="test",
        **kw,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_naive_timestamps_are_rejected():
    """Ohne Zeitzone ist der Point-in-time-Filter nicht vertrauenswürdig."""
    with pytest.raises(ValueError, match="timezone-aware"):
        NewsItem(
            published_at=datetime(2026, 7, 1, 12, 0),
            headline="x",
            source="s",
            url="u",
            provider="test",
        )


def test_empty_headline_rejected():
    with pytest.raises(ValueError, match="headline"):
        NewsItem(published_at=AS_OF, headline="   ", source="s", url="u", provider="test")


def test_unscored_is_not_neutral():
    """`None` heißt nicht bewertet — der Unterschied steuert das LLM-Scoring."""
    assert _item(1).is_scored is False
    assert _item(1, sentiment=0.0).is_scored is True


def test_to_dict_serialises_utc():
    payload = _item(1, sentiment=0.5, sentiment_label="bullish").to_dict()
    assert payload["published_at"].endswith("+00:00")
    assert payload["sentiment"] == 0.5


# ---------------------------------------------------------------------------
# Point-in-time — der Look-ahead-Guard
# ---------------------------------------------------------------------------


def test_future_headlines_are_dropped():
    items = [_item(1), _item(-1), _item(0.5)]  # eine liegt NACH as_of
    kept, dropped = filter_point_in_time(items, AS_OF)
    assert dropped == 1
    assert all(i.published_at < AS_OF for i in kept)


def test_headline_exactly_at_cutoff_is_dropped():
    """Strikt `<`: zum Analysezeitpunkt selbst ist sie nicht verlässlich da."""
    kept, dropped = filter_point_in_time([_item(0)], AS_OF)
    assert kept == [] and dropped == 1


def test_kept_items_are_newest_first():
    kept, _ = filter_point_in_time([_item(5), _item(1), _item(3)], AS_OF)
    assert [i.published_at for i in kept] == sorted(
        [i.published_at for i in kept], reverse=True
    )


def test_naive_cutoff_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        filter_point_in_time([_item(1)], datetime(2026, 7, 20, 12, 0))


# ---------------------------------------------------------------------------
# GDELT
# ---------------------------------------------------------------------------


GDELT_PAYLOAD = {
    "articles": [
        {
            "title": "SPY hits record high",
            "url": "https://news.example/1",
            "domain": "news.example",
            "seendate": "20260719T140000Z",
        },
        {
            "title": "Bond yields slip",
            "url": "https://other.example/2",
            "domain": "other.example",
            "seendate": "20260718T090000Z",
        },
        {"title": "", "seendate": "20260718T090000Z"},  # ohne Titel → raus
        {"title": "Kaputtes Datum", "seendate": "nonsense"},  # → raus
    ]
}


def test_gdelt_parsing():
    items = parse_gdelt(GDELT_PAYLOAD, symbols=["spy", "tlt"])
    assert len(items) == 2
    assert items[0].headline == "SPY hits record high"
    assert items[0].source == "news.example"
    assert items[0].provider == "gdelt"
    # GDELT taggt keine Ticker — die Query-Symbole werden übernommen.
    assert items[0].symbols == ("SPY", "TLT")
    # ... und liefert kein Sentiment.
    assert all(i.sentiment is None for i in items)


@pytest.mark.parametrize("payload", [{}, {"articles": "nope"}, None, "text"])
def test_gdelt_garbage_yields_empty(payload):
    assert parse_gdelt(payload) == []


# ---------------------------------------------------------------------------
# Alpha Vantage
# ---------------------------------------------------------------------------


AV_PAYLOAD = {
    "feed": [
        {
            "title": "Fed signals pause",
            "url": "https://av.example/1",
            "source": "Reuters",
            "summary": "Zusammenfassung",
            "time_published": "20260719T140000",
            "overall_sentiment_score": 0.1,
            "ticker_sentiment": [
                {"ticker": "SPY", "ticker_sentiment_score": "0.42"},
                {"ticker": "AAPL", "ticker_sentiment_score": "-0.9"},
            ],
        },
        {
            "title": "Ohne Ticker-Sentiment",
            "url": "https://av.example/2",
            "source": "WSJ",
            "time_published": "20260718T100000",
            "overall_sentiment_score": -0.5,
            "ticker_sentiment": [],
        },
    ]
}


def test_alpha_vantage_prefers_ticker_level_sentiment():
    """Die Frage ist „was heißt das für SPY", nicht „wie ist der Artikel gestimmt"."""
    items = parse_alpha_vantage(AV_PAYLOAD, symbols=["SPY"])
    first = items[0]
    assert first.sentiment == pytest.approx(0.42)  # nicht 0.1 (overall)
    assert first.sentiment_label == "bullish"
    assert first.symbols == ("SPY",)
    assert first.provider == "alpha_vantage"


def test_alpha_vantage_falls_back_to_overall_score():
    items = parse_alpha_vantage(AV_PAYLOAD, symbols=["SPY"])
    assert items[1].sentiment == pytest.approx(-0.5)
    assert items[1].sentiment_label == "bearish"


def test_alpha_vantage_labels_match_documented_thresholds():
    def label(score: float) -> str:
        payload = {
            "feed": [
                {
                    "title": "x",
                    "time_published": "20260719T140000",
                    "overall_sentiment_score": score,
                }
            ]
        }
        return parse_alpha_vantage(payload)[0].sentiment_label

    assert label(-0.5) == "bearish"
    assert label(-0.2) == "somewhat_bearish"
    assert label(0.0) == "neutral"
    assert label(0.2) == "somewhat_bullish"
    assert label(0.5) == "bullish"


@pytest.mark.parametrize("payload", [{}, {"feed": None}, None, []])
def test_alpha_vantage_garbage_yields_empty(payload):
    assert parse_alpha_vantage(payload) == []


# ---------------------------------------------------------------------------
# fetch_news — Verdrahtung
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, payload, recorder: list | None = None):
        self._payload = payload
        self.recorder = recorder if recorder is not None else []

    def get(self, url, params=None):
        self.recorder.append((url, dict(params or {})))
        return _FakeResponse(self._payload)

    def close(self):
        pass


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_provider_off_is_the_default(monkeypatch):
    monkeypatch.delenv("NEWS_PROVIDER", raising=False)
    assert configured_provider() == "off"
    result = fetch_news(["SPY"], as_of=AS_OF)
    assert result.items == [] and result.provider == "off"
    assert any("aus" in w for w in result.warnings)


def test_unknown_provider_env_falls_back_to_off(monkeypatch):
    monkeypatch.setenv("NEWS_PROVIDER", "bloomberg_terminal")
    assert configured_provider() == "off"


@pytest.mark.parametrize("name", NEWS_PROVIDERS)
def test_configured_provider_accepts_all_supported(name, monkeypatch):
    monkeypatch.setenv("NEWS_PROVIDER", name)
    assert configured_provider() == name


def test_fetch_gdelt_end_to_end_applies_point_in_time():
    payload = {
        "articles": [
            *GDELT_PAYLOAD["articles"],
            {
                "title": "Aus der Zukunft",
                "domain": "future.example",
                "seendate": "20260721T090000Z",  # nach AS_OF
            },
        ]
    }
    client = _FakeClient(payload)
    result = fetch_news(["SPY"], provider="gdelt", as_of=AS_OF, client=client)

    assert [i.headline for i in result.items] == ["SPY hits record high", "Bond yields slip"]
    assert result.dropped_future == 1
    assert any("Point-in-time" in w for w in result.warnings)


def test_fetch_rejects_unknown_provider():
    with pytest.raises(ValueError, match="provider muss"):
        fetch_news(["SPY"], provider="reuters", as_of=AS_OF)


def test_fetch_rejects_empty_symbols():
    with pytest.raises(ValueError, match="symbols ist leer"):
        fetch_news([], provider="gdelt", as_of=AS_OF)


def test_fetch_rejects_naive_as_of():
    with pytest.raises(ValueError, match="timezone-aware"):
        fetch_news(["SPY"], provider="gdelt", as_of=datetime(2026, 7, 20))


def test_alpha_vantage_without_key_is_actionable(monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPHA_VANTAGE_API_KEY"):
        fetch_news(["SPY"], provider="alpha_vantage", as_of=AS_OF, client=_FakeClient({}))


def test_alpha_vantage_rate_limit_note_is_not_mistaken_for_empty(monkeypatch):
    """AV antwortet auf Rate-Limit mit HTTP 200 — das darf nicht wie 'keine News' aussehen."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "secret-key")
    client = _FakeClient({"Note": "Thank you for using Alpha Vantage! Our standard API rate limit"})
    with pytest.raises(RuntimeError, match="Alpha Vantage"):
        fetch_news(["SPY"], provider="alpha_vantage", as_of=AS_OF, client=client)


def test_alpha_vantage_key_is_never_echoed_in_errors(monkeypatch):
    """Der Key steht (providerbedingt) im Query — er darf nirgends wieder rausfallen."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "super-secret")
    client = _FakeClient({"Error Message": "invalid apikey super-secret"})
    with pytest.raises(RuntimeError) as exc:
        fetch_news(["SPY"], provider="alpha_vantage", as_of=AS_OF, client=client)
    assert "super-secret" not in str(exc.value)
    assert "***" in str(exc.value)


def test_limit_is_respected():
    payload = {
        "articles": [
            {
                "title": f"Headline {n}",
                "domain": "x.example",
                "seendate": "20260719T140000Z",
            }
            for n in range(10)
        ]
    }
    result = fetch_news(["SPY"], provider="gdelt", as_of=AS_OF, limit=3, client=_FakeClient(payload))
    assert len(result.items) == 3
