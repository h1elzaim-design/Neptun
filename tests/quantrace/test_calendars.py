"""Annualisierung kommt aus den Daten (#184).

`periods_per_year` steckte als Default `252.0` in acht Statistik-Modulen. Für
US-Aktien stimmt das; für 24/7-Märkte nicht. Der Fehler wäre unsichtbar und
teuer: Sharpe skaliert mit √P, ein Crypto-Sharpe mit 252 statt 365 gerechnet
sähe um √(365/252) ≈ 1.204 **zu gut** aus — und DSR, PBO, Bootstrap und der
Governance-Score würden diesen Wert präzise weiterverarbeiten.

Der wichtigste Test hier ist `test_existing_universes_are_unchanged`: dieser PR
fasst die Statistik-Schicht an, und für alles Bestehende muss er folgenlos
bleiben.
"""

from __future__ import annotations

import math
import pathlib
from datetime import date

import pandas as pd
import pytest
import yaml

from quantrace.calendars import (
    CALENDARS,
    DEFAULT_CALENDAR,
    UnknownCalendarError,
    get_calendar,
    periods_per_year,
)
from quantrace.models import BacktestConfig, MarketData

UNIVERSE_DIR = "data/universes"


def _md(calendar: str | None = None) -> MarketData:
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    cols = pd.MultiIndex.from_product([["SPY"], ["open", "high", "low", "close", "volume"]])
    frame = pd.DataFrame(1.0, index=idx, columns=cols)
    kwargs = {"calendar": calendar} if calendar is not None else {}
    return MarketData(
        universe="u",
        symbols=["SPY"],
        timeframe="1d",
        start=date(2020, 1, 2),
        end=date(2020, 1, 3),
        provider="tiingo",
        frame=frame,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Der Kalender selbst


def test_default_is_us_equity_with_252():
    assert DEFAULT_CALENDAR == "us_equity"
    assert periods_per_year(None) == 252.0


def test_crypto_trades_every_calendar_day():
    assert periods_per_year("crypto_24_7") == 365.0


def test_unknown_calendar_raises_instead_of_falling_back():
    """Ein Tippfehler in `calendar:` darf nicht still den Default nehmen —
    dann wäre die Annualisierung falsch, ohne dass es jemand merkt."""
    with pytest.raises(UnknownCalendarError, match="Unbekannter Kalender"):
        get_calendar("us_equtiy")


def test_every_calendar_has_a_description():
    """Der Katalog wird gelesen, nicht nur ausgewertet."""
    for cal in CALENDARS.values():
        assert cal.description.strip()
        assert cal.periods_per_year > 0


# ---------------------------------------------------------------------------
# Das Datenmodell


def test_market_data_derives_periods_per_year():
    assert _md().periods_per_year == 252.0
    assert _md("crypto_24_7").periods_per_year == 365.0


def test_periods_per_year_cannot_diverge_from_the_calendar():
    """Als eigenes Feld könnte es vom Kalender abweichen — genau die stille
    Divergenz, die #184 beseitigt. Es ist deshalb eine Property."""
    md = _md("crypto_24_7")
    assert "periods_per_year" not in type(md).model_fields
    with pytest.raises(AttributeError):
        md.periods_per_year = 999.0  # type: ignore[misc]


def test_market_data_rejects_an_unknown_calendar():
    with pytest.raises(Exception, match="Unbekannter Kalender"):
        _md("mondkalender")


# ---------------------------------------------------------------------------
# Die Universen im Repo


def test_every_universe_declares_its_calendar():
    """Ehrlichkeit über Defaults: ein Universum ohne `calendar:` würde den
    Default erben, und die Annahme bliebe unsichtbar."""
    import glob

    files = sorted(glob.glob(f"{UNIVERSE_DIR}/*.yaml"))
    assert files, "keine Universen gefunden"
    for path in files:
        cfg = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
        assert "calendar" in cfg, f"{path} nennt keinen Kalender"
        get_calendar(cfg["calendar"])  # wirft bei Unfug


#: Die neun Universen, die es vor #184 gab — alle US-gelistet.
#:
#: Namentlich statt per Glob: seit Schritt C liegt `crypto_majors.yaml` daneben,
#: und ein Glob würde entweder den Regressionstest kaputtmachen oder (schlimmer)
#: still mitwandern, sobald jemand ein weiteres Nicht-Aktien-Universum anlegt.
PRE_184_UNIVERSES = (
    "europe_etfs",
    "factor_etfs",
    "global_macro",
    "international_etfs",
    "sector_spdrs",
    "us_core_etfs",
    "us_core_etfs_extended",
    "us_megacap_equities",
    "us_midcap_etfs",
)


def test_existing_universes_are_unchanged():
    """**Der Regressionstest dieses PRs.** Alle neun bestehenden Universen sind
    US-gelistet — die Umstellung darf an keiner einzigen Zahl etwas ändern."""
    for name in PRE_184_UNIVERSES:
        path = pathlib.Path(UNIVERSE_DIR) / f"{name}.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert periods_per_year(cfg["calendar"]) == 252.0, path


# ---------------------------------------------------------------------------
# Der Backtest-Runner


def test_runner_takes_the_annualization_from_the_data():
    from quantrace.backtest_runner import _resolve_annualization

    assert _resolve_annualization(BacktestConfig(), _md()) == 252.0
    assert _resolve_annualization(BacktestConfig(), _md("crypto_24_7")) == 365.0


def test_an_explicit_config_value_still_wins():
    """Wer `annualization` setzt, meint sie auch — Tests und Sonderfälle
    müssen die Ableitung übersteuern können."""
    from quantrace.backtest_runner import _resolve_annualization

    assert _resolve_annualization(BacktestConfig(annualization=12), _md("crypto_24_7")) == 12.0


def test_the_default_value_does_not_count_as_explicit():
    """Der Feld-Default 252 darf die Ableitung nicht blockieren — sonst hätte
    die ganze Umstellung keine Wirkung."""
    from quantrace.backtest_runner import _resolve_annualization

    assert BacktestConfig().annualization == 252  # Default steht unverändert
    assert _resolve_annualization(BacktestConfig(), _md("crypto_24_7")) == 365.0


# ---------------------------------------------------------------------------
# Warum das überhaupt zählt


def test_the_wrong_calendar_inflates_sharpe_by_a_fifth():
    """Die Zahl, um die es geht. Derselbe Return-Pfad, zwei Kalender."""
    from quantrace.stats.sharpe import annualised_sharpe

    returns = [0.001, -0.0005, 0.002, 0.0008, -0.001, 0.0015] * 40
    as_equity = annualised_sharpe(returns, periods_per_year=252.0)
    as_crypto = annualised_sharpe(returns, periods_per_year=365.0)

    assert as_crypto > as_equity
    assert as_crypto / as_equity == pytest.approx(math.sqrt(365 / 252), rel=1e-9)
    assert as_crypto / as_equity == pytest.approx(1.204, abs=1e-3)


# ---------------------------------------------------------------------------
# Der Weg in die Webapp


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"periods_per_year": 365.0}, 365.0),
        ({"periods_per_year": 252}, 252.0),
        ({}, 252.0),  # Alt-JSON von vor #184 — stammt aus us_equity
        (None, 252.0),
        ({"periods_per_year": 0}, 252.0),  # unsinnig → Fallback statt Division durch 0
        ({"periods_per_year": "kaputt"}, 252.0),
    ],
)
def test_analytics_reads_the_annualization_from_the_json(payload, expected):
    from api.services.analytics import periods_per_year_of

    assert periods_per_year_of(payload) == expected


# ---------------------------------------------------------------------------
# Guardrail: ein Universum, ein Kalender (#184, Schritt B)


def test_a_pure_universe_passes():
    from quantrace.calendars import validate_universe_calendar

    validate_universe_calendar(["SPY", "QQQ", "TLT"], "us_equity")
    validate_universe_calendar(["BTCUSD", "ETHUSD"], "crypto_24_7")


def test_mixing_calendars_is_refused():
    """252 wäre für BTC falsch, 365 für SPY. Statt einen der beiden Werte zu
    wählen und die Hälfte der Zahlen zu verfälschen, wird abgelehnt."""
    from quantrace.calendars import CalendarMismatchError, validate_universe_calendar

    with pytest.raises(CalendarMismatchError) as ei:
        validate_universe_calendar(["SPY", "BTCUSD"], "us_equity", universe="gemischt")
    assert "BTCUSD" in str(ei.value)
    assert "SPY" not in str(ei.value), "nur die Abweichler werden genannt"


def test_crypto_in_an_equity_universe_is_caught():
    from quantrace.calendars import CalendarMismatchError, validate_universe_calendar

    with pytest.raises(CalendarMismatchError):
        validate_universe_calendar(["BTCUSD"], "us_equity")


def test_an_unclassified_symbol_is_caught_in_a_crypto_universe():
    """Ein Symbol ohne Klasse fällt auf equity zurück — in einem
    Crypto-Universum fliegt es dadurch auf und muss klassifiziert werden."""
    from quantrace.calendars import CalendarMismatchError, validate_universe_calendar

    with pytest.raises(CalendarMismatchError, match="GIBTSNICHT"):
        validate_universe_calendar(["BTCUSD", "GIBTSNICHT"], "crypto_24_7")


def test_the_error_says_how_to_fix_it():
    """Ein Guardrail, der nur „nein" sagt, kostet den Nutzer eine Suche."""
    from quantrace.calendars import CalendarMismatchError, validate_universe_calendar

    with pytest.raises(CalendarMismatchError) as ei:
        validate_universe_calendar(["SPY", "BTCUSD"], "us_equity")
    msg = str(ei.value)
    assert "aufteilen" in msg
    assert "costs.yaml" in msg


def test_empty_universe_is_not_an_error():
    from quantrace.calendars import validate_universe_calendar

    validate_universe_calendar([], "us_equity")


def test_every_universe_in_the_repo_passes_its_own_guardrail():
    """Was der Guardrail von Nutzereingaben verlangt, müssen die mitgelieferten
    Dateien selbst erfüllen — sonst steht der Fehler im Repo."""
    import glob

    from quantrace.calendars import validate_universe_calendar

    for path in sorted(glob.glob(f"{UNIVERSE_DIR}/*.yaml")):
        cfg = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
        validate_universe_calendar(
            cfg["symbols"], cfg["calendar"], universe=str(cfg.get("name") or path)
        )


# ---------------------------------------------------------------------------
# Crypto-Kostenklassen (#184, Schritt B)


def test_crypto_costs_are_an_order_of_magnitude_above_etfs():
    """Der eigentliche Zweck der Klassen: eine Crypto-Strategie mit
    ETF-Gebühren gerechnet sieht profitabel aus, wo sie es nicht ist."""
    from quantrace.costs import resolve_symbol_costs

    costs = resolve_symbol_costs(["SPY", "BTCUSD", "SOLUSD"])

    def per_side(c):
        return c.fees_bps + c.slippage_bps + c.spread_bps / 2

    etf, major, alt = (per_side(costs[s]) for s in ("SPY", "BTCUSD", "SOLUSD"))
    assert major > 10 * etf
    assert alt > major


def test_crypto_tickers_are_classified_not_defaulted():
    """Ohne Zuordnung fielen sie auf default_class (0.5 bps) zurück — mit einer
    Log-Warnung, die niemand liest."""
    from quantrace.costs import resolve_symbol_costs

    resolved = resolve_symbol_costs(["BTCUSD", "ETHUSD", "SOLUSD"])
    assert resolved["BTCUSD"].asset_class == "crypto_major"
    assert resolved["ETHUSD"].asset_class == "crypto_major"
    assert resolved["SOLUSD"].asset_class == "crypto_alt"
