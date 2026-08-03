"""Vollständigkeit ist kalenderabhängig (#184, Nachtrag).

Der erste echte Crypto-Fetch meldete für AVAXUSD „1 Lücken > 5d" — bei 139
fehlenden Tagen von 1461. Zwei Gründe, und beide sind hier abgedeckt:

1. Der Schwellwert war eine Konstante (5), gedacht für Börsen mit Wochenende.
   Für einen 24/7-Markt ist jeder übersprungene Tag ein Loch.
2. Der Löwenanteil der fehlenden Tage lag **vor** der ersten Zeile. Zwischen
   den vorhandenen Zeilen war alles lückenlos — eine Serie, die vier Monate zu
   spät anfängt, war vom Gate aus von einer vollständigen nicht zu
   unterscheiden.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from quantrace import quality
from quantrace.calendars import get_calendar


def _frame(dates: list[str]) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    n = len(idx)
    return pd.DataFrame(
        {"open": [1.0] * n, "high": [1.0] * n, "low": [1.0] * n,
         "close": [1.0] * n, "volume": [1.0] * n},
        index=idx,
    )


def _kinds(report) -> set[str]:
    return {i.kind for i in report.issues}


def _detail(report, kind) -> str:
    return next(i.detail for i in report.issues if i.kind == kind)


# ---------------------------------------------------------------------------
# Der Schwellwert kommt aus dem Kalender


def test_the_calendars_declare_their_own_tolerance():
    assert get_calendar("crypto_24_7").max_gap_days == 1
    assert get_calendar("us_equity").max_gap_days == 5


def test_the_same_hole_is_read_differently_per_market():
    """Fr 05.01. → Mo 08.01.2024. Für die NYSE der Normalfall, für einen
    24/7-Markt zwei fehlende Tage."""
    frame = _frame(["2024-01-05", "2024-01-08"])

    assert "missing_days" not in _kinds(quality.check_symbol("SPY", frame, calendar="us_equity"))

    detail = _detail(quality.check_symbol("BTCUSD", frame, calendar="crypto_24_7"), "missing_days")
    assert "2024-01-06" in detail and "2024-01-07" in detail


def test_consecutive_days_are_clean_for_crypto():
    report = quality.check_symbol(
        "BTCUSD", _frame(["2024-01-04", "2024-01-05", "2024-01-06"]), calendar="crypto_24_7"
    )
    assert report.issues == []


# ---------------------------------------------------------------------------
# Vollständigkeit — exakt nur bei 24/7


def test_crypto_counts_every_missing_day():
    """Die Zahl, die AVAXUSD gebraucht hätte: nicht „1 Lücke", sondern wie
    viele Tage tatsächlich fehlen."""
    report = quality.check_symbol(
        "BTCUSD",
        _frame(["2024-01-01", "2024-01-02", "2024-01-06", "2024-01-07"]),
        calendar="crypto_24_7",
    )
    detail = _detail(report, "missing_days")
    assert "3 von 7 Handelstagen fehlen" in detail
    assert "2024-01-03" in detail  # die fehlenden Tage stehen namentlich da


def test_equities_are_now_checked_just_as_exactly():
    """Mit dem Börsenkalender fliegt ein einzelner fehlender Handelstag bei
    Aktien genauso auf wie bei Crypto. Vorher schwieg das Gate hier.

    2024-01-08 ist ein Montag; der Dienstag danach fehlt im Frame."""
    report = quality.check_symbol(
        "SPY", _frame(["2024-01-08", "2024-01-10"]), calendar="us_equity"
    )
    detail = _detail(report, "missing_days")
    assert "2024-01-09" in detail
    assert "1 von 3 Handelstagen fehlen" in detail


def test_a_weekend_is_not_missing_data():
    """Der Gegentest — sonst hätte man ein Gate, das jedes Wochenende meldet."""
    report = quality.check_symbol(
        "SPY", _frame(["2024-01-05", "2024-01-08"]), calendar="us_equity"
    )
    assert "missing_days" not in _kinds(report)


def test_a_holiday_is_not_missing_data():
    """Karfreitag 2024 (2024-03-29) ist NYSE-Feiertag. Ein Gate ohne
    Feiertagswissen würde ihn als Loch melden."""
    report = quality.check_symbol(
        "SPY", _frame(["2024-03-28", "2024-04-01"]), calendar="us_equity"
    )
    assert "missing_days" not in _kinds(report)


def test_the_calendar_knows_the_unscheduled_closures():
    """**Warum exakt und Toleranz sich ausschließen müssen.**

    Zwischen dem 10. und dem 17. September 2001 liegen sieben Tage, weil die
    NYSE nach den Anschlägen vier Tage geschlossen war. Der Handelskalender
    weiß das. Liefe die 5-Tage-Schwelle daneben weiter, wäre die Folge ein
    Fehlalarm für **jedes** US-Symbol im Jahr 2001 — und dasselbe für Hurrikan
    Sandy 2012 und jede Staatstrauer.
    """
    report = quality.check_symbol(
        "SPY", _frame(["2001-09-10", "2001-09-17"]), calendar="us_equity"
    )
    assert report.issues == []


def test_long_holes_are_summarised_not_dumped():
    """Vier Monate Loch dürfen den Log nicht unlesbar machen."""
    frame = _frame(["2024-01-01", "2024-06-01"])
    detail = _detail(quality.check_symbol("BTCUSD", frame, calendar="crypto_24_7"), "missing_days")
    assert "weitere)" in detail
    assert len(detail) < 300


# ---------------------------------------------------------------------------
# Der Randabschnitt — das eigentlich verborgene Problem


def test_a_series_that_starts_late_is_caught():
    """**Der Kern dieses Nachtrags.** Zwischen den Zeilen ist alles lückenlos;
    trotzdem fehlt die halbe Historie. Ohne das angeforderte Fenster sieht das
    Gate hier nichts."""
    report = quality.check_symbol(
        "AVAXUSD",
        _frame(["2021-05-20", "2021-05-21"]),
        calendar="crypto_24_7",
        expected_start=date(2021, 1, 1),
        expected_end=date(2021, 5, 21),
    )
    assert "coverage_truncated" in _kinds(report)
    detail = _detail(report, "coverage_truncated")
    assert "2021-05-20" in detail
    assert "139 Tage fehlen am Anfang" in detail


def test_a_series_that_ends_early_is_caught():
    """Der Delisting-Fall — oder ein Fetch, der stehen geblieben ist."""
    report = quality.check_symbol(
        "XYZUSD",
        _frame(["2024-01-01", "2024-01-02"]),
        calendar="crypto_24_7",
        expected_start=date(2024, 1, 1),
        expected_end=date(2024, 3, 1),
    )
    assert "endet schon 2024-01-02" in _detail(report, "coverage_truncated")


def test_a_full_year_of_equity_data_is_silent():
    """**Der Fehlalarm, den meine erste Fassung gebaut hätte.**

    Jeder Backtest fragt ein Kalenderfenster an — typisch `2023-01-01` bis
    `2023-12-31`. Die Reihe beginnt dann zwangsläufig am 3. Januar und endet
    am 29. Dezember, weil der 1. Feiertag und der 30./31. Wochenende sind.
    Nach Kalendertagen gerechnet sieht das nach fehlenden Daten aus. Es fehlt
    nichts — und ein Gate, das bei jedem einzelnen Aktien-Backtest warnt,
    bringt niemandem etwas bei, außer Warnungen zu ignorieren.
    """
    from quantrace.calendars import trading_sessions

    sessions = trading_sessions("us_equity", date(2023, 1, 1), date(2023, 12, 31))
    frame = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=sessions
    )
    report = quality.check_symbol(
        "SPY", frame, calendar="us_equity",
        expected_start=date(2023, 1, 1), expected_end=date(2023, 12, 31),
    )
    assert report.issues == []


def test_real_truncation_is_still_reported_in_trading_days():
    """Die Gegenprobe — echte fehlende Historie darf nicht mit weggekürzt
    werden, und die Einheit sind Handelstage, nicht Kalendertage."""
    report = quality.check_symbol(
        "IPO", _frame(["2023-07-03", "2023-07-05"]), calendar="us_equity",
        expected_start=date(2023, 1, 1),
    )
    detail = _detail(report, "coverage_truncated")
    assert "Handelstage fehlen am Anfang" in detail
    assert "124" in detail  # Handelstage, nicht die 183 Kalendertage


def test_a_complete_series_says_nothing():
    report = quality.check_symbol(
        "BTCUSD",
        _frame(["2024-01-01", "2024-01-02", "2024-01-03"]),
        calendar="crypto_24_7",
        expected_start=date(2024, 1, 1),
        expected_end=date(2024, 1, 3),
    )
    assert report.issues == []


def test_truncation_is_checked_for_equities_too():
    """Der Randabschnitt braucht keinen Feiertagskalender — „beginnt später als
    angefordert" ist eine Tatsache, egal welcher Markt."""
    report = quality.check_symbol(
        "SPY",
        _frame(["2024-03-01", "2024-03-04"]),
        calendar="us_equity",
        expected_start=date(2024, 1, 1),
    )
    assert "coverage_truncated" in _kinds(report)


# ---------------------------------------------------------------------------
# Nichts davon darf den Bestand verändern


def test_no_calendar_means_us_equity():
    """Der Default hat sich nicht verschoben — ein Aufruf ohne Kalender ist ein
    us_equity-Aufruf, wie vorher."""
    frame = _frame(["2024-01-02", "2024-01-12"])
    ohne = quality.check_symbol("SPY", frame)
    mit = quality.check_symbol("SPY", frame, calendar="us_equity")
    assert [(i.kind, i.detail) for i in ohne.issues] == [(i.kind, i.detail) for i in mit.issues]


def test_without_the_package_the_tolerance_takes_over(monkeypatch):
    """Der Rückfall. Fehlt `exchange_calendars`, prüft das Gate gröber — aber
    es behauptet dann auch keine Vollständigkeit, und es sagt das im Text."""
    from quantrace import calendars

    monkeypatch.setattr(calendars, "trading_sessions", lambda *a, **k: None)
    monkeypatch.setattr(quality, "trading_sessions", lambda *a, **k: None)

    report = quality.check_symbol(
        "SPY", _frame(["2024-01-02", "2024-01-12"]), calendar="us_equity"
    )
    assert "missing_days" not in _kinds(report)
    assert "ohne Handelskalender" in _detail(report, "calendar_gap")


def test_the_fallback_does_not_cry_wolf_at_new_year(monkeypatch):
    """Ein Fetch ab dem 1. Januar beginnt zwangsläufig am 2. oder 3. — Neujahr
    ist Feiertag. Ohne Handelskalender ist das nicht *beweisbar*, aber ein
    einzelner Kalendertag am Rand ist auch kein Befund: der erste echte Lauf
    gegen us_core_etfs produzierte so **sechzehn** Warnungen, eine pro Symbol,
    die alle nichts bedeuteten."""
    monkeypatch.setattr(quality, "trading_sessions", lambda *a, **k: None)

    report = quality.check_symbol(
        "SPY",
        _frame(["2018-01-02", "2018-01-03"]),
        calendar="us_equity",
        expected_start=date(2018, 1, 1),
    )
    assert "coverage_truncated" not in _kinds(report)


def test_the_fallback_still_reports_real_truncation(monkeypatch):
    """Die Gegenprobe — vier Monate fehlende Historie müssen auch ohne
    Handelskalender auffallen, nur eben als Schätzung gekennzeichnet."""
    monkeypatch.setattr(quality, "trading_sessions", lambda *a, **k: None)

    report = quality.check_symbol(
        "IPO",
        _frame(["2023-05-01", "2023-05-02"]),
        calendar="us_equity",
        expected_start=date(2023, 1, 1),
    )
    assert "geschätzt" in _detail(report, "coverage_truncated")


def test_crypto_edges_stay_exact_without_any_exchange_calendar():
    """Bei 24/7 sind Kalendertage die richtige Einheit — dort darf die
    Toleranz **nicht** greifen, sonst verschwindet ein echter Randabschnitt.
    AVAXUSD fehlten 131 Tage am Anfang; die müssen gemeldet werden."""
    report = quality.check_symbol(
        "AVAXUSD",
        _frame(["2021-05-12", "2021-05-13"]),
        calendar="crypto_24_7",
        expected_start=date(2021, 1, 1),
    )
    assert "131 Tage fehlen am Anfang" in _detail(report, "coverage_truncated")


def test_an_explicit_limit_still_reaches_the_fallback(monkeypatch):
    monkeypatch.setattr(quality, "trading_sessions", lambda *a, **k: None)
    report = quality.check_symbol(
        "SPY", _frame(["2024-01-02", "2024-01-12"]), calendar="us_equity", max_gap_days=30
    )
    assert report.issues == []


def test_severity_stays_a_warning():
    """Eine zu kurze Historie ist kein Rechenfehler — AVAX gab es 2021 noch
    nicht überall. Der Lauf muss weiterlaufen und die Entscheidung mir
    überlassen; blockieren würde `crypto_majors` unbenutzbar machen."""
    report = quality.check_symbol(
        "AVAXUSD",
        _frame(["2021-05-20", "2021-05-25"]),
        calendar="crypto_24_7",
        expected_start=date(2021, 1, 1),
    )
    assert report.ok
    assert all(i.severity == "warning" for i in report.issues)
