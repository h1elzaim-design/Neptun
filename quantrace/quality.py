"""Data-Quality-Gate — fängt stille Datenfehler, bevor sie Backtests vergiften.

Ein Backtest auf kaputten Daten ist schlimmer als kein Backtest: er produziert
plausibel aussehende, aber falsche Kennzahlen. Dieses Modul prüft eine OHLCV-
Serie auf die üblichen Verdächtigen und gibt strukturierte Issues zurück.

**Vollständigkeit ist kalenderabhängig** (#184). Für einen 24/7-Markt muss
*jeder* Kalendertag da sein; für eine Börse mit Wochenende und Feiertagen ist
eine Lücke von drei Tagen der Normalfall. Der Schwellwert kommt deshalb aus
`quantrace.calendars`, nicht aus einer Konstante hier.

Was dieses Modul **exakt** kann und was nicht:

- ``all_days_trade`` (Crypto): die erwarteten Tage sind schlicht alle Tage
  zwischen erster und letzter Zeile.
- Börsen: die erwarteten Tage kommen aus dem Handelskalender
  (``calendars.trading_sessions`` über ``exchange_calendars``, XNYS für
  us_equity). Ein einzelner fehlender Handelstag fliegt damit genauso auf wie
  ein fehlender Crypto-Tag.

Ist der Börsenkalender nicht verfügbar — Paket fehlt, oder das Fenster liegt
außerhalb der erzeugten Grenzen — fällt die Prüfung auf die Lücken-Toleranz
zurück und **behauptet keine Vollständigkeit**. Der Unterschied ist geloggt,
nicht still.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from quantrace.calendars import get_calendar, trading_sessions

log = logging.getLogger(__name__)

#: Wie viele fehlende Tage eine Meldung namentlich aufzählt, bevor sie kürzt.
#: Ein Symbol mit vier Monaten Loch soll den Log nicht unlesbar machen.
_MAX_LISTED_DATES = 5


@dataclass
class QualityIssue:
    symbol: str
    kind: str
    detail: str
    severity: str = "warning"  # "warning" | "error"


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def add(self, symbol: str, kind: str, detail: str, severity: str = "warning") -> None:
        self.issues.append(QualityIssue(symbol, kind, detail, severity))

    def log(self) -> None:
        for i in self.issues:
            fn = log.error if i.severity == "error" else log.warning
            fn("Data-Quality [%s] %s: %s", i.symbol, i.kind, i.detail)


def _list_dates(index: pd.DatetimeIndex) -> str:
    """Ein paar Datumswerte zum Anschauen, der Rest als Zahl."""
    shown = [str(d.date()) for d in index[:_MAX_LISTED_DATES]]
    rest = len(index) - len(shown)
    return ", ".join(shown) + (f" … (+{rest} weitere)" if rest > 0 else "")


def _missing_at_edge(cal, von: date, bis: date) -> tuple[int, str]:
    """Wie viel am Rand fehlt — in Handelstagen, wenn der Kalender es weiß.

    „59 Tage fehlen" ist für eine Börse die falsche Einheit: davon sind rund
    17 Wochenende. Mit dem Handelskalender steht dort die Zahl, die zählt.

    Der Rückgabewert kann **0** sein, und das ist der wichtige Fall: fragt
    jemand einen Backtest über ``2023-01-01..2023-12-31`` an, beginnt die
    Reihe zwangsläufig am 3. Januar — der 1. war Feiertag, der 2. … ebenfalls.
    Nach Kalendertagen gerechnet sähe das nach fehlenden Daten aus. Es fehlt
    nichts, und deshalb darf hier auch nichts gemeldet werden.

    Das um eins verkürzte Fenster ist Absicht: der erste vorhandene Tag ist da.

    Drei Wege, je nachdem was bekannt ist:

    * ``all_days_trade``: Kalendertage **sind** Handelstage. Exakt.
    * Börse mit Handelskalender: die Sessions im Randfenster. Exakt.
    * Börse ohne Handelskalender: unbekannt. Dann greift dieselbe Toleranz
      wie bei den Lücken — sonst meldet ein Fetch ab dem 1. Januar für
      **jedes** Symbol „1 Tag fehlt", weil Neujahr ein Feiertag ist. Sechzehn
      Warnungen, die alle nichts bedeuten, sind schlimmer als keine.
    """
    if cal.all_days_trade:
        return (bis - von).days, "Tage"

    sessions = trading_sessions(cal.name, von, bis)
    if sessions is not None:
        return max(len(sessions) - 1, 0), "Handelstage"

    tage = (bis - von).days
    return (tage if tage > cal.max_gap_days else 0), "Tage (geschätzt, ohne Handelskalender)"


def check_symbol(
    symbol: str,
    frame: pd.DataFrame,
    *,
    calendar: str | None = None,
    max_gap_days: int | None = None,
    expected_start: date | None = None,
    expected_end: date | None = None,
    report: QualityReport | None = None,
) -> QualityReport:
    """Prüft eine Einzel-Symbol-OHLCV-Serie (DatetimeIndex, OHLCV-Spalten).

    Parameters
    ----------
    calendar:
        Handelskalender des Universums. Bestimmt, ab wann eine Lücke
        verdächtig ist und ob Vollständigkeit exakt prüfbar ist.
    max_gap_days:
        Übersteuert den Kalender-Wert. Für gezielte Aufrufe und Tests.
    expected_start, expected_end:
        Das **angeforderte** Fenster. Ohne diese Angabe bleibt eine Serie, die
        schlicht später anfängt als gewünscht, unsichtbar — siehe
        ``coverage_truncated``.
    """
    report = report or QualityReport()
    cal = get_calendar(calendar)
    gap_limit = cal.max_gap_days if max_gap_days is None else max_gap_days

    if frame.empty:
        report.add(symbol, "empty", "keine Datenpunkte", "error")
        return report

    idx = frame.index
    if not idx.is_monotonic_increasing:
        report.add(symbol, "unsorted", "Index nicht monoton steigend", "error")
    if idx.has_duplicates:
        n = int(idx.duplicated().sum())
        report.add(symbol, "duplicate_dates", f"{n} doppelte Zeitstempel", "error")

    for col in ("open", "high", "low", "close"):
        if col not in frame.columns:
            report.add(symbol, "missing_column", f"Spalte {col} fehlt", "error")
            continue
        s = frame[col].astype(float)
        n_nan = int(s.isna().sum())
        if n_nan:
            report.add(symbol, "nan", f"{col}: {n_nan} NaN-Werte")
        n_nonpos = int((s <= 0).sum())
        if n_nonpos:
            report.add(symbol, "nonpositive_price", f"{col}: {n_nonpos} Werte <= 0")

    if {"high", "low"} <= set(frame.columns):
        bad = int((frame["high"] < frame["low"]).sum())
        if bad:
            report.add(symbol, "high_lt_low", f"{bad} Bars mit high < low", "error")

    if "volume" in frame.columns:
        n_zero_vol = int((frame["volume"].fillna(0) <= 0).sum())
        if n_zero_vol:
            report.add(symbol, "zero_volume", f"{n_zero_vol} Bars mit Volumen <= 0")

    # Vollständigkeit — **entweder** exakt **oder** über die Toleranzschwelle,
    # nie beides.
    #
    # Beides nebeneinander wäre falsch, nicht nur redundant: zwischen dem
    # 2001-09-10 und dem 2001-09-17 liegen sieben Tage, weil die NYSE nach den
    # Anschlägen vier Tage geschlossen war. Der Handelskalender weiß das; die
    # 5-Tage-Schwelle würde daraus einen Fehlalarm machen. Dasselbe für
    # Hurrikan Sandy 2012 und jede Staatstrauer.
    if len(idx) > 1:
        if cal.all_days_trade:
            expected: pd.DatetimeIndex | None = pd.date_range(idx.min(), idx.max(), freq="D")
        else:
            expected = trading_sessions(cal.name, idx.min().date(), idx.max().date())

        if expected is not None and len(expected):
            missing = expected.difference(idx)
            if len(missing):
                report.add(
                    symbol,
                    "missing_days",
                    f"{len(missing)} von {len(expected)} Handelstagen fehlen zwischen "
                    f"{idx.min().date()} und {idx.max().date()} "
                    f"(Kalender {cal.name}): {_list_dates(missing)}",
                )
            # Die Gegenrichtung: Bars an Tagen, an denen die Börse zu war.
            # Ein Kurs vom Feiertag ist kein harmloser Extradatensatz — er
            # verschiebt jede Rendite, die über ihn hinweg gerechnet wird, und
            # sieht dabei aus wie ein normaler Handelstag. Nur prüfbar,
            # seit der Kalender weiß, wann geschlossen war.
            unexpected = idx.difference(expected)
            if len(unexpected):
                report.add(
                    symbol,
                    "unexpected_session",
                    f"{len(unexpected)} Bars an Tagen ohne Handel "
                    f"(Kalender {cal.name}): {_list_dates(unexpected)}",
                )
        else:
            # Rückfall ohne Handelskalender: nur grobe Löcher, keine Aussage
            # über einzelne fehlende Tage.
            gaps = idx.to_series().diff().dt.days.dropna()
            big = gaps[gaps > gap_limit]
            if len(big):
                report.add(
                    symbol,
                    "calendar_gap",
                    f"{len(big)} Lücken > {gap_limit}d (max {int(big.max())}d, "
                    f"Kalender {cal.name}, ohne Handelskalender)",
                )

    # Randabschnitt: eine Serie, die schlicht später anfängt als angefordert,
    # erzeugt **keine** Lücke — zwischen ihren Zeilen ist alles lückenlos. Ohne
    # diese Prüfung ist ein Symbol mit vier Monaten fehlender Historie vom
    # Gate aus nicht von einem vollständigen zu unterscheiden.
    if expected_start is not None and idx.min().date() > expected_start:
        n, einheit = _missing_at_edge(cal, expected_start, idx.min().date())
        if n:
            report.add(
                symbol,
                "coverage_truncated",
                f"beginnt erst {idx.min().date()}, angefordert ab {expected_start} "
                f"({n} {einheit} fehlen am Anfang)",
            )
    if expected_end is not None and idx.max().date() < expected_end:
        n, einheit = _missing_at_edge(cal, idx.max().date(), expected_end)
        if n:
            report.add(
                symbol,
                "coverage_truncated",
                f"endet schon {idx.max().date()}, angefordert bis {expected_end} "
                f"({n} {einheit} fehlen am Ende)",
            )

    return report


def check_universe(frames: dict[str, pd.DataFrame], **kwargs) -> QualityReport:
    """Prüft mehrere Symbole und sammelt alle Issues in einem Report."""
    report = QualityReport()
    for sym, frame in frames.items():
        check_symbol(sym, frame, report=report, **kwargs)
    return report
