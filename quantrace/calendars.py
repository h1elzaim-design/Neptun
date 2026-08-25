"""Handelskalender — wie viele Perioden ein Jahr hat (#184).

`periods_per_year` steckte als Default `252.0` in acht Statistik-Modulen. Für
US-Aktien und ETFs stimmt das; für alles, was 24/7 handelt, nicht. Der Fehler
wäre nicht klein und nicht sichtbar: Sharpe skaliert mit √P, ein Crypto-Sharpe
mit 252 statt 365 gerechnet sähe um den Faktor

    √(365/252) ≈ 1.204

**zu gut** aus — und die ganze Disziplin-Schicht darüber (DSR, PBO, Bootstrap,
Governance-Score) würde diesen Wert präzise weiterverarbeiten. Ein Kandidat
könnte allein durch die falsche Annualisierung über die 0.70-Schwelle rutschen.

Deshalb ist der Kalender ab hier ein **Datenfeld am Universum**, keine Annahme
im Rechencode. `data/universes/*.yaml` deklariert ihn, `MarketData` trägt ihn,
der Backtest-Runner leitet die Annualisierung daraus ab. Die Defaults in
`quantrace/stats/*` bleiben stehen — sie sind Bibliotheks-Bequemlichkeit für
direkte Aufrufe; im Backtest-Pfad wird der Wert jetzt immer explizit gereicht.

Bewusst **nur Daily**: der Lake-Pfad lehnt Intraday ohnehin ab
(`_load_via_lake`), und ein Stunden-Kalender, den nichts benutzt, wäre
ungetestete Spekulation. Wenn Intraday kommt, kommt er mit eigenen Tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Calendar:
    name: str
    periods_per_year: float
    description: str
    #: Ab wie vielen Tagen zwischen zwei Datenpunkten eine Lücke verdächtig ist.
    #: Für 24/7 ist das 1 — jeder Kalendertag muss da sein. Für Börsen mit
    #: Wochenende deckt 5 ein langes Feiertagswochenende ab.
    max_gap_days: int
    #: Handelt an **jedem** Kalendertag. Dann sind die erwarteten Tage trivial
    #: und es braucht keinen Börsenkalender.
    all_days_trade: bool
    #: MIC (ISO 10383) des Handelsplatzes für `exchange_calendars`, oder None
    #: bei `all_days_trade`. Damit wird Vollständigkeit auch für Börsen mit
    #: Feiertagen exakt statt nur näherungsweise prüfbar.
    exchange: str | None


#: Der Kalender, den ein Universum ohne Angabe bekommt.
#:
#: `us_equity` und nicht etwa ein Fehler: alle neun bestehenden Universen sind
#: US-gelistet, und ein Default, der die Wirklichkeit trifft, hält den Diff
#: dieses PRs bei null Verhaltensänderung. Sobald ein Universum einen anderen
#: Kalender braucht, muss es ihn nennen — und der Guardrail (Schritt B) lehnt
#: gemischte Universen ab, statt still einen zu wählen.
DEFAULT_CALENDAR = "us_equity"

CALENDARS: dict[str, Calendar] = {
    "us_equity": Calendar(
        name="us_equity",
        periods_per_year=252.0,
        description="US-Börsen: ~252 Handelstage pro Jahr, Wochenenden und Feiertage frei.",
        # Fr→Mo sind 3 Tage, ein langes Feiertagswochenende 4. 5 lässt beides
        # durch und meldet alles darüber. Bleibt als Rückfall, falls der
        # Börsenkalender nicht verfügbar ist.
        max_gap_days=5,
        all_days_trade=False,
        exchange="XNYS",
    ),
    "crypto_24_7": Calendar(
        name="crypto_24_7",
        periods_per_year=365.0,
        description="Durchgehender Handel: jeder Kalendertag ist ein Handelstag.",
        max_gap_days=1,
        all_days_trade=True,
        exchange=None,
    ),
}


class UnknownCalendarError(ValueError):
    """Ein Universum nennt einen Kalender, den es nicht gibt."""


def get_calendar(name: str | None) -> Calendar:
    """Kalender nachschlagen. ``None`` → Default.

    Ein unbekannter Name wirft, statt auf den Default zurückzufallen: ein
    Tippfehler in `calendar:` würde sonst still die falsche Annualisierung
    ergeben, und genau das soll dieses Modul verhindern.
    """
    key = (name or DEFAULT_CALENDAR).strip().lower()
    if key not in CALENDARS:
        raise UnknownCalendarError(
            f"Unbekannter Kalender '{name}'. Erlaubt: {', '.join(sorted(CALENDARS))}."
        )
    return CALENDARS[key]


def periods_per_year(name: str | None) -> float:
    """Kurzform für ``get_calendar(name).periods_per_year``.

    **Nicht** durch die echte Jahres-Sessionzahl ersetzen, auch wenn
    ``trading_sessions`` sie jetzt liefern könnte. Die tatsächliche Zahl der
    NYSE-Handelstage schwankt zwischen 248 (2001, vier Tage nach dem 11.
    September) und 254 (1996) — durch Wochentagsverschiebung, Feiertage auf
    Wochenenden und ungeplante Schließungen (Sandy 2012, Staatstrauer).

    252 ist eine **Konvention**, kein Messwert. Ihr Zweck ist, dass zwei Sharpe-
    Werte aus verschiedenen Jahren vergleichbar sind. Jahresgenau annualisiert
    ergäbe derselbe Return-Pfad je nach Jahr bis zu √(254/248) ≈ 1.2 % andere
    Kennzahlen — Rauschen aus dem Kalender, das wie Signal aussieht.

    Der Unterschied 252 ↔ 365 ist eine andere Größenordnung (Faktor 1.45) und
    kommt aus der Struktur des Marktes. Genau deshalb lohnt der eine und der
    andere nicht.
    """
    return get_calendar(name).periods_per_year


# ---------------------------------------------------------------------------
# Börsenkalender (exchange_calendars)


#: Feste Grenzen für den erzeugten Kalender.
#:
#: `exchange_calendars` liefert ohne Angabe ein **rollierendes** Fenster von
#: rund ±20 Jahren um heute. Das ist die gefährlichste Eigenschaft der
#: Bibliothek: ein Backtest über 2005 (us_core_etfs nennt 2005–2024) läge
#: außerhalb, und alles davor sähe aus wie durchgehend Feiertag. Feste Grenzen
#: statt Default — und wer außerhalb fragt, bekommt eine Warnung statt einer
#: stillen Falschantwort.
CALENDAR_START = "1970-01-01"
CALENDAR_END = "2035-12-31"


@lru_cache(maxsize=4)
def _exchange_calendar(mic: str):
    """Börsenkalender bauen und cachen (~0.5 s, ~15k Sessions für XNYS)."""
    import exchange_calendars as xcals

    return xcals.get_calendar(mic, start=CALENDAR_START, end=CALENDAR_END)


@lru_cache(maxsize=1)
def _warn_missing_package(exc: str) -> None:
    """Einmal warnen, nicht pro Symbol."""
    log.warning(
        "exchange_calendars nicht verfügbar (%s) — Vollständigkeitsprüfung für "
        "Börsenkalender fällt auf die Toleranzschwelle zurück.",
        exc,
    )


def trading_sessions(name: str | None, start: date, end: date) -> pd.DatetimeIndex | None:
    """Die Handelstage eines Kalenders im Fenster ``start..end``.

    Returns
    -------
    pd.DatetimeIndex | None
        ``None`` bedeutet **nicht** „keine Handelstage", sondern „nicht
        bestimmbar" — drei Fälle, alle geloggt:

        * ``all_days_trade``: der Aufrufer kennt die Antwort selbst (jeder Tag).
        * ``exchange_calendars`` fehlt: der Aufrufer fällt auf die
          Toleranzschwelle zurück.
        * Fenster außerhalb ``CALENDAR_START..CALENDAR_END``.

        Die Unterscheidung ist wichtig: ein leerer Index hieße „hier wurde
        nie gehandelt", und daraus würde die Qualitätsprüfung das Gegenteil
        des Richtigen schließen.
    """
    cal = get_calendar(name)
    if cal.exchange is None:
        return None

    if start < date.fromisoformat(CALENDAR_START) or end > date.fromisoformat(CALENDAR_END):
        log.warning(
            "Fenster %s..%s liegt außerhalb des erzeugten Kalenders (%s..%s) — "
            "Vollständigkeit wird nicht geprüft.",
            start, end, CALENDAR_START, CALENDAR_END,
        )
        return None

    try:
        xcal = _exchange_calendar(cal.exchange)
    except ImportError as exc:
        _warn_missing_package(str(exc))
        return None

    return xcal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))


# ---------------------------------------------------------------------------
# Guardrail: ein Universum, ein Kalender


#: Welche Kostenklasse zu welchem Kalender gehört. Alles, was hier nicht steht,
#: ist ``us_equity`` — die Klassen in `config/costs.yaml` sind sonst durchweg
#: US-gelistete Wertpapiere.
CLASS_CALENDARS: dict[str, str] = {
    "crypto_major": "crypto_24_7",
    "crypto_alt": "crypto_24_7",
}


def calendar_for_class(asset_class: str) -> str:
    return CLASS_CALENDARS.get(asset_class, DEFAULT_CALENDAR)


class CalendarMismatchError(ValueError):
    """Ein Universum mischt Kalender — der Backtest wird abgelehnt."""


def validate_universe_calendar(
    symbols: list[str] | tuple[str, ...],
    calendar: str | None,
    *,
    universe: str = "",
    cost_class: str | None = None,
) -> None:
    """Prüfen, dass alle Symbole zum deklarierten Kalender passen.

    Ein Universum aus BTC **und** SPY hätte keine ehrliche Annualisierung: 252
    wäre für BTC falsch, 365 für SPY. Statt einen der beiden Werte zu wählen
    und die Hälfte der Zahlen zu verfälschen, wird der Lauf abgelehnt.

    Die Kalender-Zuordnung läuft über die Kostenklasse aus `config/costs.yaml`
    — dort steht ohnehin schon, was ein Symbol ist.

    **Eine Lücke, die hier ehrlich benannt gehört:** ein *unklassifiziertes*
    Symbol fällt auf ``default_class`` (und damit ``us_equity``) zurück. In
    einem Crypto-Universum fliegt es dadurch auf — es sieht wie Equity aus und
    passt nicht zum deklarierten Kalender. In einem us_equity-Universum bleibt
    ein unklassifiziertes Crypto-Symbol dagegen unentdeckt, weil an dieser
    Stelle niemand weiß, dass es Crypto ist. Das wäre erst mit einer Abfrage
    gegen den Instrument-Master zu schließen; bis dahin bleibt die Warnung aus
    `resolve_symbol_costs` das einzige Signal.
    """
    if not symbols:
        return

    declared = get_calendar(calendar).name

    # Lazy: `costs` importiert `models`, und `models` importiert dieses Modul.
    # Ein Top-Level-Import wäre ein Zyklus.
    from quantrace.costs import resolve_symbol_costs

    # Ein konstruiertes Universum deklariert seine Klasse im YAML. Ohne diesen
    # Durchreicher fiele jedes seiner Symbole auf `default_class` und damit auf
    # `us_equity` — hier zufällig richtig, aber aus dem falschen Grund: die
    # Prüfung wäre wirkungslos statt bestanden.
    resolved = resolve_symbol_costs(list(symbols), fallback_class=cost_class)
    offenders: dict[str, str] = {}
    for sym, profile in resolved.items():
        actual = calendar_for_class(profile.asset_class)
        if actual != declared:
            offenders[sym] = f"{profile.asset_class} → {actual}"

    if offenders:
        where = f" '{universe}'" if universe else ""
        listed = ", ".join(f"{s} ({why})" for s, why in sorted(offenders.items()))
        raise CalendarMismatchError(
            f"Universum{where} deklariert calendar '{declared}', aber diese Symbole "
            f"gehören zu einem anderen Kalender: {listed}. Ein Universum, ein "
            "Kalender — sonst wäre die Annualisierung für einen Teil der Symbole "
            "falsch. Symbole aufteilen oder in config/costs.yaml korrekt "
            "klassifizieren."
        )
