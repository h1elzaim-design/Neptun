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

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Calendar:
    name: str
    periods_per_year: float
    description: str


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
    ),
    "crypto_24_7": Calendar(
        name="crypto_24_7",
        periods_per_year=365.0,
        description="Durchgehender Handel: jeder Kalendertag ist ein Handelstag.",
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
    """Kurzform für ``get_calendar(name).periods_per_year``."""
    return get_calendar(name).periods_per_year


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

    resolved = resolve_symbol_costs(list(symbols))
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
