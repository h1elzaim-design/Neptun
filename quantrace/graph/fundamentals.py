"""Fundamentaldaten auf den Kursindex bringen — ohne Look-ahead.

Die Graph-IR schließt Look-ahead strukturell aus (siehe `nodes.py`): jede
Operation ist kausal, und der einzige zeitverschiebende Knoten schiebt nur in
die Vergangenheit. Fundamentaldaten sind die erste Quelle, die diese Garantie
**von außen** brechen könnte — und zwar an einer Stelle, die harmlos aussieht.

**Der Fehler, um den es geht.** Ein Geschäftsbericht trägt zwei Daten: das Ende
des Berichtszeitraums und den Tag der Einreichung. Dazwischen liegen Wochen.
Apples Geschäftsjahr endet Ende September, der 10-K kommt Ende Oktober. Wer den
Umsatz ab dem **Periodenende** in die Zeitreihe legt, gibt der Strategie einen
Monat lang Zahlen, die noch niemand kennen konnte — und weil die Reihe lückenlos
aussieht, fällt es nirgends auf.

Deshalb richtet dieses Modul nach ``Fact.usable_from`` aus, nicht nach dem
Periodenende — und nicht einmal nach ``filed``. Denn ``filed`` ist ein Datum
**ohne Uhrzeit**, und gemessen über AAPL, MSFT, JNJ, XOM und JPM werden 93 von
108 Geschäftsberichten (86 %) **nach Börsenschluss** angenommen. Wer `filed`
nimmt, verteilt in der Mehrheit der Fälle Information, die an dem Tag nicht mehr
handelbar war.

Ist die Annahmezeit bekannt (aus ``submissions``), wird genau unterschieden.
Sonst gilt der Folgetag: höchstens ein Tag zu spät, nie einen zu früh.

**Restatements** kommen dazu: dieselbe Periode wird später korrigiert
eingereicht. Die Stufenfunktion bildet das ab — der Wert ändert sich am Tag der
Korrektur, nicht rückwirkend. Ein Backtest sieht dann genau das, was ein
Mensch an dem Tag gesehen hätte, inklusive der Korrektur als Ereignis.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from quantrace.providers import edgar
from quantrace.providers.edgar import Fact

log = logging.getLogger(__name__)

#: Company-Facts sind ein Request pro Firma und ändern sich täglich höchstens
#: einmal. Ein Graph mit drei Fundamental-Knoten würde sonst dreimal dasselbe
#: holen — bei 0,15 s Mindestabstand und einem Universum aus 16 Symbolen ist
#: das der Unterschied zwischen 2,5 und 7,5 Sekunden pro Lauf.
_facts_cache: dict[str, dict | None] = {}

#: Accession → Annahmezeit, je Ticker. Zweiter Request pro Firma, aber er kauft
#: die Unterscheidung „vor oder nach Börsenschluss eingereicht" — und damit für
#: 14 % der Berichte einen Tag, den die pauschale Regel verschenken würde.
_acceptance_cache: dict[str, dict] = {}
#: (Ticker, Stichtag) → CIK. Der Stichtag gehört in den Schlüssel: derselbe
#: Ticker kann zu zwei Zeitpunkten zwei Firmen bezeichnen.
_cik_cache: dict[tuple[str, date], str | None] = {}


def clear_cache() -> None:
    """Caches leeren. Für Tests und lange laufende Prozesse."""
    _facts_cache.clear()
    _acceptance_cache.clear()
    _cik_cache.clear()


def _cik(ticker: str, as_of: date) -> str | None:
    """CIK **zum Stichtag**, oder ``None`` mit Begründung im Log.

    Nicht ``edgar.ticker_to_cik``: das ist die Heute-Karte, und die zeigt für
    ein neu vergebenes Kürzel auf die Nachfolgefirma. Für ein Universum aus
    dem Querschnitt von 2007 lägen damit die Bilanzzahlen von Overstock auf
    den Kursen von Bed Bath & Beyond — zwei Fehlzuordnungen, die sich
    gegenseitig plausibel machen und von denen keine eine Meldung auslöst.

    ``resolve.resolve_cik`` prüft das gegen Schicht 2 und liefert `None`,
    sobald der Stichtag in ein abgelöstes Segment fällt.
    """
    schluessel = (ticker, as_of)
    if schluessel not in _cik_cache:
        from quantrace.resolve import resolve_cik

        try:
            aufloesung = resolve_cik(ticker, as_of)
        except Exception as exc:  # noqa: BLE001 - Lake nicht erreichbar
            log.warning("CIK-Auflösung für %s nicht möglich (%s).", ticker, exc)
            _cik_cache[schluessel] = None
            return None

        if not aufloesung.usable:
            log.info("Fundamentals: %s → %s. %s", ticker, aufloesung.status, aufloesung.reason)
        elif aufloesung.status == "unverified":
            log.warning("Fundamentals: %s ungeprüft. %s", ticker, aufloesung.reason)
        _cik_cache[schluessel] = aufloesung.cik if aufloesung.usable else None
    return _cik_cache[schluessel]


def _company_facts(ticker: str, as_of: date) -> dict | None:
    if ticker not in _facts_cache:
        cik = _cik(ticker, as_of)
        _facts_cache[ticker] = edgar.fetch_company_facts(cik) if cik else None
    return _facts_cache[ticker]


def _acceptance(ticker: str, as_of: date) -> dict:
    """Annahmezeiten der jüngsten Einreichungen. Leeres dict, wenn nicht abrufbar.

    Nie werfen: ohne Annahmezeit fällt `Fact.usable_from` auf den Folgetag
    zurück. Das ist konservativ und korrekt — ein Ausfall hier darf keinen
    Backtest kippen, er macht ihn nur um höchstens einen Tag träger.
    """
    if ticker not in _acceptance_cache:
        try:
            cik = _cik(ticker, as_of)
            subs = edgar.fetch_submissions(cik) if cik else None
            _acceptance_cache[ticker] = edgar.acceptance_map(subs)
        except Exception as exc:  # noqa: BLE001 — siehe Docstring
            log.warning("Annahmezeiten für %s nicht abrufbar (%s).", ticker, exc)
            _acceptance_cache[ticker] = {}
    return _acceptance_cache[ticker]


def knowledge_steps(facts: list[Fact]) -> dict[date, float]:
    """Verfügbarkeitsdatum → Wert der dann jüngsten bekannten Periode.

    Der Schlüssel ist `Fact.usable_from`, nicht `filed`: 86 % der 10-K/10-Q
    werden nach Börsenschluss angenommen und sind am Einreichungstag nicht
    handelbar.

    Eine Stufenfunktion des Wissensstands. An jedem Tag, an dem eingereicht
    wurde, steht dort der Wert, den man ab diesem Tag genannt hätte.

    Zwei Feinheiten, beide mit Folgen:

    * Eine **Korrektur einer alten Periode** ändert den Wert nicht, solange eine
      neuere Periode bekannt ist — man berichtet weiter die neueste Zahl.
    * Eine **Korrektur der neuesten Periode** ändert ihn sehr wohl, und zwar am
      Tag der Korrektur. Das ist kein Fehler, sondern genau der Verlauf, den ein
      Beobachter erlebt hätte.
    """
    best: dict[tuple[date | None, date], Fact] = {}
    steps: dict[date, float] = {}

    for f in sorted(facts, key=lambda x: (x.usable_from, x.period_end)):
        # Spätere Einreichung derselben Periode überschreibt die frühere.
        best[(f.period_start, f.period_end)] = f
        newest = max(best.values(), key=lambda x: x.period_end)
        steps[f.usable_from] = newest.value

    return steps


def align_to_index(steps: dict[date, float], index: pd.Index) -> pd.Series:
    """Stufenfunktion auf einen Kursindex legen, vorwärts gefüllt.

    Vor der ersten Einreichung steht ``NaN`` — nicht 0.0. Das ist bewusst: eine
    Firma ohne veröffentlichte Zahlen hat keinen Umsatz von null, sie hat einen
    unbekannten. Der Unterschied entscheidet, ob ein Screener sie aussortiert
    oder für die billigste Aktie im Universum hält. (`_score_realism` hat diese
    Lektion schon einmal geliefert — fehlende Werte sind nicht Null.)
    """
    if not steps:
        return pd.Series(float("nan"), index=index, dtype="float64")

    raw = pd.Series(steps, dtype="float64")
    raw.index = pd.to_datetime(list(steps.keys()))
    raw = raw.sort_index()

    # `reindex(..., method="ffill")` auf der Vereinigung, dann auf den Zielindex:
    # so wirken auch Einreichungen an handelsfreien Tagen ab dem nächsten
    # Handelstag, statt verloren zu gehen.
    target = pd.DatetimeIndex(index)
    combined = raw.reindex(raw.index.union(target)).ffill()
    return combined.reindex(target)


def fundamental_frame(concept: str, symbols: list[str], index: pd.Index) -> pd.DataFrame:
    """Breite Matrix (Index=Handelstage, Spalten=Symbole) für eine Kennzahl.

    Symbole ohne CIK — ETFs, Nicht-US-Filer — bekommen eine reine NaN-Spalte
    statt zu fehlen. Eine fehlende Spalte würde jede nachgelagerte Rechnung
    stillschweigend auf ein kleineres Universum verkürzen; eine NaN-Spalte
    fällt in jeder Vergleichsoperation korrekt heraus und bleibt sichtbar.

    **Dasselbe gilt für einen neu vergebenen Ticker.** Die Identität wird zum
    **Fensteranfang** aufgelöst, nicht nach der Heute-Karte: ein Universum aus
    dem Querschnitt von 2007 bekommt sonst die Bilanz der Nachfolgefirma auf
    den Kursen der alten. Fällt der Fensteranfang in ein abgelöstes Segment,
    ist die Spalte NaN — die vorsichtige Antwort, nicht die schmeichelhafte.
    """
    period = edgar.DEFAULT_PERIODS.get(concept, "any")
    cols: dict[str, pd.Series] = {}

    # Der Anfang des Fensters, nicht sein Ende: die Auswahl des Universums
    # geschah zu diesem Zeitpunkt, und über einen Besitzerwechsel hinweg gibt
    # es ohnehin keine richtige Wahl (siehe `resolve.resolve_symbols`).
    stichtag = pd.Timestamp(index[0]).date() if len(index) else date.today()

    for sym in symbols:
        try:
            facts = edgar.concept_series(
                _company_facts(sym, stichtag), concept, period=period
            )
            facts = edgar.attach_acceptance(facts, _acceptance(sym, stichtag))
        except Exception as exc:  # noqa: BLE001 — eine kaputte Firma kippt nicht den Lauf
            log.warning("Fundamentals: %s/%s nicht abrufbar (%s).", sym, concept, exc)
            facts = []
        cols[sym] = align_to_index(knowledge_steps(facts), index)

    return pd.DataFrame(cols, index=pd.DatetimeIndex(index))
