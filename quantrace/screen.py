"""Universum als **Regel zu einem Stichtag** statt als abgetippte Liste.

Eine heute geschriebene Symbolliste ist eine Liste von Überlebenden. Das ist
kein Vorwurf an den Schreiber — es ist eine Eigenschaft des Zeitpunkts: wer
2008 pleiteging, steht in keinem Screener von 2026. Deshalb misst dieses Modul
den **Querschnitt zum Stichtag**: welche Papiere handelten am 30.06.2007 über
5 Mio. $ am Tag? Die Antwort enthält Lehman Brothers.

## Die eine Regel, die dieses Modul einhält

**Es wird keine Partition gelesen, deren Datum nach dem Stichtag liegt.**

Das ist die gesamte Point-in-time-Garantie, und sie ist billig zu halten,
solange sie an *einer* Stelle steht: ``_fenster()`` filtert die Tagesliste, und
alles andere sieht nur noch, was übrig bleibt. ``test_screen.py`` legt eine
Partition *nach* dem Stichtag mit absurden Werten an — taucht sie im Ergebnis
auf, ist die Garantie gebrochen und der Test rot.

## Warum Median und nicht Mittelwert

Ein einzelner Block-Trade hebt das mittlere Dollarvolumen eines sonst toten
Papiers über jede Schwelle. Der Median über ~60 Handelstage fragt stattdessen:
*handelt das hier üblicherweise?* — und genau das ist die Frage, die hinter
einer Liquiditätsschwelle steht.

## Was der Trichter beantwortet

„Warum sind in meinem Universum nur drei Namen?" ist sonst eine Stunde Arbeit.
``ScreenResult.funnel`` zählt je Stufe mit, wie viele Kandidaten dort
ausschieden — in fester Reihenfolge, damit die Zahlen sich addieren.

## Zwei Formen, und die zweite ist die ehrlichere

``screen()`` liefert einen **Schnappschuss** zum Stichtag: korrekt für den Tag,
an dem er gezogen wurde, und danach zunehmend veraltet, weil Neuemissionen nie
hinzukommen. Wer 2000 screent und bis 2009 backtestet, handelt neun Jahre lang
einen Korb, den es nach 2000 nie wieder gab — kein Google (IPO 2004), kein
Tesla (2010).

``reconstitute()`` wertet dieselbe Regel **alle N Monate neu** aus und führt
eine zeitvariable Mitgliedschaft, so wie ein Index es tut (#255). Die
Point-in-time-Garantie gilt dabei je Stichtag unverändert: keiner der Läufe
liest über seinen eigenen Stichtag hinaus, und keiner weiß vom anderen.

Die Zahl, an der man die beiden Formen gegeneinander abwägt, ist
``Reconstitution.drift``: wie viel des ersten Korbs die Regel am letzten
Stichtag nicht mehr wählt. Solange sie klein ist, war der Schnappschuss
vertretbar. Ist sie es nicht, hat sie es nie verraten — genau deshalb wird sie
gemessen.

Durchgesetzt wird die Mitgliedschaft nicht hier, sondern beim Lesen:
``quantrace.membership``.
"""

from __future__ import annotations

import logging
import re
from calendar import monthrange
from dataclasses import dataclass, field, replace
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from quantrace import storage
from quantrace.instruments import US_EQUITY_PREFIX
from quantrace.resolve import DEFAULT_ACTIVE_TOLERANCE_DAYS

if TYPE_CHECKING:  # pragma: no cover - nur für Typannotationen
    from quantrace.membership import Membership

log = logging.getLogger(__name__)

#: Handelstage im Rückblickfenster. 60 ≈ ein Quartal — lang genug, dass ein
#: einzelner ruhiger Monat das Bild nicht kippt, kurz genug, dass ein Papier,
#: das gerade erst liquide wurde, nicht ein Jahr lang draußen bleibt.
DEFAULT_LOOKBACK_DAYS = 60

#: Anteil der Fenstertage, an denen ein Papier gehandelt haben muss. Fängt
#: frische Emissionen und ausgesetzte Titel ab, ohne einen einzelnen
#: Feiertagsausfall zu bestrafen.
DEFAULT_MIN_COVERAGE = 0.8

#: Unterste Liquiditätsschwelle, die dieses Projekt zu bepreisen wagt. Darunter
#: ist keine statische bps-Zahl mehr ehrlich — siehe ``costs.class_for_liquidity``.
MIN_SUPPORTED_DOLLAR_VOLUME = 1_000_000.0

#: Wie weit das Rückblickfenster hinter dem Stichtag enden darf (Kalendertage).
#:
#: 30 Tage decken Feiertage, Wochenenden und einzelne fehlende Partitionen ab.
#: Alles darüber ist keine Unschärfe mehr, sondern ein anderer Zeitraum — siehe
#: ``_fenster``.
MAX_WINDOW_STALENESS_DAYS = 30

#: EODHDs Sentinel für „kein Kurs ermittelbar" — kein Nullwert, sondern eine
#: konkrete Zahl, die wie ein echter Kurs aussieht. Gefunden am 2026-08-26 beim
#: Bau der ersten Point-in-Time-Universen: ein Screen zum 2007-06-29 wählte auf
#: den Rängen 1, 2, 8, 12 und 14 Papiere mit exakt diesem „Kurs" — macht daraus
#: ein Dollarvolumen im zweistelligen Milliardenbereich und verdrängt echte
#: liquide Titel (AAPL, GOOG, MSFT) aus der Top-Auswahl. `_aggregat` filtert das
#: jetzt vorne raus, nicht bloß `close > 0`.
_EODHD_NULL_PRICE_SENTINEL = 999999.9999


class ScreenError(ValueError):
    """Fachlicher Fehler beim Screen. Der Router macht daraus 409."""


@dataclass(frozen=True)
class ScreenCriteria:
    """Die Regel. Jedes Feld ist eine Aussage, die im YAML landet.

    Die Reihenfolge der Filter ist Teil der Semantik, nicht Implementierungs-
    detail: ``funnel`` zählt je Stufe die Kandidaten, die *alle vorherigen*
    Stufen bestanden haben. Nur so addieren sich die Zahlen zur Kandidatenzahl.
    """

    as_of: date
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    min_dollar_volume: float = 5_000_000.0
    min_price: float = 5.0
    min_coverage: float = DEFAULT_MIN_COVERAGE
    top_n: int | None = None
    #: Leer = alle Börsen des Feeds. Sonst z.B. ``("NYSE", "NASDAQ")``.
    exchanges: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.lookback_days < 1:
            raise ScreenError("lookback_days muss mindestens 1 sein.")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ScreenError("min_coverage ist ein Anteil zwischen 0 und 1.")
        if self.min_dollar_volume < MIN_SUPPORTED_DOLLAR_VOLUME:
            raise ScreenError(
                f"Liquiditätsschwelle {self.min_dollar_volume:,.0f} $ liegt unter "
                f"{MIN_SUPPORTED_DOLLAR_VOLUME:,.0f} $. Darunter lässt sich der "
                "Handel nicht mehr mit einer festen bps-Zahl bepreisen — eine "
                "Market-Order nahe dem Close bewegt dort den Kurs selbst. Setz "
                "die Schwelle höher, statt Kosten zu erfinden."
            )
        if self.top_n is not None and self.top_n < 1:
            raise ScreenError("top_n muss mindestens 1 sein, oder None für alle.")

    def as_dict(self) -> dict[str, object]:
        """Die Regel als YAML-fähige Abbildung — die Reproduzierbarkeitszusage."""
        return {
            "as_of": self.as_of.isoformat(),
            "lookback_days": self.lookback_days,
            "min_dollar_volume": float(self.min_dollar_volume),
            "min_price": float(self.min_price),
            "min_coverage": float(self.min_coverage),
            "top_n": self.top_n,
            "exchanges": list(self.exchanges),
        }


@dataclass(frozen=True)
class Survivorship:
    """Der **Beleg**, nicht die Behauptung.

    ``delisted_included`` steht in jedem Universe-YAML. Für handverlesene
    Listen ist es eine Zusage, die niemand prüft. Für einen Regel-Schnappschuss
    ist es messbar: wie viele der ausgewählten Papiere hören im Lake *vor
    dessen Ende* auf zu handeln? Papiere, die ein Screener von heute nie
    gefunden hätte.

    **Das ist kein Look-ahead.** Die Zahl beschreibt das Universum, sie wählt
    nicht aus — die Auswahl war zum Stichtag abgeschlossen, bevor hier auch nur
    eine Zeile gelesen wurde. Wer sie als Signal benutzt, hat allerdings
    Wissen aus der Zukunft: deshalb wandert die *Anzahl* ins YAML, aber nicht
    die Liste der Todeszeitpunkte.
    """

    n_selected: int
    n_delisted: int
    lake_last: date | None
    #: Eine Handvoll Beispiele für die Anzeige — bewusst ohne Datum.
    examples: tuple[str, ...] = ()

    @property
    def includes_delisted(self) -> bool:
        return self.n_delisted > 0

    @property
    def share(self) -> float:
        return self.n_delisted / self.n_selected if self.n_selected else 0.0

    def note(self) -> str:
        if not self.n_selected:
            return "Kein Papier ausgewählt — nichts zu belegen."
        if not self.n_delisted:
            return (
                "Kein ausgewähltes Papier endet vor dem Lake-Ende. Entweder ist "
                "der Stichtag zu nah an heute, oder der Lake reicht nicht weit "
                "genug — in beiden Fällen ist dieses Universum survivor-lastig."
            )
        return (
            f"{self.n_delisted} von {self.n_selected} Papieren "
            f"({self.share:.1%}) handeln nach dem Stichtag nicht mehr bis zum "
            "Lake-Ende. Genau diese hätte eine heute geschriebene Liste nicht "
            "enthalten."
        )


@dataclass(frozen=True)
class ScreenResult:
    """Was die Regel zum Stichtag ergab — samt Weg dorthin."""

    criteria: ScreenCriteria
    window_start: date
    window_end: date
    n_window_days: int
    #: Eine Zeile je zulässigem Papier, nach Dollarvolumen absteigend.
    rows: pd.DataFrame
    #: Stufenname → wie viele Kandidaten dort ausschieden. Reihenfolge = Ablauf.
    funnel: dict[str, int] = field(default_factory=dict)
    survivorship: Survivorship = field(
        default_factory=lambda: Survivorship(0, 0, None)
    )

    @property
    def symbols(self) -> list[str]:
        return [str(c) for c in self.rows["code"]] if not self.rows.empty else []

    @property
    def n_selected(self) -> int:
        return int(len(self.rows))

    @property
    def liquidity_floor(self) -> float:
        """Das **kleinste** Dollarvolumen unter den Ausgewählten.

        Nicht die konfigurierte Schwelle: bei aktivem ``top_n`` liegt der
        tatsächliche Boden oft weit darüber. Er bestimmt die Kostenklasse,
        also muss er gemessen sein.
        """
        if self.rows.empty:
            return 0.0
        return float(self.rows["dollar_volume"].min())


def _fenster(as_of: date, lookback_days: int, alle: list[date] | None = None) -> list[date]:
    """Die Handelstage des Rückblickfensters — **nie über den Stichtag hinaus**.

    Die einzige Stelle, an der die Point-in-time-Garantie hängt. Sie ist
    absichtlich klein und langweilig: eine Garantie, die über drei Funktionen
    verteilt ist, hält irgendwann eine davon nicht mehr ein.

    ``alle`` ist die Tagesliste, falls der Aufrufer sie schon hat
    (``reconstitute`` ruft den Screen zwanzigmal — zwanzig LIST-Operationen
    über denselben Präfix für dieselbe Antwort wären auf R2 die teuerste
    Zeile der Funktion). Der Filter darunter bleibt derselbe: **nicht** die
    Liste ist die Garantie, sondern der Vergleich mit dem Stichtag.
    """
    if alle is None:
        alle = storage.list_day_partitions(US_EQUITY_PREFIX)
    bis_stichtag = [d for d in alle if d <= as_of]
    return bis_stichtag[-lookback_days:]


def _aggregat(tage: list[date]) -> pd.DataFrame:
    """Je (code, exchange) über das Fenster: Median-Dollarvolumen, letzter Kurs.

    Ein DuckDB-Durchlauf über die Fenster-Partitionen. Das Ergebnis ist klein
    (Größenordnung 10^4 Zeilen), deshalb passiert das Filtern danach in pandas
    — dort ist der Trichter drei Zeilen statt einer verschachtelten Query.
    """
    if not tage:
        return pd.DataFrame(
            columns=["code", "exchange", "n_days", "dollar_volume", "last_close"]
        )

    pfade = [
        storage.cache_path(f"{US_EQUITY_PREFIX}/date={t.isoformat()}/*.parquet")
        for t in tage
    ]
    platzhalter = ",".join(["?"] * len(pfade))
    sql = f"""
        SELECT
            code,
            -- Gruppiert wird über den Code allein, weil ein Universum
            -- Ticker führt: derselbe Code zweimal wäre ein doppeltes Gewicht
            -- auf einem Papier. `max` statt `any_value`, damit ein Code auf
            -- zwei Handelsplätzen deterministisch dieselbe Zeile ergibt.
            max(exchange_short_name)            AS exchange,
            COUNT(*)                            AS n_days,
            median(close * volume)              AS dollar_volume,
            arg_max(close, date)                AS last_close
        FROM read_parquet([{platzhalter}], union_by_name=true)
        WHERE close IS NOT NULL AND volume IS NOT NULL AND close > 0
          AND close != ?
        GROUP BY code
    """
    con = storage._duckdb_conn()
    try:
        return con.execute(sql, [*pfade, _EODHD_NULL_PRICE_SENTINEL]).df()
    except Exception as exc:
        raise ScreenError(
            f"Querschnitte für {tage[0]}..{tage[-1]} nicht lesbar: {exc}"
        ) from exc
    finally:
        con.close()


def _survivorship(
    codes: list[str], as_of: date, alle: list[date] | None = None
) -> Survivorship:
    """Wie viele der Ausgewählten handeln später nicht mehr?

    Gelesen wird eine **Stichprobe der Partitionen nach dem Stichtag** — für
    die Frage „läuft dieses Papier bis zum Lake-Ende weiter?" reicht der Blick
    auf die letzten Querschnitte, und ein Vollscan über tausende Tage kostet
    Minuten für eine Zahl, die nach Sekunden feststeht.

    Bewusst getrennt von der Auswahl: dieselbe Funktion, die hier liest, darf
    im Auswahlpfad nicht vorkommen (siehe Modul-Docstring).
    """
    if alle is None:
        alle = storage.list_day_partitions(US_EQUITY_PREFIX)
    danach = [d for d in alle if d > as_of]
    if not codes or not danach:
        return Survivorship(
            n_selected=len(codes), n_delisted=0, lake_last=max(alle) if alle else None
        )

    lake_last = max(danach)
    # „Handelt noch" heißt: taucht in den letzten Querschnitten auf. Dieselbe
    # Toleranz wie in `resolve` — eine Definition von „aktiv", nicht zwei.
    schwelle = pd.Timestamp(lake_last) - pd.Timedelta(days=DEFAULT_ACTIVE_TOLERANCE_DAYS)
    aktuelle = [d for d in danach if pd.Timestamp(d) >= schwelle]
    if not aktuelle:
        aktuelle = danach[-5:]

    pfade = [
        storage.cache_path(f"{US_EQUITY_PREFIX}/date={t.isoformat()}/*.parquet")
        for t in aktuelle
    ]
    platzhalter = ",".join(["?"] * len(pfade))
    con = storage._duckdb_conn()
    try:
        lebende = {
            str(r[0])
            for r in con.execute(
                f"SELECT DISTINCT code FROM read_parquet([{platzhalter}], union_by_name=true)",
                pfade,
            ).fetchall()
        }
    except Exception as exc:  # pragma: no cover - defekte Partition
        # Kein Beleg ist etwas anderes als „keine Delistings". Ohne Messung
        # bleibt die Zahl 0 und `includes_delisted` False — die vorsichtige
        # Aussage, nicht die schmeichelhafte.
        log.warning("Survivorship-Beleg nicht messbar: %s", exc)
        return Survivorship(n_selected=len(codes), n_delisted=0, lake_last=lake_last)
    finally:
        con.close()

    tot = [c for c in codes if c not in lebende]
    return Survivorship(
        n_selected=len(codes),
        n_delisted=len(tot),
        lake_last=lake_last,
        examples=tuple(sorted(tot)[:8]),
    )


def screen(
    criteria: ScreenCriteria,
    *,
    measure_survivorship: bool = True,
    day_partitions: list[date] | None = None,
) -> ScreenResult:
    """Die Regel zum Stichtag auswerten.

    ``measure_survivorship=False`` lässt den Beleg weg — für die
    Rekonstitution, die ihn einmal über die Vereinigung misst statt einmal je
    Stichtag. Der Default bleibt an: wer einen einzelnen Screen fährt, soll die
    Zahl bekommen, ohne sie anzufordern.

    ``day_partitions`` reicht eine bereits geladene Tagesliste durch (siehe
    ``_fenster``).

    Raises
    ------
    ScreenError
        Wenn im Fenster keine Querschnitte liegen. Ein leeres Universum
        stillschweigend zurückzugeben wäre die teurere Antwort: es sieht aus
        wie „keine Treffer" und ist in Wahrheit „keine Daten".
    """
    tage = _fenster(criteria.as_of, criteria.lookback_days, day_partitions)
    if not tage:
        raise ScreenError(
            f"Keine Querschnitte bis zum {criteria.as_of} im Lake. Der Stichtag "
            "liegt vor dem geladenen Zeitraum — 'python scripts/load_us_equities.py "
            "--status' zeigt, was da ist."
        )

    # Der Lake lädt rückwärts durch die Historie: „irgendwas vor dem Stichtag"
    # ist deshalb keine Zusage, dass es *nahe* am Stichtag liegt. Ohne diese
    # Prüfung würde ein Screen zum 2007-06-29 auf einem Fenster rechnen, das
    # am 2001-09-17 endet — und das Ergebnis trüge trotzdem 2007 als `as_of`
    # in der YAML. Eine falsche Herkunftsangabe ist teurer als ein Abbruch:
    # sie sieht aus wie ein Befund.
    rueckstand = (criteria.as_of - tage[-1]).days
    if rueckstand > MAX_WINDOW_STALENESS_DAYS:
        raise ScreenError(
            f"Der letzte Querschnitt vor dem {criteria.as_of} ist vom {tage[-1]} — "
            f"{rueckstand} Tage davor. Das Fenster läge damit in einem anderen "
            "Zeitraum als der Stichtag, und das Universum trüge trotzdem den "
            "Stichtag als Herkunft. Entweder den Stichtag auf das geladene Ende "
            "legen oder die Historie nachladen ('python scripts/load_us_equities.py "
            "--status' zeigt, wie weit sie reicht)."
        )

    agg = _aggregat(tage)
    kandidaten = int(len(agg))
    trichter: dict[str, int] = {}

    if agg.empty:
        return ScreenResult(
            criteria=criteria,
            window_start=tage[0],
            window_end=tage[-1],
            n_window_days=len(tage),
            rows=agg,
            funnel={"kandidaten": 0},
        )

    trichter["kandidaten"] = kandidaten

    # Reihenfolge ist Semantik: jede Stufe zählt, was *nach* allen vorherigen
    # übrig war. Umsortieren ändert die Zahlen, nicht das Endergebnis.
    noetig = max(1, int(round(criteria.min_coverage * len(tage))))
    vorher = len(agg)
    agg = agg[agg["n_days"] >= noetig]
    trichter[f"zu wenige Handelstage (< {noetig} von {len(tage)})"] = vorher - len(agg)

    vorher = len(agg)
    agg = agg[agg["last_close"] >= criteria.min_price]
    trichter[f"Kurs unter {criteria.min_price:g} $"] = vorher - len(agg)

    vorher = len(agg)
    agg = agg[agg["dollar_volume"] >= criteria.min_dollar_volume]
    trichter[f"Dollarvolumen unter {criteria.min_dollar_volume:,.0f} $"] = vorher - len(agg)

    if criteria.exchanges:
        erlaubt = {e.upper() for e in criteria.exchanges}
        vorher = len(agg)
        agg = agg[agg["exchange"].astype(str).str.upper().isin(erlaubt)]
        trichter[f"Börse nicht in {', '.join(sorted(erlaubt))}"] = vorher - len(agg)

    agg = agg.sort_values("dollar_volume", ascending=False).reset_index(drop=True)
    trichter["zulässig"] = len(agg)

    if criteria.top_n is not None and len(agg) > criteria.top_n:
        trichter[f"Rang über {criteria.top_n}"] = len(agg) - criteria.top_n
        agg = agg.head(criteria.top_n).reset_index(drop=True)

    agg["rank"] = range(1, len(agg) + 1)
    trichter["ausgewählt"] = len(agg)

    codes = [str(c) for c in agg["code"]]
    beleg = (
        _survivorship(codes, criteria.as_of, day_partitions)
        if measure_survivorship
        else Survivorship(n_selected=len(codes), n_delisted=0, lake_last=None)
    )
    return ScreenResult(
        criteria=criteria,
        window_start=tage[0],
        window_end=tage[-1],
        n_window_days=len(tage),
        rows=agg,
        funnel=trichter,
        survivorship=beleg,
    )

# ---------------------------------------------------------------------------
# Periodische Rekonstitution (#255) — die Regel über die Zeit statt zum Tag


#: Erlaubte Rekonstitutions-Frequenzen als Text: ``"6M"``, ``"12M"``, ``"1Y"``.
#: Untergrenze ist ein Monat — häufiger zu screenen misst nicht mehr Wahrheit,
#: sondern mehr Rauschen: der Liquiditätsmedian über 60 Handelstage ändert sich
#: nicht wöchentlich, die Handelskosten des Umschlags schon.
_REBALANCE_RE = re.compile(r"^\s*(\d{1,2})\s*([MY])\s*$", re.IGNORECASE)

#: Obergrenze. Wer seltener als alle zwei Jahre neu schirmt, hat einen
#: Schnappschuss mit Zwischenstopps, keine Mitgliedschaft.
MAX_REBALANCE_MONTHS = 24

#: Obergrenze für die **Zahl** der Stichtage. Der Takt allein begrenzt sie
#: nicht: `1M` über 26 Jahre sind 310 Stichtage, also 310 Fenster-Aggregate à
#: 60 Partitionen — rund 18.600 Partitionslesungen in einem HTTP-Request, den
#: kein Router-Timeout überlebt. 60 trägt 30 Jahre halbjährlich, 15 Jahre
#: quartalsweise oder 5 Jahre monatlich; darüber ist die Absage die ehrlichere
#: Antwort als ein Lauf, der nach zehn Minuten abbricht.
MAX_STICHTAGE = 60


def parse_rebalance(value: str) -> int:
    """``"6M"`` → 6, ``"1Y"`` → 12. Gibt Monate zurück.

    Eigene Funktion statt eines Enums, weil der Wert im YAML steht und dort
    lesbar bleiben soll. Ein ``rebalance: 6M`` erklärt sich; ein
    ``rebalance: SEMI_ANNUAL`` erklärt sich auch, aber nur, wenn man weiß,
    dass es die Konstante gibt.
    """
    m = _REBALANCE_RE.match(str(value or ""))
    if not m:
        raise ScreenError(
            f"'{value}' ist keine Rekonstitutions-Frequenz. Erwartet wird "
            "'<n>M' oder '<n>Y' — z.B. 6M für halbjährlich."
        )
    monate = int(m.group(1)) * (12 if m.group(2).upper() == "Y" else 1)
    if monate < 1 or monate > MAX_REBALANCE_MONTHS:
        raise ScreenError(
            f"Rekonstitution alle {monate} Monate liegt außerhalb von 1.."
            f"{MAX_REBALANCE_MONTHS}. Häufiger misst Rauschen statt Liquidität, "
            "seltener ist ein Schnappschuss mit Zwischenstopps."
        )
    return monate


def _plus_monate(d: date, monate: int) -> date:
    """Monatsarithmetik ohne dateutil — Monatsenden werden gekürzt.

    Der 31.08. plus einen Monat ist der 30.09., nicht der 01.10.

    **Immer vom Anker rechnen, nie vom Vorgänger.** Gesteppt ist die Kürzung
    keine einmalige Klammer, sondern eine Ratsche: 31.01. → 28.02. → *28.03.*
    → 28.04., und ab dem dritten Stichtag liegt die ganze Reihe dauerhaft auf
    dem 28. Der Kalender, den man liest, wäre dann ein anderer als der, den
    `rebalance: 1M` verspricht — und aus `(as_of, rebalance)` nicht mehr
    reproduzierbar. Vom Anker: 31.01. → 28.02. → 31.03. → 30.04.
    """
    m = d.month - 1 + monate
    jahr, monat = d.year + m // 12, m % 12 + 1
    return date(jahr, monat, min(d.day, monthrange(jahr, monat)[1]))


@dataclass(frozen=True)
class ReconstitutionPeriod:
    """Eine Mitgliedschaftsperiode samt ihrer Herkunft.

    ``as_of`` ist der Stichtag, an dem geschirmt wurde; ``start`` ist der Tag,
    ab dem die Auswahl gilt. Beide sind derselbe Tag — dieselbe Konvention wie
    beim Schnappschuss, dessen ``usable_window.start`` ebenfalls der Stichtag
    ist. Die Auswahl kannte nur Daten **bis** zu diesem Tag; der Signal-Lag der
    Strategie liegt darüber, nicht darunter.
    """

    as_of: date
    start: date
    end: date | None
    symbols: tuple[str, ...]
    n_added: int
    n_removed: int
    n_candidates: int
    liquidity_floor: float
    window_start: date
    window_end: date
    #: Der erste Stichtag hat keinen Vorgänger — gesetzt vom Erzeuger, damit
    #: `turnover` das nicht aus `n_added == 0` raten muss (was auch ein
    #: Stichtag ohne jede Veränderung wäre, und das ist etwas anderes).
    is_first: bool = False
    #: Der Trichter dieses Stichtags. Mitgeführt statt später nachgerechnet:
    #: „warum sind da nur drei Namen drin" ist auch bei einer Rekonstitution
    #: die erste Rückfrage, und ein zweiter Screen-Lauf für die Antwort wäre
    #: derselbe Lauf ein zweites Mal.
    funnel: dict[str, int] = field(default_factory=dict)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def turnover(self) -> float | None:
        """Anteil neuer Namen an diesem Stichtag — ``None`` für den ersten.

        Beim ersten Stichtag ist *alles* neu, und genau deshalb steht dort
        keine Zahl: eine 100 % im Median würde jede Aussage über den Umschlag
        verfälschen, ohne etwas über ihn zu sagen.
        """
        if self.is_first:
            return None
        return self.n_added / self.n_symbols if self.n_symbols else 0.0


@dataclass(frozen=True)
class Reconstitution:
    """Was die Regel über mehrere Stichtage ergab — samt Umschlag.

    Der Rückgabewert von ``reconstitute``. Er trägt die Mitgliedschaft
    (``periods``) und die Zahl, die über die Vertretbarkeit des Schnappschusses
    entscheidet (``drift``).
    """

    criteria: ScreenCriteria
    rebalance_months: int
    until: date
    periods: tuple[ReconstitutionPeriod, ...]
    survivorship: Survivorship

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    @property
    def union(self) -> list[str]:
        alle: set[str] = set()
        for p in self.periods:
            alle |= set(p.symbols)
        return sorted(alle)

    @property
    def turnovers(self) -> list[float]:
        return [p.turnover for p in self.periods if p.turnover is not None]

    @property
    def median_turnover(self) -> float:
        werte = sorted(self.turnovers)
        if not werte:
            return 0.0
        mitte = len(werte) // 2
        if len(werte) % 2:
            return werte[mitte]
        return (werte[mitte - 1] + werte[mitte]) / 2

    @property
    def max_turnover(self) -> float:
        return max(self.turnovers, default=0.0)

    @property
    def liquidity_floor(self) -> float:
        """Der **niedrigste** Boden über alle Perioden.

        Die Kostenklasse gilt für das ganze Universum über den ganzen Zeitraum,
        also bestimmt sie der schwächste Titel am schwächsten Stichtag. Den
        Mittelwert zu nehmen wäre der Fehler in die billige Richtung.
        """
        return min((p.liquidity_floor for p in self.periods), default=0.0)

    @property
    def drift(self) -> float:
        """Anteil des **ersten** Korbs, der am letzten Stichtag fehlt.

        Die Zahl, die die eigentliche Frage beantwortet: wäre ein Schnappschuss
        vertretbar gewesen? Bei 0,39 hätte ein Backtest auf der Liste vom
        ersten Stichtag am Ende zu 39 % aus Namen bestanden, die die Regel dort
        längst nicht mehr wählt — und die er trotzdem weitergehandelt hätte.
        """
        if self.n_periods < 2:
            return 0.0
        erste, letzte = set(self.periods[0].symbols), set(self.periods[-1].symbols)
        if not erste:
            return 0.0
        return 1.0 - len(erste & letzte) / len(erste)

    def churn_note(self) -> str:
        if self.n_periods < 2:
            return "Nur ein Stichtag — das ist ein Schnappschuss, kein Umschlag."
        return (
            f"{self.n_periods} Stichtage alle {self.rebalance_months} Monate. "
            f"Median-Umschlag {self.median_turnover:.1%} je Rekonstitution "
            f"(max. {self.max_turnover:.1%}). Vom Korb des "
            f"{self.periods[0].as_of} fehlen am {self.periods[-1].as_of} "
            f"{self.drift:.1%} — so falsch wäre ein Schnappschuss geworden."
        )


def reconstitute(
    criteria: ScreenCriteria,
    *,
    rebalance: str = "6M",
    until: date | None = None,
) -> Reconstitution:
    """Die Regel alle N Monate neu auswerten — zeitvariable Mitgliedschaft.

    Der Unterschied zu ``screen`` ist nicht die Menge der Läufe, sondern die
    Aussage. Ein Screen sagt: *das war das Universum am 30.06.2007*. Eine
    Rekonstitution sagt: *das war es an jedem Stichtag bis heute* — und macht
    damit sichtbar, was der Schnappschuss verschweigt, nämlich alles, was nach
    seinem Stichtag an die Börse kam.

    Jede Periode hält die Point-in-time-Garantie einzeln: ``screen`` liest zu
    keinem Stichtag über diesen hinaus. Die Reihenfolge der Stichtage ändert
    daran nichts, weil keiner vom anderen weiß.

    Parameters
    ----------
    criteria
        Die **Vorlage**. ``as_of`` ist der erste Stichtag; alle weiteren sind
        derselbe Filtersatz zu einem späteren Tag.
    rebalance
        ``"6M"``, ``"1Y"``, … siehe ``parse_rebalance``.
    until
        Letzter möglicher Stichtag. Ohne Angabe das Ende des **längsten
        zusammenhängenden** Blocks im Lake — nicht die letzte Partition
        überhaupt: das ist ein Extremum und kann eine Streu-Partition aus
        einem alten Testlauf sein.

    Raises
    ------
    ScreenError
        Wenn ein einzelner Stichtag nicht auswertbar ist. **Bewusst ein
        Abbruch, kein Überspringen:** einen fehlenden Stichtag stillschweigend
        auszulassen hieße, den vorherigen Korb weiterlaufen zu lassen — genau
        der Schnappschuss-Fehler, nur lokal und unsichtbar. Auch wenn weniger
        als zwei Stichtage herauskommen: das ist ein Schnappschuss, und dafür
        gibt es ``screen``.
    """
    monate = parse_rebalance(rebalance)
    alle_tage = storage.list_day_partitions(US_EQUITY_PREFIX)
    if not alle_tage:
        raise ScreenError(
            "Keine Querschnitte im Lake — eine Rekonstitution braucht Historie, "
            "nicht nur einen Stichtag."
        )
    # NICHT `alle_tage[-1]`: das ist ein Extremum, nicht die Front. Am
    # 2026-08-15 lag im Lake eine Streu-Partition vom Sommer 2011 aus einem
    # frühen Loader-Test; der Default hätte damit Stichtage bis 2011 erzeugt
    # und wäre im ersten Loch abgebrochen — mit einer Meldung, die den
    # Stichtag beschuldigt statt die Partition.
    block = storage.longest_run(alle_tage)
    ende = until or (block[1] if block else alle_tage[-1])

    # **Auch der Beleg rechnet nur im Block.** Sonst ankert `_survivorship`
    # sein `lake_last` auf der Streu-Partition von 2011 und hält jedes Papier
    # für tot, das dort nicht handelt — gemessen: 4 von 4 statt 1 von 4, mit
    # allen noch handelnden Namen als Beispiele. Diese Zahl landet als
    # `delisted_after_as_of` in der YAML, unter einem Kommentar, der sie
    # „GEMESSEN, nicht behauptet" nennt. Eine erfundene Messung ist teurer als
    # gar keine.
    im_block = [d for d in alle_tage if d <= ende] if block else alle_tage
    if ende < criteria.as_of:
        raise ScreenError(
            f"Das Ende {ende} liegt vor dem ersten Stichtag {criteria.as_of}."
        )

    stichtage: list[date] = []
    i = 0
    while True:
        t = _plus_monate(criteria.as_of, i * monate)
        if t > ende:
            break
        stichtage.append(t)
        i += 1
        if i > MAX_STICHTAGE:
            raise ScreenError(
                f"Mehr als {MAX_STICHTAGE} Stichtage zwischen {criteria.as_of} "
                f"und {ende} im {monate}-Monats-Raster. Jeder Stichtag ist ein "
                "eigener Screen über 60 Tagesquerschnitte — das ist kein Lauf "
                "mehr, sondern ein Batch-Job. Gröberen Takt wählen oder den "
                "Zeitraum kürzen."
            )
    if len(stichtage) < 2:
        raise ScreenError(
            f"Zwischen {criteria.as_of} und {ende} liegt nur ein Stichtag im "
            f"{monate}-Monats-Raster. Das ist ein Schnappschuss — dafür ist "
            "`screen` da, und der sagt auch dazu, dass er einer ist."
        )

    perioden: list[ReconstitutionPeriod] = []
    vorher: set[str] = set()
    for i, tag in enumerate(stichtage):
        teil = replace(criteria, as_of=tag)
        try:
            res = screen(teil, measure_survivorship=False, day_partitions=im_block)
        except ScreenError as exc:
            raise ScreenError(
                f"Stichtag {tag} ({i + 1} von {len(stichtage)}) ist nicht "
                f"auswertbar: {exc} — Rekonstitution abgebrochen. Einen "
                "Stichtag zu überspringen hieße, den Korb davor "
                "weiterzuschreiben, und das ist die Verzerrung, gegen die "
                "diese Funktion gebaut ist."
            ) from exc
        if not res.symbols:
            raise ScreenError(
                f"Stichtag {tag} wählt kein Papier aus. Der Trichter: "
                f"{res.funnel}. Eine leere Periode wäre ein Zeitraum ohne "
                "Universum."
            )
        jetzt = set(res.symbols)
        perioden.append(
            ReconstitutionPeriod(
                as_of=tag,
                start=tag,
                end=stichtage[i + 1] if i + 1 < len(stichtage) else None,
                symbols=tuple(res.symbols),
                n_added=len(jetzt - vorher) if i else 0,
                n_removed=len(vorher - jetzt) if i else 0,
                n_candidates=int(res.funnel.get("kandidaten", 0)),
                liquidity_floor=res.liquidity_floor,
                window_start=res.window_start,
                window_end=res.window_end,
                is_first=i == 0,
                funnel=dict(res.funnel),
            )
        )
        vorher = jetzt

    vereinigung = sorted({s for p in perioden for s in p.symbols})
    return Reconstitution(
        criteria=criteria,
        rebalance_months=monate,
        until=ende,
        periods=tuple(perioden),
        # Einmal über die Vereinigung, zum letzten Stichtag: „wie viele der je
        # Ausgewählten handeln am Ende nicht mehr?" Je Periode gemessen wäre
        # dieselbe Frage n-mal gestellt und n-mal teuer beantwortet.
        survivorship=_survivorship(vereinigung, stichtage[-1], im_block),
    )


def membership_from(reconstitution: Reconstitution) -> Membership:
    """Die Mitgliedschaft, die aus einer Rekonstitution folgt.

    Getrennt gehalten, weil die beiden Seiten verschiedene Leben haben: die
    Rekonstitution ist ein einmaliger Messvorgang mit Trichter und Umschlag,
    die Mitgliedschaft ist das, was danach in der YAML steht und jeden Backtest
    begrenzt.
    """
    from quantrace.membership import Membership, MembershipPeriod

    return Membership(
        periods=tuple(
            MembershipPeriod(start=p.start, end=p.end, symbols=frozenset(p.symbols))
            for p in reconstitution.periods
        ),
        frequency=f"{reconstitution.rebalance_months}M",
    )


__all__ = [
    "DEFAULT_LOOKBACK_DAYS",
    "MIN_SUPPORTED_DOLLAR_VOLUME",
    "ScreenCriteria",
    "ScreenError",
    "ScreenResult",
    "Survivorship",
    "MAX_REBALANCE_MONTHS",
    "MAX_STICHTAGE",
    "Reconstitution",
    "ReconstitutionPeriod",
    "membership_from",
    "parse_rebalance",
    "reconstitute",
    "screen",
]
