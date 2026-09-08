"""Lesepfad auf Schicht 2 — mit Adjustierung aus den eigenen Corporate Actions.

Zwei Dinge macht dieses Modul, und das zweite ist das wichtigere.

## 1 · Adjustieren aus den eigenen Daten

Die Spalte ``adjusted_close`` aus dem Bulk ist zum **Abrufzeitpunkt**
adjustiert. Sie zu benutzen hätte zwei Fehler auf einmal:

* **Look-ahead.** Der Wert für den 12.09.2008 trägt alle Splits und Dividenden
  ein, die bis zum Ladetag passiert sind — also auch die aus 2015. Ein
  Backtest über 2008 rechnet damit mit Wissen von 2026.
* **Nahtstellen.** Wer 1996–2010 heute lädt und 2011–2026 nächstes Jahr,
  klebt zwei Abschnitte mit verschiedenen Faktoren aneinander. An jedem Split
  dazwischen springt die Reihe, und der Sprung sieht aus wie eine Rendite.

Deshalb: Rohkurse aus Schicht 2, Aktionen aus ``us_equity_splits`` und
``us_equity_dividends``, und die Adjustierung passiert **beim Lesen** über
``quantrace.adjust.adjust_ohlcv``. Damit ist sie eine Funktion der Daten statt
eine Funktion des Abrufdatums.

## 2 · Nicht so tun, als wäre adjustiert, was nicht adjustiert ist

Der Lake lädt über Tage. Solange die Actions-Feeds hinterherhinken, gibt es
Fenster mit Kursen und ohne Splits. Die naheliegende Implementierung liefert
dort stillschweigend die Rohreihe — und der Aufrufer hält sie für adjustiert.

Das ist exakt das Muster, an dem sich dieses Projekt schon zweimal die Finger
verbrannt hat: `_score_realism` las fehlende Kosten als 0,0, `us_core_etfs`
versprach ein Fenster, das es nicht hatte. **Fehlend ist ein eigener Fall, kein
Nullwert.**

Also trägt jedes Ergebnis eine ``Adjustment``-Auskunft: ob adjustiert wurde,
welches Fenster die Actions abdecken, und was das für die Reihe bedeutet. Wer
sie ignoriert, tut das sichtbar.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from quantrace import storage
from quantrace.adjust import UnadjustableActionError, adjust_ohlcv
from quantrace.befunde import Befund, entscheidung_fuer, lade_entscheidungen
from quantrace.einheiten import AUSSCHLAG, faktorkurve, klassifiziere
from quantrace.instruments import US_DIVIDENDS_PREFIX, US_SPLITS_PREFIX
from quantrace.invarianten import MAX_JE_INVARIANTE
from quantrace.invarianten import pruefe as invarianten_pruefen
from quantrace.resolve import RESOLVED_PREFIX, materialised_keys

log = logging.getLogger(__name__)

#: Ab hier ist ein Actions-Read **über die Tagespartitionen** (ein GET je Tag,
#: zwei Feeds) auf Herokus 30s-Router-Timeout nicht mehr verlässlich —
#: gemessen, nicht geraten. ~8 Jahre lassen komfortabel Luft.
#:
#: **Gilt nur noch für den Partitionsweg.** Liegt eine Zusammenfassung, die das
#: Fenster abdeckt, kostet derselbe Read einen GET: nachgemessen am 2026-09-06
#: für `us_top500_liquid` über 2000–2023 — Splits 353 s → 5,8 s, Dividenden
#: 852 s → 13,0 s, und 20 Symbole über 23 Jahre voll adjustiert in 6,1 s. Dann
#: schützt die Grenze nichts mehr und kostet die Adjustierung.
_DEFAULT_MAX_ADJUST_WINDOW_DAYS = 3000


def _max_adjust_window_days() -> int:
    """Die Grenze in Tagen; ``0`` hebt sie auf.

    **Warum sie überhaupt verstellbar sein muss.** Sie schützt einen
    HTTP-Request vor einem Router-Timeout — außerhalb eines Requests schützt
    sie nichts und kostet die Adjustierung. Ein Walk-Forward über 2000–2015
    (5.511 Tage) lief am 2026-08-27 genau deshalb ins Leere: die Actions lagen
    im Lake, wurden aber nicht gelesen, und der Lauf brach ab, weil eine rohe
    Reihe für einen Backtest zu Recht abgelehnt wird. Lokal darf der Read
    Minuten dauern; es wartet niemand mit einer offenen Verbindung.

    Der Default bleibt streng: auf Heroku ist die Grenze richtig, und sie
    wegzunehmen hieße, den H12 zurückzuholen, gegen den sie gebaut wurde.
    """
    roh = os.environ.get("QUANTRACE_MAX_ADJUST_WINDOW_DAYS", "").strip()
    if not roh:
        return _DEFAULT_MAX_ADJUST_WINDOW_DAYS
    try:
        return max(int(roh), 0)
    except ValueError:
        log.warning(
            "QUANTRACE_MAX_ADJUST_WINDOW_DAYS=%r ist keine Zahl — Default %d gilt.",
            roh,
            _DEFAULT_MAX_ADJUST_WINDOW_DAYS,
        )
        return _DEFAULT_MAX_ADJUST_WINDOW_DAYS

#: EODHDs Platzhalter für „kein Kurs ermittelbar" — kein Nullwert, sondern eine
#: konkrete Zahl, die wie ein echter Kurs aussieht. Steht in **beiden**
#: Kursspalten, häufiger in ``adjusted_close`` als in ``close`` (am 2012-06-29:
#: 113 gegen 29 Zeilen, davon 93 nur dort).
EODHD_NULL_PRICE_SENTINEL = 999999.9999


def dollar_volume(frame: pd.DataFrame) -> pd.Series:
    """Tages-Dollarvolumen aus einem **rohen** Bulk-Frame.

    Die eine Stelle, an der diese Rechnung steht. ``quantrace.screen._aggregat``
    spiegelt sie in SQL, weil es über tausende Partitionen aggregiert — wer
    eine der beiden ändert, muss die andere mitziehen.

    **Warum es nicht ``close * volume`` ist.** Im EODHD-Bulk stehen Kurs und
    Volumen auf verschiedenen Zeitbasen: ``close`` ist roh und zeitgenau,
    ``volume`` ist auf die **heutige** Stückzahl split-adjustiert. Bewiesen am
    Split-Tag — AAPLs 2:1-Split am 2005-02-28 halbiert ``close``, lässt
    ``volume`` aber ohne Sprung durchlaufen. Mit ``S`` = Split-Faktor und
    ``D`` = Dividendenfaktor gilt::

        close          * volume  =  DV_wahr * S    (kaputt bei Splits)
        adjusted_close * volume  =  DV_wahr / D    (D >= 1, Untergrenze)

    Die zweite Zeile ist eine garantierte Untergrenze, geht aber dort schief,
    wo ``adjusted_close`` selbst Müll ist (Insolvenz mit gelöschtem
    Eigenkapital). Die beiden Fehlerfälle überschneiden sich nicht, deshalb das
    Minimum. Ohne den Fix kam AAPL zum 2012-06-29 auf 246 Mrd. $ Tagesumsatz
    statt 8,4. Ausführlich in ``screen._aggregat`` und #296.

    Fehlt ``adjusted_close`` ganz, gibt es **keine** Split-Information — dann
    ist ``close * volume`` die einzig verfügbare Schätzung und wird als solche
    zurückgegeben. Das ist kein stiller Rückfall auf den alten Fehler: die
    Spalte fehlt nur bei Frames, die nicht aus dem Bulk stammen.
    """
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    roh = close * volume
    if "adjusted_close" not in frame.columns:
        return roh

    adjusted = frame["adjusted_close"].astype(float)
    aus_adjustiert = adjusted * volume
    # Der Sentinel ist kein Kurs. Wo er steht, trägt die betroffene Schätzung
    # nichts bei — sonst gewänne sie das Minimum mit einer Fantasiezahl.
    aus_adjustiert = aus_adjustiert.where(
        (adjusted > 0) & (adjusted != EODHD_NULL_PRICE_SENTINEL)
    )
    roh = roh.where((close > 0) & (close != EODHD_NULL_PRICE_SENTINEL))
    return pd.concat([roh, aus_adjustiert], axis=1).min(axis=1)


@dataclass(frozen=True)
class Adjustment:
    """Was mit den Kursen passiert ist — und was nicht.

    ``status`` ist die eine Angabe, auf die es ankommt:

    ``full``
        Splits und Dividenden decken das angefragte Fenster ab. Die Reihe ist
        total-return-adjustiert.
    ``partial``
        Die Actions decken nur einen Teil ab. Vor ``covered_from`` ist die
        Reihe roh — an einem Split dort springt sie.
    ``none``
        Keine Actions im Lake. Die Reihe ist **roh**, nicht adjustiert.
    """

    status: str
    covered_from: date | None = None
    covered_to: date | None = None
    #: Das *angefragte* Fenster. Ohne diese beiden weiß `warning()` nicht,
    #: welche Seite fehlt — und nannte deshalb immer den Anfang.
    requested_from: date | None = None
    requested_to: date | None = None
    n_splits: int = 0
    n_dividends: int = 0
    #: Instrumente, deren Reihe wegen eines defekten Aktionseintrags **ganz**
    #: verworfen wurde: Code → Begründung (#312).
    #:
    #: Bis zum 2026-09-06 stand das nur in einer Log-Zeile. Wer das Ergebnis
    #: später las, sah ein Symbol, das im Universum steht und keine Kurse hat —
    #: ununterscheidbar von einem, das im Fenster nie gehandelt hat. Genau die
    #: Unterscheidung ist der Punkt: „nie geladen" ist eine Lücke im Lake,
    #: „geladen und verworfen" ein Datenfehler mit Namen und Grund.
    #: Schlüssel ist das **Instrument** (``code.WY.US.s1``), nicht der Ticker.
    unadjustable: dict[str, str] = field(default_factory=dict)

    @property
    def is_total_return(self) -> bool:
        return self.status == "full"

    def warning(self) -> str | None:
        """Ein Satz für Log, API und UI — oder ``None``, wenn alles sauber ist."""
        if self.status == "full":
            return None
        if self.status == "skipped":
            return (
                "Corporate Actions wurden NICHT GELESEN — das Fenster liegt über "
                "der Adjustierungs-Grenze, nicht etwa die Daten fehlen. Die "
                "Grenze schützt einen HTTP-Request vor dem Router-Timeout; für "
                "einen lokalen Lauf hebt QUANTRACE_MAX_ADJUST_WINDOW_DAYS=0 sie "
                "auf. Bis dahin ist die Reihe ROH."
            )
        if self.status == "none":
            return (
                "Keine Splits/Dividenden im Lake — die Reihe ist ROH, nicht "
                "adjustiert. Renditen unterschätzen den Total Return, und an "
                "jedem Split springt der Kurs."
            )
        # **Welche Seite fehlt, entscheidet die Meldung.** Bis zum 2026-08-27
        # nannte sie ausnahmslos den Anfang — auch dann, wenn der exakt
        # abgedeckt war und genau ein Tag am *Ende* fehlte, weil der Ladelauf
        # Kurse einen Tag weiter geschrieben hatte als die Actions. Wer das
        # liest, prüft die Historie von 2000 und findet dort nichts.
        vorne = (
            self.requested_from is not None
            and self.covered_from is not None
            and self.covered_from > self.requested_from
        )
        hinten = (
            self.requested_to is not None
            and self.covered_to is not None
            and self.covered_to < self.requested_to
        )
        if hinten and not vorne:
            return (
                f"Corporate Actions enden am {self.covered_to}, angefragt ist bis "
                f"{self.requested_to} — der Rest der Reihe ist roh. Meist steht der "
                "Ladelauf schlicht einen Tag weiter bei den Kursen als bei den "
                "Actions; dann genügt ein Enddatum bis zum abgedeckten Tag."
            )
        if vorne and hinten:
            return (
                f"Corporate Actions decken nur {self.covered_from} … "
                f"{self.covered_to}, angefragt ist {self.requested_from} … "
                f"{self.requested_to} — außerhalb ist die Reihe roh."
            )
        return (
            f"Corporate Actions decken erst ab {self.covered_from} — davor ist "
            "die Reihe roh. Ein Split vor diesem Datum erscheint als Rendite."
        )


def _actions_window(prefix: str) -> tuple[date | None, date | None]:
    days = storage.list_day_partitions(prefix)
    return (min(days), max(days)) if days else (None, None)


#: Der zusammengefasste Feed neben den Tagespartitionen. Ein GET statt tausender.
KONSOLIDIERT = "_konsolidiert.parquet"
#: Welche Tage darin stecken. Ohne diese Liste ist ein Tag ohne Aktion nicht von
#: einem Tag zu unterscheiden, den die Zusammenfassung nie gesehen hat — und
#: „keine Dividende" als „nicht geladen" zu lesen (oder umgekehrt) ist genau die
#: Sorte stiller Fehler, gegen die der ganze `Adjustment`-Apparat gebaut ist.
KONSOLIDIERT_TAGE = "_konsolidiert_tage.parquet"


def konsolidiert_pfade(prefix: str) -> tuple[str, str]:
    """(Daten, Tagesliste) für den zusammengefassten Feed eines Prefix."""
    return (
        storage.cache_path(f"{prefix}/{KONSOLIDIERT}"),
        storage.cache_path(f"{prefix}/{KONSOLIDIERT_TAGE}"),
    )


def schreibe_konsolidiert(prefix: str, daten: pd.DataFrame, tage: Sequence[date]) -> None:
    """Schreibt Zusammenfassung und Tagesliste. Siehe `konsolidiere_actions`."""
    pfad, tage_pfad = konsolidiert_pfade(prefix)
    storage.write_parquet(daten, pfad)
    storage.write_parquet(pd.DataFrame({"date": sorted(set(tage))}), tage_pfad)


def _konsolidierte_tage(prefix: str) -> set[date] | None:
    """Die abgedeckten Tage, oder ``None``, wenn es keine Zusammenfassung gibt."""
    pfad, tage_pfad = konsolidiert_pfade(prefix)
    if not storage.exists(pfad) or not storage.exists(tage_pfad):
        return None
    try:
        df = storage.read_parquet(tage_pfad)
    except Exception as exc:  # pragma: no cover - defekte Datei
        # Eine kaputte Zusammenfassung darf nichts kosten ausser sich selbst.
        log.warning("Zusammenfassung unter %s nicht lesbar: %s", prefix, exc)
        return None
    return {pd.Timestamp(d).date() for d in df["date"]}


def _aus_konsolidiert(
    prefix: str, codes: list[str], start: date, end: date, gebraucht: set[date]
) -> pd.DataFrame | None:
    """Aus der Zusammenfassung lesen — oder ``None``, wenn sie nicht reicht.

    **Reicht** heisst: sie deckt *jeden* Tag ab, den der Partitionslauf lesen
    würde. Eine teilweise Abdeckung stillschweigend zu benutzen hiesse,
    Aktionen zu verlieren — und eine verlorene Dividende ist eine Rendite, die
    im Backtest fehlt, ohne dass irgendwo etwas rot wird.
    """
    abgedeckt = _konsolidierte_tage(prefix)
    if abgedeckt is None or not gebraucht <= abgedeckt:
        return None
    pfad, _ = konsolidiert_pfade(prefix)
    con = storage._duckdb_conn()
    try:
        codes_ph = ",".join(["?"] * len(codes))
        return con.execute(
            f"SELECT * FROM read_parquet(?) "
            f"WHERE code IN ({codes_ph}) AND date >= ? AND date <= ?",
            [pfad, *codes, str(start), str(end)],
        ).df()
    except Exception as exc:  # pragma: no cover - defekte Datei
        log.warning("Zusammenfassung unter %s nicht lesbar: %s", prefix, exc)
        return None
    finally:
        con.close()


def _zusammenfassung_deckt(start: date, end: date) -> bool:
    """Decken **beide** Actions-Zusammenfassungen dieses Fenster ab?

    Beide, nicht eine: fehlt die Splitseite, läuft der Read wieder über die
    Tagespartitionen und dauert Minuten. Eine halbe Beschleunigung reicht
    nicht, um die Grenze aufzuheben, die genau davor schützt.

    Gefragt wird nach denselben Tagen, die `_read_actions` lesen würde — nicht
    nach dem Kalenderfenster. Ein Feed, der an einem Tag keine Partition hat,
    kann sie auch nicht in der Zusammenfassung haben.
    """
    for prefix in (US_SPLITS_PREFIX, US_DIVIDENDS_PREFIX):
        tage = storage.list_day_partitions(prefix)
        gebraucht = {d for d in tage if start <= d <= end}
        if not gebraucht:
            continue  # nichts zu lesen — kein Grund für die Bremse
        abgedeckt = _konsolidierte_tage(prefix)
        if abgedeckt is None or not gebraucht <= abgedeckt:
            return False
    return True


def _read_actions(prefix: str, codes: list[str], start: date, end: date) -> pd.DataFrame:
    """Splits oder Dividenden für die Codes im Fenster. Leer, wenn nichts liegt.

    Nur die Partitionen im angefragten Fenster. ``date=*`` über den ganzen
    Feed wären auf R2 tausend HTTP-GETs für eine Adjustierung, die nur das
    Chart-Jahr braucht — DuckDB kann Hive-Pruning erst anwenden, nachdem
    der Glob expandiert ist, und die Expansion *ist* der teure Teil.

    **Kein Wildcard je Tag.** ``scripts/load_us_equities.py`` schreibt
    Actions-Partitionen immer als ``date=…/data.parquet`` (anders als die
    Schicht-2-Instrumentpfade, wo `PARTITION_BY` mal `data_0.parquet`
    hinterlässt). Ein `*.parquet` je Tag zwingt DuckDB, für **jeden** Tag im
    Fenster erst das Verzeichnis zu LISTen, bevor es lesen kann — bei einem
    17-Jahre-Fenster (~4.300 Handelstage × zwei Feeds) war genau das der
    30-Sekunden-Timeout auf Heroku (H12), der als "Max"-Chart aufschlug. Der
    exakte Pfad braucht kein LIST, nur ein GET.
    """
    days = storage.list_day_partitions(prefix)
    if not days or not codes:
        return pd.DataFrame()
    im_fenster = [d for d in days if start <= d <= end]
    if not im_fenster:
        return pd.DataFrame()

    # **Ein GET statt tausender, wenn die Zusammenfassung reicht.** Der Aufwand
    # eines Actions-Reads ist reine Latenz: ein Zwanzig-Jahre-Fenster sind
    # ~11.600 Einzel-GETs gegen R2, jeder mit 80-150 ms. Gemessen am
    # 2026-09-06 an derselben Messung über 15 und über 1.924 Papiere — das
    # Achtzigfache an Daten kostete nur das Zweieinhalbfache an Zeit, also
    # dominiert der Fixkostenblock. Latenz faellt mit der Zahl der Anfragen,
    # nicht mit der Bandbreite; mehr Mbit/s helfen hier nicht.
    aus_zusammenfassung = _aus_konsolidiert(prefix, codes, start, end, set(im_fenster))
    if aus_zusammenfassung is not None:
        return aus_zusammenfassung

    pfade = [
        storage.cache_path(f"{prefix}/date={d.isoformat()}/data.parquet") for d in im_fenster
    ]
    con = storage._duckdb_conn()
    try:
        dateien = ",".join(["?"] * len(pfade))
        codes_ph = ",".join(["?"] * len(codes))
        sql = (
            f"SELECT * FROM read_parquet([{dateien}], union_by_name=true) "
            f"WHERE code IN ({codes_ph})"
        )
        return con.execute(sql, [*pfade, *codes]).df()
    except Exception as exc:  # pragma: no cover - defekte Partition
        # Eine kaputte Actions-Partition darf den Kurs-Read nicht killen —
        # aber sie darf auch nicht als „keine Actions" durchgehen. Der
        # Aufrufer sieht das am `status`, der dann nicht `full` wird.
        log.warning("Actions unter %s nicht lesbar: %s", prefix, exc)
        return pd.DataFrame()
    finally:
        con.close()


#: Die vier Spalten, die zusammen einen Bar ausmachen.
_OHLC_SPALTEN = ("open", "high", "low", "close")


def _nullbars_als_luecke(prices: pd.DataFrame) -> pd.DataFrame:
    """Ein Bar aus lauter Nullen ist keine Beobachtung, sondern eine Lücke.

    **Der Fall.** EODHD schreibt für manche Tage eine Zeile mit
    ``open=high=low=close=volume=0``. Am 2026-09-05 gemessen: 136 von 2.245
    Zeilen bei ``ANG``, ``DIC`` und ``IVL``. Eine Aktie hat an keinem Tag zu
    0,00 $ gehandelt — die Null ist ein Platzhalter für „nichts geliefert",
    kein Kurs.

    **Warum das nicht durchgehen darf.** Das Qualitätstor stuft
    ``nonpositive_price`` als *Warnung* ein, also kommen die Nullen bis in den
    Backtest. Dort brach vectorbt den Lauf ab —
    ``order.price must be finite and greater than 0`` —, und zwar Minuten nach
    einem Load, der eine halbe Stunde gebraucht hat. Schlimmer wäre der Fall,
    in dem er *nicht* abbricht: eine Rendite von −100 % auf dem Nullbar und
    +∞ am Tag danach ist keine Kennzahl, sondern Rauschen mit Vorzeichen.

    **Warum Zeile löschen und nicht Kurs raten.** Die Zeile zu behalten und
    fortzuschreiben hiesse, einen Handel zu behaupten, den es nicht gab. Der
    Lake stellt „kein Handel" ohnehin als fehlende Zeile dar; eine Null-Zeile
    ist dieselbe Aussage in schlechterer Schreibweise, und sie wird hier in die
    richtige übersetzt. Was danach passiert, ist gelöst und getestet: die
    Qualitätsprüfung meldet die Lücke, und
    ``backtest_runner._close_untradable`` unterscheidet sie vom Ende eines
    Papiers.

    Ein *einzelner* Nullwert neben positiven bleibt stehen — das ist ein
    anderer Defekt, und die Warnung dafür soll ihn weiter treffen.
    """
    vorhanden = [c for c in _OHLC_SPALTEN if c in prices.columns]
    if not vorhanden:
        return prices
    leer = (prices[vorhanden] <= 0).all(axis=1)
    if not bool(leer.any()):
        return prices

    if "code" in prices.columns:
        je_code = prices.loc[leer, "code"].value_counts()
        details = ", ".join(f"{c}: {n}" for c, n in list(je_code.items())[:8])
    else:
        details = f"{int(leer.sum())} Zeilen"
    log.warning(
        "%d Bars aus lauter Nullen als Lücke gelesen statt als Kurs (%s). "
        "Eine Aktie handelt an keinem Tag zu 0,00 — die Null ist ein "
        "Platzhalter des Feeds.",
        int(leer.sum()),
        details,
    )
    return prices.loc[~leer].reset_index(drop=True)


def _fortgeschriebene_als_luecke(prices: pd.DataFrame) -> pd.DataFrame:
    """Ein Tag ohne Umsatz hat keinen Kurs — der Feed schreibt den letzten fort.

    **Der Fall.** ``LDG`` steht nach dem Delisting von Longs Drug Stores im
    Oktober 2008 **sieben Jahre** auf 71,51::

        2008-10-29    71.51   Volumen 0      letzter echter Handelstag
        2008-10-30    71.51   Volumen 0   <- ab hier 1.700 Tage identisch
           …          71.51   Volumen 0
        2015-08-12  8043.67              <- hier erst kommt ein anderes Papier

    Jede Zeile ist für sich plausibel, die Nachbarn widersprechen sich nicht,
    der Aktionsfaktor steht still — **alle bestehenden Prüfungen sind dort
    blind.** Für einen Backtest sind das sieben Jahre mit Rendite null und
    Volatilität null, und das verzerrt jede Kennzahl: ein Sharpe braucht eine
    Streuung im Nenner, ein Drawdown eine Bewegung.

    Es ist dieselbe Familie wie die Nullbars (#322), nur mit einer plausiblen
    Zahl statt einer Null — und dieselbe Antwort: **der Lake stellt „kein
    Handel" als fehlende Zeile dar; hier wird die schlechtere Schreibweise in
    die richtige übersetzt.**

    **Die Bedingung ist nicht die Länge, sondern das Volumen.** Über
    `us_top500_liquid` am 2026-09-08 gemessen, Strecken mit identischem Kurs::

        5-19 Tage    40,7 % der Tage mit Umsatz   -> echter Handel
        20-99        13,7 %
        100+          3,9 %                        -> praktisch keiner

    Eine Regel über die Streckenlänge hätte keine Schwelle: die Verteilung
    läuft stetig durch, jede Grenze wäre gesetzt statt gemessen. Der Umsatz
    dagegen trennt sauber — ``PTT`` steht 2.467 Tage auf 5,12 mit **null**
    Umsatz, ``UDS`` 263 Tage mit 30 % Handelstagen und ist echt.

    **Aber `volume = 0` heisst nicht überall „kein Handel".** Bei 25,8 % aller
    Nulltage bewegt sich der Kurs — dort fehlt schlicht die Angabe::

        AYE   3.945 Nulltage von 4.276 Zeilen (92 %), davon 3.224 MIT Bewegung
        UPR   2.934 von 3.132 (94 %),  590 mit Bewegung

    Deshalb entscheidet die Reihe über sich selbst: **das Volumen eines
    Papiers zählt nur, wenn es kein einziges Mal einer Kursbewegung
    widerspricht.** Ein Feld, das auch nur einmal lügt, taugt nicht als
    Nachweis — dann weiss man nicht, ob es ausgerechnet an *diesem* Tag lügt.
    Das trifft 290 der 459 Papiere mit Nulltagen; die übrigen bleiben
    unangetastet und werden von der `eingefroren`-Invariante gemeldet, statt
    still behandelt zu werden.

    Kein Schwellenwert, keine Toleranz, keine gesetzte Zahl.
    """
    noetig = {"instrument", "close", "volume"}
    if prices.empty or not noetig <= set(prices.columns):
        return prices

    raus = pd.Series(False, index=prices.index)
    je_instrument: dict[str, int] = {}
    for instrument, teil in prices.groupby("instrument", sort=False):
        teil = teil.sort_values("date")
        close = pd.to_numeric(teil["close"], errors="coerce")
        vol = pd.to_numeric(teil["volume"], errors="coerce")
        vorher = close.shift(1)
        # `vol == 0` und nicht `<= 0`: ein fehlender Wert (NaN) ist keine
        # Aussage über Handel, und `NaN <= 0` wäre ohnehin False.
        ohne_umsatz = (vol == 0) & vorher.notna() & close.notna()
        if bool((ohne_umsatz & (close != vorher)).any()):
            continue  # Das Volumen dieser Reihe trägt nichts — Finger weg.
        tot = ohne_umsatz & (close == vorher)
        if bool(tot.any()):
            raus.loc[teil.index[tot]] = True
            je_instrument[str(instrument)] = int(tot.sum())

    if not je_instrument:
        return prices

    log.warning(
        "%d Instrument(e) mit fortgeschriebenen Kursen — %d Zeilen ohne Umsatz "
        "und ohne Kursänderung als Lücke gelesen statt als Kurs (#333): %s. "
        "Wo nicht gehandelt wurde, gibt es keinen Kurs; der Feed wiederholt "
        "den letzten.",
        len(je_instrument),
        sum(je_instrument.values()),
        ", ".join(f"{i}: {n}" for i, n in sorted(je_instrument.items())[:10]),
    )
    return prices.loc[~raus].reset_index(drop=True)


def read_instruments(
    instruments: list[str],
    start: date,
    end: date,
    *,
    adjust: bool = True,
) -> tuple[pd.DataFrame, Adjustment]:
    """Long-OHLCV aus Schicht 2, optional adjustiert.

    Returns
    -------
    (frame, adjustment)
        ``frame`` im Long-Format (``instrument``, ``code``, ``date``, OHLCV).
        ``adjustment`` sagt, ob und wie weit adjustiert wurde — **immer
        prüfen**, bevor die Zahlen irgendwo landen.
    """
    # Ein LIST-Aufruf gegen den Prefix statt eines `exists` je Instrument.
    # Der Dateiname innerhalb der Partition steht nicht fest — DuckDBs
    # PARTITION_BY schreibt `data_0.parquet`, ein Einzel-Write `data.parquet`
    # —, deshalb wird pro Instrument geglobt statt geraten.
    vorhandene = materialised_keys()
    vorhanden = [i for i in instruments if i in vorhandene]
    if not vorhanden:
        return pd.DataFrame(), Adjustment(status="none")

    pfade = [
        storage.cache_path(f"{RESOLVED_PREFIX}/instrument={i}/*.parquet") for i in vorhanden
    ]
    con = storage._duckdb_conn()
    try:
        platzhalter = ",".join(["?"] * len(pfade))
        sql = (
            f"SELECT * FROM read_parquet([{platzhalter}], union_by_name=true) "
            "WHERE date >= ? AND date <= ? ORDER BY instrument, date"
        )
        prices = con.execute(sql, [*pfade, str(start), str(end)]).df()
    finally:
        con.close()

    if prices.empty:
        return prices, Adjustment(status="none")

    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = _nullbars_als_luecke(prices)
    # **Nach den Nullbars.** Eine Nullzeile ist keine Kursänderung, sondern
    # gar kein Kurs — stünde sie noch drin, sähe sie hier wie ein
    # Widerspruch aus und schaltete die Regel für die ganze Reihe ab.
    prices = _fortgeschriebene_als_luecke(prices)

    if not adjust:
        return prices, Adjustment(status="none")

    # **Die Grenze gilt nur, wo sie noch etwas schützt.** Sie war gegen den
    # Partitionsweg gebaut (~11.600 GETs für zwanzig Jahre); mit einer
    # Zusammenfassung, die das Fenster abdeckt, ist derselbe Read ein GET.
    # Sie stehen zu lassen hiesse, für einen Timeout zu bezahlen, den es nicht
    # mehr gibt — und der Preis sind **rohe** Kurse, auf denen kein Backtest
    # laufen darf.
    grenze = 0 if _zusammenfassung_deckt(start, end) else _max_adjust_window_days()
    if grenze and (end - start).days > grenze:
        # Ein Actions-Read ist ein GET je Tagespartition (~80-150ms, R2 kennt
        # keine größere Einheit). Gemessen: 3.452 Tage (17 Jahre AAPL) brauchen
        # ~23s bei 64 Threads — für EINEN Feed. Beide Feeds seriell hätten
        # Herokus 30s-Router-Timeout gerissen (H12), genau das Symptom, das
        # den „Max"-Chart auf ein leeres 503 reduzierte. Lieber ROHE Kurse
        # ausliefern als gar keine: die Reihe ist wahrheitsgemäß `status=none`,
        # nicht `full` erschwindelt.
        log.warning(
            "Fenster %s…%s (%d Tage) über der Adjustierungs-Grenze (%d) — "
            "Kurse bleiben ROH. Aufheben mit QUANTRACE_MAX_ADJUST_WINDOW_DAYS=0.",
            start,
            end,
            (end - start).days,
            grenze,
        )
        # **Nicht `none`.** Der Unterschied ist der zwischen „liegt nicht im
        # Lake" und „wurde nicht gelesen" — die alte Meldung behauptete das
        # Erste, während 3.795 Tage Actions danebenlagen. Wer das liest, sucht
        # einen Ladelauf statt eine Konfigurationszeile.
        return prices, Adjustment(status="skipped")

    codes = sorted({str(c) for c in prices["code"].dropna().unique()})
    # Beide Feeds parallel statt seriell: zwei unabhängige R2-Reads, die sich
    # nicht in die Quere kommen — seriell war das genau die Verdopplung, die
    # ein grenzwertiges Fenster über die 30s kippte.
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_splits = pool.submit(_read_actions, US_SPLITS_PREFIX, codes, start, end)
        f_divs = pool.submit(_read_actions, US_DIVIDENDS_PREFIX, codes, start, end)
        splits = f_splits.result()
        divs = f_divs.result()

    s_von, s_bis = _actions_window(US_SPLITS_PREFIX)
    d_von, d_bis = _actions_window(US_DIVIDENDS_PREFIX)

    if s_von is None and d_von is None:
        # Kein einziger Actions-Tag im Lake. Roh zurückgeben — und es sagen.
        info = Adjustment(status="none")
        log.warning("%s", info.warning())
        return prices, info

    von = max([d for d in (s_von, d_von) if d is not None])
    bis = min([d for d in (s_bis, d_bis) if d is not None])
    voll = von <= start and bis >= end

    # **Erst adjustieren, dann die Auskunft bauen.** `Adjustment` ist frozen,
    # und das soll es bleiben: eine Ehrlichkeitsauskunft, die der Aufrufer
    # nachträglich umschreiben kann, ist keine. Also muss der Ausschluss
    # vorliegen, bevor das Objekt entsteht.
    angewandt, unadjustierbar = _apply_actions(prices, splits, divs)

    info = Adjustment(
        status="full" if voll else "partial",
        covered_from=von,
        covered_to=bis,
        requested_from=start,
        requested_to=end,
        n_splits=int(len(splits)),
        n_dividends=int(len(divs)),
        unadjustable=unadjustierbar,
    )
    if not voll:
        log.warning("%s", info.warning())

    return angewandt, info


#: Die Befunde des letzten `_apply_actions`-Laufs. Bewusst ein Modulzustand
#: und kein Rückgabewert: `read_instruments` hat einen festen Vertrag
#: ``(frame, Adjustment)``, den ein Scan-Skript nicht ändern soll — und
#: `Adjustment` ist frozen, damit niemand die Ehrlichkeitsauskunft
#: nachträglich umschreibt. Wer die Befunde will, holt sie hier ab.
_letzte_befunde: list[Befund] = []


def letzte_befunde() -> list[Befund]:
    """Was der letzte Lesevorgang gemeldet hat. Siehe `_letzte_befunde`."""
    return list(_letzte_befunde)


def _fuer_den_bericht(befunde: list[Befund]) -> list[Befund]:
    """Je Art hoechstens `MAX_JE_INVARIANTE` Befunde — der Deckel fuers Melden.

    Er sass frueher in der Invariante selbst. Seit dem 2026-09-08 findet die
    Invariante alles, weil eine Entscheidung ueber einen Abschnitt *jede*
    betroffene Zeile treffen muss; gekuerzt wird erst hier, wo es um
    Lesbarkeit geht. `DIC` hat 1.666 kaputte Tage, und die Gesamtzahl steht
    in `n_zeilen`.
    """
    gezaehlt: dict[str, int] = {}
    aus: list[Befund] = []
    for b in befunde:
        n = gezaehlt.get(b.art, 0)
        if n >= MAX_JE_INVARIANTE:
            continue
        gezaehlt[b.art] = n + 1
        aus.append(b)
    return aus


def _apply_actions(
    prices: pd.DataFrame, splits: pd.DataFrame, divs: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Baut je Instrument ``divCash``/``splitFactor`` und adjustiert **die Kurse**.

    Gibt neben dem Frame die verworfenen Instrumente zurück (Code → Grund) —
    siehe ``Adjustment.unadjustable``.

    Die beiden Spaltennamen sind Tiingo-Vokabular — bewusst übernommen, damit
    ``adjust.adjust_ohlcv`` unverändert weiterbenutzt wird. Eine zweite
    Adjustierungs-Implementierung wäre eine zweite Wahrheit.

    **``volume`` bleibt, wie es aus dem Lake kommt** — es steht dort schon auf
    heutiger Stückzahl und braucht keine Adjustierung mehr (#304). Begründung
    an der Stelle, wo der Frame gebaut wird.
    """
    # Lazy: `providers.eodhd` zieht auf Modulebene `httpx` herein. Ein
    # Lesepfad, der nur einen Bruch parst, soll keinen HTTP-Client brauchen —
    # dieselbe Funktion bleibt trotzdem die einzige Implementierung.
    from quantrace.providers.eodhd import parse_split_ratio

    split_map: dict[tuple[str, date], float] = {}
    if not splits.empty and "split" in splits.columns:
        for row in splits.itertuples(index=False):
            faktor = parse_split_ratio(getattr(row, "split", None))
            if faktor is None or faktor <= 0:
                continue
            tag = pd.to_datetime(row.date).date()
            split_map[(str(row.code), tag)] = faktor

    div_map: dict[tuple[str, date], float] = {}
    if not divs.empty and "dividend" in divs.columns:
        for row in divs.itertuples(index=False):
            betrag = getattr(row, "dividend", None)
            if betrag is None or pd.isna(betrag):
                continue
            tag = pd.to_datetime(row.date).date()
            div_map[(str(row.code), tag)] = float(betrag)

    teile: list[pd.DataFrame] = []
    unadjustierbar: dict[str, str] = {}
    #: Zeilen, die auf falscher Stückzahl standen: Instrument → Anzahl (#324).
    #: **Nur die Automatik** — was ein Mensch im Register entschieden hat,
    #: steht daneben. Eine Meldung, die beides zusammenwirft, schickt die
    #: Suche zum falschen Detektor.
    einheiten_raus: dict[str, int] = {}
    #: Zeilen, die eine Entscheidung aus `data/corrections/` verworfen hat:
    #: Instrument → Anzahl (#333). Kein Fehler, sondern ein Urteil — deshalb
    #: eine eigene Meldung mit eigenem Ton.
    entschieden_raus: dict[str, int] = {}
    #: Alles, was auffiel — ohne Urteil darüber, was es bedeutet (#327).
    #: Der Lesepfad **meldet**; entschieden wird in `data/corrections/`.
    befunde: list[Befund] = []

    # Die Entscheidungen werden einmal gelesen, nicht je Instrument: es sind
    # Dutzende Einträge, und ein YAML-Read je Papier wäre bei 1.934 Papieren
    # der teuerste Teil dieser Funktion.
    entscheidungen = lade_entscheidungen()
    for instrument, teil in prices.groupby("instrument", sort=True):
        teil = teil.sort_values("date").reset_index(drop=True)
        code = str(teil["code"].iloc[0])

        # **Zeilen auf falscher Stückzahl fliegen raus, bevor adjustiert
        # wird** (#324). `AAPL` fällt am 2003-01-07 um 98,2 % und steigt am
        # 2003-01-10 um 5.453 % — beides hat nie stattgefunden. Die Zeile ist
        # in sich plausibel (high > low, Kurse positiv, keine Nullen), falsch
        # ist nur ihr Verhältnis zu den Nachbarn.
        #
        # Verworfen und nicht zurückgerechnet: dieselbe Begründung wie bei den
        # Nullbars (#322). Der Lake stellt „kein Handel" als fehlende Zeile
        # dar; eine Zeile in falschen Einheiten ist dieselbe Nicht-Beobachtung
        # in schlechterer Schreibweise, und sie mit dem vermuteten Faktor
        # zurückzurechnen hiesse, einen Handel zu behaupten, den niemand
        # gesehen hat.
        #
        # Nur `ausschlag`, nicht `stufe`: eine Stufe heisst, dass im Feed eine
        # Aktion fehlt — die Rohzeilen sind dort richtig, und sie zu verwerfen
        # kostete halbe Historien.
        if "adjusted_close" in teil.columns:
            close_roh = teil["close"].astype(float)
            kurve = faktorkurve(teil["adjusted_close"], close_roh)
            if bool(kurve.notna().any()):
                aktion = pd.Series(
                    [(code, d) in split_map or (code, d) in div_map for d in teil["date"]],
                    index=teil.index,
                )
                k = klassifiziere(kurve, aktion, close_roh)
                # Die Invariantenliste — Felder gegeneinander statt jede
                # Zeile für sich (#333). Sie läuft **vor** der Klassifikation
                # und unabhängig davon: `DIC` trägt denselben Fehler in beiden
                # Kursspalten, also bewegt sich die Faktorkurve nie und der
                # Einheiten-Detektor ist dort blind.
                # **Ungekürzt nur, wo das Register etwas sagt.** Eine
                # Entscheidung über einen Abschnitt muss *jede* betroffene
                # Zeile treffen — `DIC` hat 1.666, gemeldet werden fünf. Wo
                # nichts entschieden ist, bleibt es bei den fünf: alles
                # andere wären Objekte für einen Bericht, der sie kürzt.
                entschieden_hier = any(e.code == code for e in entscheidungen.values())
                inv_befunde = invarianten_pruefen(
                    teil, str(instrument), code,
                    je_invariante=None if entschieden_hier else MAX_JE_INVARIANTE,
                )
                befunde.extend(_fuer_den_bericht(inv_befunde))

                # **Melden, bevor gehandelt wird.** Jede Auffälligkeit geht
                # ins Register — auch die, die der Lesepfad selbst behandelt.
                # Wer später fragt „warum fehlt dort eine Zeile", findet die
                # Antwort im Bericht statt im Log von damals.
                for pos in range(len(k)):
                    art = k["art"].iloc[pos]
                    if not art:
                        continue
                    if pos and k["art"].iloc[pos - 1] == art:
                        continue  # nur der Beginn eines Abschnitts
                    befunde.append(
                        Befund(
                            instrument=str(instrument),
                            code=code,
                            tag=teil["date"].iloc[pos],
                            art=f"einheiten_{art}",
                            belege={
                                "f_vorher": float(kurve.iloc[pos - 1]) if pos else None,
                                "f": float(kurve.iloc[pos]),
                                "close_vorher": float(close_roh.iloc[pos - 1])
                                if pos
                                else None,
                                "close": float(close_roh.iloc[pos]),
                                "n_zeilen": int((k["art"] == art).sum()),
                            },
                        )
                    )

                # **Entschiedenes hat Vorrang vor Automatik.** Ein Eintrag in
                # `data/corrections/` ist von jemandem angesehen worden; die
                # Klassifikation ist es nicht.
                schlecht = (k["art"] == AUSSCHLAG).to_numpy()
                # **Woher eine verworfene Zeile kommt, gehört in die Meldung.**
                # Sonst steht dort „auf falscher Stückzahl (#324)" über Zeilen,
                # die ein Mensch im Register entschieden hat — und die Suche
                # beginnt beim falschen Detektor. Dieselbe Falle wie beim
                # `DENIED: denied` des Loaders (PLAN.md §9.4).
                aus_register: set[int] = set()
                for pos, tag in enumerate(teil["date"]):
                    e = entscheidungen.get(f"{code}@{tag.isoformat()}")
                    if e is None:
                        continue
                    if e.was == "verwerfen":
                        schlecht[pos] = True
                        aus_register.add(pos)
                    elif e.was == "aktion" and e.faktor:
                        split_map[(code, tag)] = (
                            split_map.get((code, tag), 1.0) * e.faktor
                        )
                        schlecht[pos] = False
                    elif e.was == "akzeptieren":
                        schlecht[pos] = False

                # **Was ein Abschnitt im Register über Invariantenbefunde
                # sagt** (#333). Der Weg darüber trifft nur, was der
                # Einheiten-Detektor findet; `kursniveau` ist dort blind, weil
                # `DIC` denselben Fehler in beiden Kursspalten trägt.
                #
                # Nur `verwerfen` wirkt hier. `akzeptieren` heisst
                # ausdrücklich „stehenlassen", und `aktion` ergibt bei einer
                # Reihe mit zwei Niveaus keinen Sinn — dort fehlt kein Split,
                # dort stehen zwei Papiere in einer Spalte. Wer das einträgt,
                # bekommt keine stille Sonderbehandlung, sondern nichts.
                if entschieden_hier:
                    pos_von_tag = {t: i for i, t in enumerate(teil["date"])}
                    for b in inv_befunde:
                        e = entscheidung_fuer(b, entscheidungen)
                        if e is None or e.was != "verwerfen":
                            continue
                        pos = pos_von_tag.get(b.tag)
                        if pos is not None:
                            schlecht[pos] = True
                            aus_register.add(pos)

                if schlecht.any():
                    entschieden = sum(1 for p in aus_register if schlecht[p])
                    if entschieden:
                        entschieden_raus[str(instrument)] = entschieden
                    if int(schlecht.sum()) - entschieden:
                        einheiten_raus[str(instrument)] = (
                            int(schlecht.sum()) - entschieden
                        )
                    teil = teil.loc[~schlecht].reset_index(drop=True)
                    if teil.empty:
                        continue
        idx = pd.DatetimeIndex(pd.to_datetime(teil["date"]))
        # `.to_numpy()` ist hier Pflicht, nicht Stil: mit einem expliziten
        # `index` reindiziert pandas übergebene Series auf diesen Index — die
        # Positionen 0..n-1 träfen auf Zeitstempel, und der ganze Frame käme
        # als NaN zurück. Roh-Arrays haben keinen Index, der kollidieren kann.
        # **`volume` wird `adjust_ohlcv` gar nicht erst gezeigt** (#304). Die
        # Funktion skaliert es mit dem Split-Faktor, weil sie für Tiingo gebaut
        # wurde: dort kam `volume` roh herein. Im EODHD-Bulk steht es bereits
        # auf heutiger Stückzahl — bewiesen am Split-Tag, siehe `dollar_volume`
        # weiter oben. Ein zweites Skalieren machte daraus `V · S²`: für AAPL
        # vor 2005 das 784-fache der gehandelten Stücke statt des 28-fachen.
        #
        # Weggelassen statt hinterher überschrieben: was hier nicht ankommt,
        # kann auch niemand versehentlich wieder aus `adj` zurückkopieren.
        # `adjust_ohlcv` rührt fehlende Spalten nicht an.
        #
        # Damit ist das Ergebnis in sich stimmig — `adjusted_close · volume`
        # ergibt wieder ein echtes Dollarvolumen, weil beide Seiten auf
        # derselben Stückzahl stehen.
        roh = pd.DataFrame(
            {
                "open": teil["open"].astype(float).to_numpy(),
                "high": teil["high"].astype(float).to_numpy(),
                "low": teil["low"].astype(float).to_numpy(),
                "close": teil["close"].astype(float).to_numpy(),
                "splitFactor": [split_map.get((code, d), 1.0) for d in teil["date"]],
                "divCash": [div_map.get((code, d), 0.0) for d in teil["date"]],
            },
            index=idx,
        )
        try:
            adj = adjust_ohlcv(roh)
        except UnadjustableActionError as exc:
            # **Ein defekter Aktionseintrag darf nicht das ganze Universum
            # kippen** — dasselbe Muster wie bei mehrdeutigen Kürzeln (#311).
            # Das Instrument fällt heraus und wird benannt; wer es braucht,
            # sieht warum. Verworfen wird die *Reihe*, nicht bloss die Aktion:
            # eine halb adjustierte Reihe wäre wieder etwas, das aussieht wie
            # ein Total Return und keiner ist.
            log.warning("%s: nicht adjustierbar, fällt heraus — %s", code, exc)
            # **Schlüssel ist das Instrument, nicht der Code.** Ein Ticker
            # kann im Lake mehrere Segmente haben (`code.WY.US.s1`, `…s2`),
            # und verworfen wird immer genau eines. Der Code steht in der
            # Begründung, damit die Meldung trotzdem lesbar bleibt.
            unadjustierbar[str(instrument)] = f"unadjustable_action [{code}]: {exc}"
            continue
        adj = adj.reset_index(drop=True)
        neu = teil.copy()
        for col in ("open", "high", "low", "close"):
            if col in adj.columns:
                neu[col] = adj[col].to_numpy()
        teile.append(neu)

    if einheiten_raus:
        log.warning(
            "%d Instrument(e) mit Zeilen auf falscher Stückzahl — %d Zeilen "
            "als Lücke gelesen statt als Kurs (#324): %s",
            len(einheiten_raus),
            sum(einheiten_raus.values()),
            ", ".join(f"{i}: {n}" for i, n in sorted(einheiten_raus.items())[:10]),
        )
    if entschieden_raus:
        # `info`, nicht `warning`: hier ist nichts schiefgegangen. Jemand hat
        # hingesehen und entschieden, und das Ergebnis ist die gewollte Lücke.
        log.info(
            "%d Instrument(e) mit entschiedenen Zeilen — %d Zeilen nach "
            "`data/corrections/` als Lücke gelesen (#333): %s",
            len(entschieden_raus),
            sum(entschieden_raus.values()),
            ", ".join(f"{i}: {n}" for i, n in sorted(entschieden_raus.items())[:10]),
        )
    if unadjustierbar:
        log.warning(
            "%d Instrument(e) ohne adjustierbare Reihe: %s",
            len(unadjustierbar),
            ", ".join(sorted(unadjustierbar)[:10]),
        )
    frame = pd.concat(teile, ignore_index=True) if teile else prices.iloc[0:0]
    _letzte_befunde.clear()
    _letzte_befunde.extend(befunde)
    return frame, unadjustierbar


__all__ = ["Adjustment", "read_instruments"]
