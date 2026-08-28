"""Schicht 2 — aus Tagesquerschnitten werden Zeitreihen mit Identität.

Schicht 1 (``us_equities/date=…``) hält fest, was EODHD an einem Tag geliefert
hat: ein Querschnitt mit ``code``, sonst nichts. Für einen Backtest braucht man
die andere Achse — **eine Reihe pro Wertpapier über die Zeit**. Der Schritt
dazwischen ist keine Umsortierung, sondern eine Behauptung: *diese Zeile vom
2003 und jene von 2024 gehören demselben Papier.*

Genau diese Behauptung geht schief, wenn man sie dem Ticker überlässt::

    tot     BBBY_old   Bed Bath & Beyond Inc     ISIN US0758961009
    tot     BBBYQ      Bed Bath & Beyond Inc.    ISIN US0758961009
    aktiv   BBBY       Bed Bath & Beyond, Inc.   ISIN US6903701018   ← Overstock

Overstock kaufte die Marke aus der Insolvenzmasse. Wer `code == "BBBY"` über
die ganze Historie verkettet, ersetzt einen Totalverlust durch den Kursverlauf
einer fremden Firma — und zwar lückenlos, ohne Fehlermeldung.

## Die zwei Regeln, die dieses Modul durchsetzt

**1 · Ein Code wird an langen Lücken zerschnitten.** Verschwindet ein Code für
länger als ``gap_days`` und taucht wieder auf, sind das zwei Segmente und damit
zwei Instrumente. Die Schwelle ist bewusst grob: ein fälschlich *getrenntes*
Papier ist eine fragmentierte Reihe, die man sieht — ein fälschlich
*verschmolzenes* ist eine falsche Historie, die man nicht sieht. Nur einer der
beiden Fehler ist teuer, also fällt die Voreinstellung auf die sichere Seite.

**2 · Eine heute erhobene ISIN gilt nur für ein heute noch lebendes Segment.**
Die Karte in ``data/instruments.yaml`` ordnet ``BBBY`` der ISIN von *Overstock*
zu — das ist korrekt für heute und grundfalsch für 2019. Deshalb bekommt ein
Segment die ISIN nur, wenn es bis ans Ende des geladenen Fensters reicht. Jedes
frühere Segment ist historisch und bekommt einen synthetischen Schlüssel, der
sich nicht als ISIN ausgibt.

Was dabei **nicht** passiert: aus einem fehlenden ISIN-Eintrag wird nie
stillschweigend „dann eben der Ticker". Ein synthetischer Schlüssel ist als
solcher markiert (``identity_source``), damit ein Lauf darauf eine bewusste
Entscheidung bleibt und keine Verwechslung.

## Schlüsselformat

``isin.US0378331005``    — belegt, prüfsummengeprüft, Segment noch aktiv
``code.AAPL.US.s1``     — synthetisch: Code, Börse, laufendes Segment

Beide sind pfadsicher (nur ``[A-Za-z0-9._-]``) und tragen ihre Herkunft im
Namen. Wer in einem Backtest-Ergebnis ``code.…`` liest, weiß ohne Nachschlagen,
dass die Identität dieser Reihe eine Konstruktion ist.

Der Bau läuft **ohne EODHD-Netz** — nur Lake-Read und Lake-Write. Das ist der
Grund, warum Schicht 1 roh bleibt: eine korrigierte Zuordnungsregel kostet
einen Rebuild, keinen zweiten Download.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from quantrace import storage
from quantrace.instruments import (
    US_EQUITY_PREFIX,
    isin_checksum_ok,
    load_instrument_map,
    normalise_isin,
)

log = logging.getLogger(__name__)

#: Wohin Schicht 2 schreibt. Bewusst ein eigener Prefix und nicht ``raw/``:
#: das Hive-Schema ist ein anderes, und ein gemischtes Verzeichnis wäre für
#: DuckDB ein Schema-Konflikt beim ersten Glob.
RESOLVED_PREFIX = "us_equities_resolved"

#: Die Instrumenten-Übersicht der Schicht. Eine Datei, damit Coverage und
#: Auswahl **einen** Read kosten und nicht einen pro Instrument.
MANIFEST_PATH = f"{RESOLVED_PREFIX}/_manifest.parquet"

#: Ab wann eine Lücke als Besitzerwechsel gilt — in **Lake-Handelstagen**,
#: nicht in Kalendertagen.
#:
#: Ein Jahr ist grob und soll es sein. Handelsaussetzungen über mehr als ein
#: Jahr enden fast immer im Delisting; Ticker werden nach einer Pleite typisch
#: binnen Monaten neu vergeben. Wer die Schwelle senkt, zerlegt mehr echte
#: Reihen (sichtbarer Schaden); wer sie hebt, verkettet mehr fremde
#: (unsichtbarer Schaden).
#:
#: **Warum Handelstage und nicht Kalendertage — der Fehler vom 2026-08-12.**
#: Der Lake lädt die Historie in Blöcken, und dazwischen liegen Löcher. Am
#: 2026-08-12 sah er so aus::
#:
#:     1996-01-02 … 1999-09-29     ← Block aus dem ersten Ladelauf
#:     2000-01-03 … 2001-09-17     ← Block aus dem laufenden
#:     2011-07-01 … 2011-08-25     ← Streu-Partitionen aus einem Loader-Test
#:
#: Zwischen 2001 und 2011 liegen 3.574 Kalendertage — und **keine einzige
#: Partition**. In Kalendertagen gemessen sah damit *jeder* Code, der in beiden
#: Blöcken vorkommt, wie ein Besitzerwechsel aus: 16.396 von 58.607 Segmenten
#: wurden als „abgelöst" markiert, also 28 %. Real sind das Datenlücken, keine
#: Übernahmen.
#:
#: Die Folgen wären still gewesen: fragmentierte Instrument-Schlüssel, und
#: `resolve_cik` hätte für 16.396 Codes `recycled` gemeldet — also keine
#: Fundamentaldaten, mit einer Begründung, die nicht stimmt.
#:
#: In Lake-Handelstagen gemessen verschwindet das Problem: wo keine Partition
#: liegt, hat auch niemand gehandelt, und die Abwesenheit eines Codes sagt
#: nichts über ihn aus. Die Lücke zählt nur, wenn der Code fehlte, **während
#: andere handelten**.
DEFAULT_GAP_TRADING_DAYS = 252

#: Wie nah ein Segment ans Ende des geladenen Fensters reichen muss, um als
#: „lebt heute noch" zu gelten — und damit eine ISIN aus der Karte zu bekommen.
#: Grosszügig, weil der Lake beim Laden hinterherhinkt und ein Papier nicht
#: deshalb historisch wird, weil der Bulk-Load drei Wochen alt ist.
DEFAULT_ACTIVE_TOLERANCE_DAYS = 45

#: Wie viele Bars ein **beendetes** späteres Segment mindestens tragen muss,
#: damit es das frühere ablöst. Segmente, die bis an die Front reichen, sind
#: ausgenommen — dort ist die Frage nicht entscheidbar (siehe unten).
#:
#: **Der Fall.** Am 2026-08-27 im Lake gemessen: von 4.457 Ablösungen hingen
#: **304 an einem späteren Segment mit genau einem Bar**. `ICTXX` und `CAGXX`
#: etwa je einer am 2012-08-01 — derselbe Tag, nach 3.837 bzw. 3.836 Bars
#: Historie, danach nie wieder. Zwei Codes, ein Tag, je eine Zeile: ein
#: Zombie-Ticker in einem einzelnen Querschnitt, keine Firma.
#:
#: Die Folge war nicht nur eine schiefe Zahl. `resolve_symbols` bricht ab,
#: sobald zwei Segmente das Fenster schneiden — zu Recht, das ist der
#: BBBY-Schutz. Dieser eine Bar machte damit jeden Backtest über den Code
#: unmöglich, mit der Begründung, er habe den Besitzer gewechselt. Er hatte
#: nicht.
#:
#: **Warum aktive Segmente ausgenommen bleiben.** Ein kurzes Segment *an der
#: Front* ist von einer echten Neunotierung nicht zu unterscheiden: `KENT` trug
#: am 2026-08-27 genau 34 Bars seit dem 2014-12-16, und BBBYs Nachfolger hatte
#: am Anfang auch nicht mehr. Wer dort stillschweigend die alte Reihe
#: zurückgibt, begeht genau den BBBY-Fehler, nur später. Beendet **und** kurz
#: ist dagegen eine abgeschlossene Aussage: eine Firma, die einen Tag existiert
#: hat, hat es nicht gegeben. Und weil die Front nächtlich wandert, entscheidet
#: sich der offene Fall von selbst — eine echte Neunotierung wächst über die
#: Schwelle, ein Zombie bleibt stehen, wo er steht.
#:
#: **Die Zahl ist ein Urteil, keine Messung.** Die Verteilung der Ablösungen
#: hat keinen Knick (6,8 % unter 2 Bars, 17,3 % unter 5, 35,0 % unter 20,
#: 49,4 % unter 60): die Daten sagen, was jede Wahl kostet, nicht welche
#: richtig ist. Ein Handelsmonat ist der Kompromiss.
#:
#: Die Fehlerrichtung stimmt: wird ein Segment übergangen, **fehlen seine Tage**
#: im Ergebnis. In die fremde Reihe geraten können sie nicht, weil
#: `materialise` je Instrument auf `date BETWEEN first AND last` joint.
#: Weglassen statt erfinden — dieselbe Richtung wie überall sonst hier.
DEFAULT_MIN_SEGMENT_BARS = 20

_SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class Segment:
    """Ein zusammenhängender Lebensabschnitt eines Codes."""

    code: str
    exchange: str
    index: int
    first: date
    last: date
    n_bars: int

    @property
    def synthetic_key(self) -> str:
        code = _SAFE_KEY_RE.sub("-", self.code.strip().upper()) or "UNKNOWN"
        exch = _SAFE_KEY_RE.sub("-", (self.exchange or "US").strip().upper()) or "US"
        return f"code.{code}.{exch}.s{self.index}"


@dataclass
class IdentityMap:
    """Das Ergebnis der Auflösung: Segmente mit Schlüsseln, plus Kennzahlen.

    ``rows`` ist die eigentliche Karte (ein Eintrag je Segment). Die Zähler
    daneben sind kein Beiwerk — sie beantworten die Frage „mit welcher
    Genauigkeit", ohne dass jemand die Karte selbst auszählen muss.
    """

    rows: list[dict] = field(default_factory=list)
    lake_first: date | None = None
    lake_last: date | None = None
    gap_trading_days: int = DEFAULT_GAP_TRADING_DAYS
    min_segment_bars: int = DEFAULT_MIN_SEGMENT_BARS

    @property
    def n_instruments(self) -> int:
        return len(self.rows)

    @property
    def n_with_isin(self) -> int:
        return sum(1 for r in self.rows if r["identity_source"] == "isin")

    @property
    def n_superseded(self) -> int:
        """Segmente, die *vor* einem späteren Segment desselben Codes liegen.

        Das ist die Zahl, die den BBBY-Fall zählt: jedes dieser Segmente wäre
        bei naiver Verkettung stillschweigend in einer fremden Reihe gelandet.
        """
        return sum(1 for r in self.rows if r["superseded"])

    @property
    def n_short(self) -> int:
        """Segmente unter der Schwelle — die Fälle, die nichts ablösen können.

        Sie stehen neben ``n_superseded``, weil erst beide zusammen die Frage
        beantworten, ob eine Ablösung ein Besitzerwechsel war: ein Feed, der
        Zombie-Ticker in einzelne Querschnitte streut, produziert beliebig
        viele Segmente, die wie ein Neuanfang aussehen.
        """
        return sum(1 for r in self.rows if r["n_bars"] < self.min_segment_bars)

    def to_frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=list(_MANIFEST_COLUMNS))
        df = pd.DataFrame(self.rows)
        return df[list(_MANIFEST_COLUMNS)].sort_values("instrument").reset_index(drop=True)


_MANIFEST_COLUMNS = (
    "instrument",
    "code",
    "exchange",
    "segment",
    "identity_source",
    "isin",
    "first",
    "last",
    "n_bars",
    "superseded",
    "active",
)


def _prices_glob() -> str:
    return storage.cache_path(f"{US_EQUITY_PREFIX}/date=*/*.parquet")


def code_calendar(*, limit_days: int | None = None) -> pd.DataFrame:
    """Wann ist welcher Code im Lake zu sehen? Eine Zeile je (Code, Börse, Tag).

    Liest über DuckDB nur die drei Spalten, die für die Identität zählen —
    bei 30 Jahren × 26.000 Symbolen wäre alles andere ein Vielfaches an IO.

    ``limit_days`` schneidet auf die **ersten N Handelstage** zu. Gedacht zum
    Prototypen auf einem teilgeladenen Lake: gesampelt wird über ``date``, nie
    über Zeilen — ein halber Querschnitt wäre eine falsche Auskunft über die
    Breite eines Tages.

    **Ein leerer Lake ist eine Antwort, kein Fehler** — leerer Frame statt
    Ausnahme. DuckDB sieht das anders: ``read_parquet`` über ein Glob ohne
    Treffer wirft ``IOException``, und zwar *bevor* der Aufrufer sein
    ``if kalender.empty`` erreicht. Die Prüfung dagegen war damit toter Code,
    und `build_resolved.py` warf am 2026-08-12 einen Traceback, wo eine
    freundliche Zeile stehen sollte. Der LIST-Aufruf vorweg ist billig (einer,
    nicht einer pro Tag) und macht die Absicht wieder wahr.
    """
    if not storage.list_day_partitions(US_EQUITY_PREFIX):
        return pd.DataFrame(columns=["code", "exchange", "date"])

    glob = _prices_glob()
    con = storage._duckdb_conn()
    try:
        bedingungen = ["code IS NOT NULL"]
        if limit_days:
            tage = con.execute(
                f"SELECT DISTINCT date FROM read_parquet('{glob}', hive_partitioning=true) "
                "ORDER BY date LIMIT ?",
                [int(limit_days)],
            ).df()
            if tage.empty:
                return pd.DataFrame(columns=["code", "exchange", "date"])
            bis = pd.to_datetime(tage["date"]).max().date()
            bedingungen.append(f"date <= DATE '{bis.isoformat()}'")

        sql = (
            "SELECT code, "
            "COALESCE(NULLIF(TRIM(exchange_short_name), ''), 'US') AS exchange, "
            "date "
            f"FROM read_parquet('{glob}', hive_partitioning=true) "
            f"WHERE {' AND '.join(bedingungen)} "
            "ORDER BY code, exchange, date"
        )
        df = con.execute(sql).df()
    except Exception as exc:  # noqa: BLE001
        # Zweiter Boden: die LIST-Prüfung oben deckt den leeren Lake ab, aber
        # ein Glob kann auch scheitern, wenn Partitionen zwischen LIST und
        # Query verschwinden — oder wenn eine Datei kaputt ist. Beides ist
        # „keine Karte baubar", nicht „Absturz".
        log.warning("Schicht 1 nicht lesbar: %s", exc)
        return pd.DataFrame(columns=["code", "exchange", "date"])
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame(columns=["code", "exchange", "date"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def find_codes(codes: list[str], *, sample_days: int = 24) -> pd.DataFrame:
    """Kommen diese Codes in Schicht 1 vor — und wann?

    Die Frage, die vor jeder Architekturentscheidung steht: *deckt der Bulk
    überhaupt ab, was ich brauche?* Am 2026-08-12 war offen, ob der
    US-Querschnitt auch ETFs enthält oder nur Einzelaktien — davon hängt ab, ob
    Tiingo als zweite Quelle bleiben muss.

    Gelesen wird eine **Stichprobe** über das geladene Fenster, nicht alles:
    für ein Ja/Nein reichen zwei Dutzend Querschnitte, und ein voller Scan über
    7.700 Partitionen kostet Minuten für eine Antwort, die nach Sekunden
    feststeht.

    Returns
    -------
    DataFrame mit ``code``, ``n_tage`` (in der Stichprobe), ``first``, ``last``.
    Codes ohne Treffer erscheinen mit ``n_tage = 0`` — **die Zeile fehlt nicht**,
    denn „nicht gefunden" ist die eigentliche Auskunft.
    """
    gesucht = [c.strip().upper() for c in codes if c.strip()]
    if not gesucht:
        return pd.DataFrame(columns=["code", "n_tage", "first", "last"])

    tage = storage.list_day_partitions(US_EQUITY_PREFIX)
    leer = pd.DataFrame({"code": gesucht, "n_tage": 0, "first": None, "last": None})
    if not tage:
        return leer

    if len(tage) > sample_days:
        schritt = (len(tage) - 1) / (sample_days - 1)
        idx = sorted({round(i * schritt) for i in range(sample_days)})
        stichprobe = [tage[i] for i in idx]
    else:
        stichprobe = tage

    pfade = [
        storage.cache_path(f"{US_EQUITY_PREFIX}/date={t.isoformat()}/data.parquet")
        for t in stichprobe
    ]
    con = storage._duckdb_conn()
    try:
        platzhalter = ",".join(["?"] * len(pfade))
        code_platzhalter = ",".join(["?"] * len(gesucht))
        sql = (
            f"SELECT code, COUNT(*) AS n_tage, MIN(date) AS first, MAX(date) AS last "
            f"FROM read_parquet([{platzhalter}], union_by_name=true) "
            f"WHERE code IN ({code_platzhalter}) GROUP BY code"
        )
        treffer = con.execute(sql, [*pfade, *gesucht]).df()
    except Exception as exc:  # noqa: BLE001
        log.warning("Code-Suche nicht möglich: %s", exc)
        return leer
    finally:
        con.close()

    if treffer.empty:
        return leer
    # Nicht gefundene Codes bleiben mit 0 stehen — ein Ergebnis, kein Loch.
    zusammen = leer.merge(treffer, on="code", how="left", suffixes=("_leer", ""))
    zusammen["n_tage"] = zusammen["n_tage"].fillna(0).astype(int)
    return zusammen[["code", "n_tage", "first", "last"]]


def segment_codes(
    calendar: pd.DataFrame, gap_trading_days: int = DEFAULT_GAP_TRADING_DAYS
) -> list[Segment]:
    """Zerschneidet jeden Code an Lücken > ``gap_trading_days`` **Handelstagen**.

    Gezählt werden die Tage, an denen der Lake *überhaupt* einen Querschnitt
    hat — nicht Kalendertage. Wo keine Partition liegt, hat auch niemand
    gehandelt, und die Abwesenheit eines Codes sagt dort nichts über ihn aus
    (siehe ``DEFAULT_GAP_TRADING_DAYS``).

    Die Tagesliste kommt aus dem Kalender selbst: er enthält jede (Code, Tag)-
    Kombination, seine eindeutigen Daten **sind** die Handelstage des Lakes.
    Ein zusätzlicher Parameter wäre eine zweite Quelle für dieselbe Auskunft.

    Pure Funktion über einen Frame mit ``code``/``exchange``/``date`` — damit
    ohne Lake testbar, und der BBBY-Fall lässt sich als Fixture schreiben.
    """
    if calendar.empty:
        return []

    df = calendar.sort_values(["code", "exchange", "date"]).reset_index(drop=True)
    rang = {d: i for i, d in enumerate(sorted(set(df["date"])))}
    grenze = int(gap_trading_days)
    out: list[Segment] = []

    for (code, exchange), teil in df.groupby(["code", "exchange"], sort=True):
        dates = list(teil["date"])
        start = dates[0]
        prev = dates[0]
        n = 1
        index = 1
        for d in dates[1:]:
            if rang[d] - rang[prev] > grenze:
                out.append(Segment(str(code), str(exchange), index, start, prev, n))
                index += 1
                start = d
                n = 0
            prev = d
            n += 1
        out.append(Segment(str(code), str(exchange), index, start, prev, n))
    return out


def segments_from_lake(
    *, gap_trading_days: int = DEFAULT_GAP_TRADING_DAYS, limit_days: int | None = None
) -> list[Segment]:
    """Dieselben Segmente wie ``segment_codes``, aber ohne den Kalender im RAM.

    **Warum es die zweite Fassung gibt.** ``code_calendar`` gibt eine Zeile je
    (Code, Börse, Tag) zurück — am 2026-08-12 waren das **19,6 Millionen**
    Zeilen für sechs Jahre Historie, und der volle Zeitraum wird ein Vielfaches
    davon. In pandas sind das Gigabytes für eine Frage, deren Antwort
    fünfstellig ist: *wo sind die Lücken?*

    Die Segmentgrenzen rechnet DuckDB mit zwei Fensterfunktionen aus und gibt
    nur sie zurück — Größenordnung 10⁴ Zeilen statt 10⁷.

    ``segment_codes`` bleibt: es ist die pure Fassung über einen Frame, an der
    die Regeln testbar sind (der BBBY-Fall als Fixture). Dass beide dasselbe
    liefern, prüft ``test_resolve_roundtrip`` gegen einen echten kleinen Lake —
    zwei Implementierungen ohne Gleichheitsbeweis wären zwei Wahrheiten.
    """
    if not storage.list_day_partitions(US_EQUITY_PREFIX):
        return []

    glob = _prices_glob()
    con = storage._duckdb_conn()
    try:
        bis_klausel = ""
        if limit_days:
            tage = con.execute(
                f"SELECT DISTINCT date FROM read_parquet('{glob}', hive_partitioning=true) "
                "ORDER BY date LIMIT ?",
                [int(limit_days)],
            ).df()
            if tage.empty:
                return []
            bis = pd.to_datetime(tage["date"]).max().date()
            bis_klausel = f"AND date <= DATE '{bis.isoformat()}'"

        # `GROUP BY` im ersten Schritt, weil ein Tag denselben Code zweimal
        # tragen kann (zwei Handelsplätze, doppelte Zeile im Feed). Ohne das
        # zählte `n_bars` Duplikate mit, und die Lückenrechnung sähe eine
        # Differenz von 0 Tagen.
        #
        # `lake_tage` nummeriert die Tage, an denen der Lake überhaupt einen
        # Querschnitt hat. Die Lücke wird gegen diese Nummer gemessen, nicht
        # gegen das Kalenderdatum — sonst zählt ein Loch im Lake als
        # Besitzerwechsel (siehe `DEFAULT_GAP_TRADING_DAYS`).
        sql = f"""
            WITH quelle AS (
                SELECT code,
                       COALESCE(NULLIF(TRIM(exchange_short_name), ''), 'US') AS exchange,
                       date
                FROM read_parquet('{glob}', hive_partitioning=true)
                WHERE code IS NOT NULL {bis_klausel}
                GROUP BY 1, 2, 3
            ),
            lake_tage AS (
                SELECT date, row_number() OVER (ORDER BY date) AS tag_nr
                FROM (SELECT DISTINCT date FROM quelle)
            ),
            tage AS (
                SELECT q.code, q.exchange, q.date, l.tag_nr
                FROM quelle q JOIN lake_tage l ON q.date = l.date
            ),
            brueche AS (
                SELECT code, exchange, date,
                       CASE
                           WHEN tag_nr - lag(tag_nr) OVER (
                                    PARTITION BY code, exchange ORDER BY date
                                ) > {int(gap_trading_days)}
                           THEN 1 ELSE 0
                       END AS bruch
                FROM tage
            ),
            nummeriert AS (
                SELECT code, exchange, date,
                       SUM(bruch) OVER (
                           PARTITION BY code, exchange ORDER BY date
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                       ) AS segment
                FROM brueche
            )
            SELECT code, exchange, segment,
                   min(date) AS first, max(date) AS last, count(*) AS n_bars
            FROM nummeriert
            GROUP BY code, exchange, segment
            ORDER BY code, exchange, segment
        """
        df = con.execute(sql).df()
    finally:
        con.close()

    if df.empty:
        return []

    df["first"] = pd.to_datetime(df["first"]).dt.date
    df["last"] = pd.to_datetime(df["last"]).dt.date
    return [
        Segment(
            code=str(r.code),
            exchange=str(r.exchange),
            # `segment` zählt ab 0, `Segment.index` ab 1 — dieselbe Konvention
            # wie `segment_codes`, sonst hießen dieselben Segmente in beiden
            # Fassungen anders (`code.BBBY.US.s1` vs. `…s0`).
            index=int(r.segment) + 1,
            first=r.first,
            last=r.last,
            n_bars=int(r.n_bars),
        )
        for r in df.itertuples(index=False)
    ]


def _isin_candidate(code: str) -> str | None:
    """ISIN aus der geprüften Karte — nur wenn sie eine ISIN *ist*."""
    raw = load_instrument_map().get(code.strip().upper())
    if not raw:
        return None
    try:
        isin = normalise_isin(raw)
    except ValueError:
        return None
    return isin if isin_checksum_ok(isin) else None


def build_identity_map(
    calendar: pd.DataFrame | None = None,
    *,
    segments: list[Segment] | None = None,
    gap_trading_days: int = DEFAULT_GAP_TRADING_DAYS,
    active_tolerance_days: int = DEFAULT_ACTIVE_TOLERANCE_DAYS,
    min_segment_bars: int = DEFAULT_MIN_SEGMENT_BARS,
) -> IdentityMap:
    """Segmente + Schlüssel + Herkunft. Das Herzstück der Schicht.

    Die ISIN wird **nur** an ein Segment vergeben, das bis ans Ende des
    geladenen Fensters reicht. Alles davor ist historisch: die Karte kennt den
    heutigen Inhaber des Codes, nicht den damaligen.

    Abgelöst ist ein Segment nur durch einen Nachfolger, der entweder
    ``min_segment_bars`` Bars trägt oder bis an die Front reicht (siehe
    ``DEFAULT_MIN_SEGMENT_BARS``). Ein einzelner, längst beendeter
    Tagesquerschnitt hinter einer fünfzehnjährigen Historie ist kein
    Besitzerwechsel, sondern ein Zombie-Ticker — und würde die echte Reihe
    grundlos als „abgelöst" abstempeln.

    ``segments`` überspringt den Kalender-Schritt — der Weg für den echten
    Lake, wo ``segments_from_lake`` die Grenzen in DuckDB rechnet, statt 19,6
    Mio Code-Tage in den Speicher zu holen. ``calendar`` bleibt der Weg für
    Tests und kleine Frames.
    """
    if segments is None:
        if calendar is None:
            raise ValueError("build_identity_map braucht `calendar` oder `segments`.")
        segments = segment_codes(calendar, gap_trading_days=gap_trading_days)
    if not segments:
        return IdentityMap(rows=[], gap_trading_days=gap_trading_days)

    lake_last = max(s.last for s in segments)
    lake_first = min(s.first for s in segments)
    aktiv_ab = lake_last - timedelta(days=int(active_tolerance_days))

    # Bis zu welchem Index hat der Code ein **substanzielles** Segment? Nur so
    # lässt sich sagen, ob ein Segment von einem späteren abgelöst wurde — und
    # ein beendeter Stummel dahinter löst nichts ab, er beweist nur, dass der
    # Feed den Code an einem Tag noch einmal geführt hat. Reicht das kurze
    # Segment dagegen bis an die Front, bleibt es zählend: dort sieht eine echte
    # Neunotierung genauso aus (siehe DEFAULT_MIN_SEGMENT_BARS).
    letzte: dict[tuple[str, str], int] = {}
    for s in segments:
        if s.n_bars < min_segment_bars and s.last < aktiv_ab:
            continue
        key = (s.code, s.exchange)
        letzte[key] = max(letzte.get(key, 0), s.index)

    rows: list[dict] = []
    for s in segments:
        superseded = s.index < letzte.get((s.code, s.exchange), 0)
        active = s.last >= aktiv_ab
        isin = None
        if active and not superseded:
            isin = _isin_candidate(s.code)

        if isin:
            instrument = f"isin.{isin}"
            quelle = "isin"
        else:
            instrument = s.synthetic_key
            quelle = "synthetic"

        rows.append(
            {
                "instrument": instrument,
                "code": s.code,
                "exchange": s.exchange,
                "segment": s.index,
                "identity_source": quelle,
                "isin": isin,
                "first": s.first,
                "last": s.last,
                "n_bars": s.n_bars,
                "superseded": superseded,
                "active": active,
            }
        )

    # Zwei Segmente dürfen nie denselben Schlüssel bekommen. Passieren kann das
    # nur, wenn die Karte zwei aktive Codes auf dieselbe ISIN zeigt (Zweit-
    # notierung, oder ein Fehler in der Karte). Beide dann synthetisch zu
    # führen ist die konservative Auflösung: lieber zwei getrennte Reihen als
    # eine stillschweigend verschmolzene.
    zaehler: dict[str, int] = {}
    for r in rows:
        zaehler[r["instrument"]] = zaehler.get(r["instrument"], 0) + 1
    for r in rows:
        if zaehler[r["instrument"]] > 1 and r["identity_source"] == "isin":
            log.warning(
                "ISIN %s ist mehreren aktiven Codes zugeordnet — %s wird synthetisch geführt.",
                r["isin"],
                r["code"],
            )
            r["instrument"] = Segment(
                r["code"], r["exchange"], r["segment"], r["first"], r["last"], r["n_bars"]
            ).synthetic_key
            r["identity_source"] = "synthetic"
            r["isin"] = None

    return IdentityMap(
        rows=rows,
        lake_first=lake_first,
        lake_last=lake_last,
        gap_trading_days=gap_trading_days,
        min_segment_bars=min_segment_bars,
    )


#: Eine Marktseite liest die Karte sonst vier Mal (Detail, Profil zweimal,
#: Chart). Sie ändert sich nur bei einem Rebuild — sechzig Sekunden reichen.
_MANIFEST_TTL_S = 60.0
#: (pfad, monotonic-ts, frame). Der Pfad ist der Schlüssel: Tests und Prod
#: zeigen auf verschiedene Lakes, und ein Cache ohne ihn würde die eine
#: Welt als die andere ausgeben.
_manifest_memo: tuple[str, float, pd.DataFrame] | None = None


def _drop_manifest_cache() -> None:
    global _manifest_memo
    _manifest_memo = None


def _manifest_cache_key(path: str) -> str:
    """Absolut, sonst teilen lokale Tests denselben Relativpfad über ``chdir``."""
    if path.startswith("s3://"):
        return path
    return str(Path(path).resolve())


def write_manifest(imap: IdentityMap) -> str:
    """Schreibt die Karte als eine Parquet-Datei in den Lake."""
    path = storage.cache_path(MANIFEST_PATH)
    storage.write_parquet(imap.to_frame(), path)
    _drop_manifest_cache()
    return path


def read_manifest() -> pd.DataFrame:
    """Die Karte aus dem Lake. Leerer Frame, wenn Schicht 2 nie gebaut wurde.

    Der Frame wird kopiert zurückgegeben: Aufrufer filtern und schreiben
    Spalten, und eine geteilte Referenz würde den nächsten Read vergiften.
    """
    global _manifest_memo
    path = storage.cache_path(MANIFEST_PATH)
    key = _manifest_cache_key(path)
    now = time.monotonic()
    if (
        _manifest_memo is not None
        and _manifest_memo[0] == key
        and now - _manifest_memo[1] < _MANIFEST_TTL_S
    ):
        return _manifest_memo[2].copy()

    if not storage.exists(path):
        leer = pd.DataFrame(columns=list(_MANIFEST_COLUMNS))
        _manifest_memo = (key, now, leer)
        return leer.copy()
    df = storage.read_parquet(path)
    for col in ("first", "last"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date
    _manifest_memo = (key, now, df)
    return df.copy()


def materialise(imap: IdentityMap, *, only: list[str] | None = None) -> int:
    """Schreibt je Instrument eine Parquet-Datei aus Schicht 1.

    ``only`` beschränkt auf bestimmte Instrument-Schlüssel — der Weg, mit dem
    man ein Universum materialisiert, ohne 26.000 Dateien anzufassen.

    **Ein Durchlauf, nicht einer pro Instrument.** Die naheliegende Fassung
    filtert Schicht 1 je Instrument einmal; bei 26.000 Instrumenten über 7.700
    Tagespartitionen wären das 26.000 volle Scans. Stattdessen wird die Karte
    als Tabelle in DuckDB registriert, einmal gegen die Kurse gejoint und
    partitioniert geschrieben. Der Join über ``date BETWEEN first AND last``
    ist dabei die Stelle, an der die Segmentierung wirkt: eine Zeile von 2007
    kann nicht im Instrument des Nachfolgers landen, weil sie außerhalb seines
    Fensters liegt.

    Gibt die Zahl geschriebener Instrumente zurück. Bestehende Dateien werden
    überschrieben: die Schicht ist abgeleitet, ein Rebuild ist ihr Normalfall.
    """
    rows = imap.rows
    if only:
        gewuenscht = set(only)
        rows = [r for r in rows if r["instrument"] in gewuenscht]
    if not rows:
        return 0

    karte = pd.DataFrame(
        [
            {
                "instrument": r["instrument"],
                "code": r["code"],
                "exchange": r["exchange"],
                "first": r["first"],
                "last": r["last"],
            }
            for r in rows
        ]
    )

    glob = _prices_glob()
    ziel = storage.cache_path(RESOLVED_PREFIX)
    con = storage._duckdb_conn()
    try:
        con.register("identitaet", karte)
        con.execute(
            f"""
            COPY (
                SELECT
                    m.instrument AS instrument,
                    p.code AS code,
                    COALESCE(NULLIF(TRIM(p.exchange_short_name), ''), 'US') AS exchange,
                    p.date AS date,
                    p.open, p.high, p.low, p.close, p.volume, p.adjusted_close
                FROM read_parquet('{glob}', hive_partitioning=true) p
                JOIN identitaet m
                  ON p.code = m.code
                 AND COALESCE(NULLIF(TRIM(p.exchange_short_name), ''), 'US') = m.exchange
                 AND p.date >= m.first AND p.date <= m.last
                ORDER BY m.instrument, p.date
            ) TO '{ziel}'
            (FORMAT PARQUET, PARTITION_BY (instrument), OVERWRITE_OR_IGNORE)
            """
        )
    finally:
        con.close()

    # Was wirklich entstanden ist, sagt der Lake — nicht die Absicht. Bei
    # einem Instrument ohne passende Kurszeile schreibt DuckDB keine Partition.
    _drop_materialised_cache()
    return len(materialised_keys() & {r["instrument"] for r in rows})


class AmbiguousCodeError(ValueError):
    """Ein Code gehörte im angefragten Fenster mehr als einem Papier.

    Das ist der BBBY-Fall in seiner teuersten Form: die Anfrage überspannt den
    Besitzerwechsel. Es gibt keine richtige Antwort — eines der beiden Papiere
    zu wählen erfindet Historie, beide zu verketten auch.
    """


def resolve_symbols(
    codes: list[str],
    start: date,
    end: date,
    *,
    manifest: pd.DataFrame | None = None,
    min_segment_bars: int = DEFAULT_MIN_SEGMENT_BARS,
) -> tuple[dict[str, str], list[str]]:
    """Ticker → Instrument-Schlüssel **für ein Zeitfenster**.

    Returns
    -------
    (aufgeloest, fehlend)
        ``aufgeloest`` bildet den angefragten Code auf den Instrument-Schlüssel
        ab, dessen Segment das Fenster überlappt. ``fehlend`` sind Codes, für
        die im Fenster nichts liegt.

    Raises
    ------
    AmbiguousCodeError
        Wenn **zwei** Segmente desselben Codes das Fenster überlappen. Dann hat
        der Ticker während der Anfrage den Besitzer gewechselt, und jede
        automatische Wahl wäre eine Erfindung. Der Aufrufer muss das Fenster
        teilen oder den Instrument-Schlüssel direkt nennen.

        **Nicht** aber, wenn die anderen beendete Stummel unter
        ``min_segment_bars`` sind: ein einzelner Querschnitt aus einem toten
        Code ist kein Besitzerwechsel, und ein Abbruch dafür kostet einen
        Backtest, den nichts gefährdet. Ein kurzes Segment, das bis an die
        Front reicht, löst den Abbruch weiterhin aus — dort sieht eine echte
        Neunotierung genauso aus. Die Tage des kurzen Segments fehlen dann im Ergebnis
        — sie können nicht in die fremde Reihe geraten, weil ``materialise``
        je Instrument auf ``date BETWEEN first AND last`` joint.

    **Warum das Fenster mitentscheidet.** Ohne Zeitbezug wäre `BBBY` heute
    Overstock — und ein Backtest über 2019 bekäme die Kurse einer Firma, die
    den Namen erst 2023 kaufte. Die Segmentierung aus `build_identity_map`
    trennt die beiden; erst dieser Aufruf macht die Trennung nutzbar.
    """
    karte = read_manifest() if manifest is None else manifest
    gesucht = [c.strip().upper() for c in codes if c.strip()]
    if karte.empty:
        return {}, gesucht

    aufgeloest: dict[str, str] = {}
    fehlend: list[str] = []
    for code in gesucht:
        kandidaten = karte[karte["code"].astype(str).str.upper() == code]
        # Überlappung im halboffenen Sinn: das Segment muss im Fenster
        # tatsächlich Bars haben, Berührung an genau einem Tag reicht.
        passend = kandidaten[(kandidaten["first"] <= end) & (kandidaten["last"] >= start)]
        if passend.empty:
            fehlend.append(code)
            continue
        if len(passend) > 1:
            # Erst die beendeten Stummel aussortieren — sonst blockiert ein
            # einzelner Zombie-Bar eine Historie, die niemand bestreitet. Was
            # bis an die Front reicht, bleibt stehen: dort ist eine echte
            # Neunotierung nicht zu unterscheiden. Bleibt danach mehr als eines
            # übrig, ist es ein Besitzerwechsel und der Abbruch bleibt.
            lang = pd.to_numeric(passend["n_bars"], errors="coerce").fillna(0) >= min_segment_bars
            aktiv = (
                passend["active"].fillna(True).astype(bool)
                if "active" in passend.columns
                else True
            )
            substanziell = passend[lang | aktiv]
            if len(substanziell) == 1:
                passend = substanziell
        if len(passend) > 1:
            teile = ", ".join(
                f"{r.instrument} ({r.first}…{r.last})" for r in passend.itertuples(index=False)
            )
            raise AmbiguousCodeError(
                f"{code} gehörte zwischen {start} und {end} mehr als einem Papier: {teile}. "
                "Fenster teilen oder den Instrument-Schlüssel direkt angeben — "
                "eine automatische Wahl würde Historie erfinden."
            )
        aufgeloest[code] = str(passend.iloc[0]["instrument"])
    return aufgeloest, fehlend


#: Ein LIST über `resolved/` bei 42.000+ materialisierten Instrumenten paginiert
#: auf S3/R2 (~1000 Keys je Seite) — gemessen ~11.6s für einen einzelnen Read.
#: `read_instruments` ruft das bei **jedem** Candle-Request auf, bevor es
#: überhaupt ein Fenster kennt: ein Chart über eine Woche zahlte dieselbe
#: Steuer wie „Max". Ändert sich nur bei `materialise`/`prune_stale`
#: (Operator-Lauf, beide löschen den Cache selbst) — fünf Minuten wie die
#: Universe-Coverage (`api.services.market._COVERAGE_TTL_SECONDS`), nicht nur
#: sechzig wie beim Manifest: die Liste ist teurer zu erneuern und ändert sich
#: seltener als die Karte.
_MATERIALISED_TTL_S = 300.0
_materialised_memo: tuple[str, float, frozenset[str]] | None = None


def _drop_materialised_cache() -> None:
    global _materialised_memo
    _materialised_memo = None


def materialised_keys() -> set[str]:
    """Die Instrument-Schlüssel, für die Kursdateien liegen.

    Ein LIST-Aufruf statt eines ``exists`` pro Instrument — bei 26.000
    Instrumenten ist der Unterschied zwischen einem Roundtrip und 26.000.
    """
    global _materialised_memo
    key = _manifest_cache_key(storage.cache_path(RESOLVED_PREFIX))
    now = time.monotonic()
    if (
        _materialised_memo is not None
        and _materialised_memo[0] == key
        and now - _materialised_memo[1] < _MATERIALISED_TTL_S
    ):
        return set(_materialised_memo[2])

    keys = frozenset(
        name.removeprefix("instrument=")
        for name in storage.list_children(RESOLVED_PREFIX)
        if name.startswith("instrument=")
    )
    _materialised_memo = (key, now, keys)
    return set(keys)


class PruneRefusedError(RuntimeError):
    """Aufräumen abgelehnt — die Karte beschreibt den Lake nicht vollständig.

    Löschen ist die einzige Operation dieser Schicht, die sich nicht durch
    einen Rebuild rückgängig machen lässt (die Rohdaten schon, die abgeleiteten
    Dateien kosten einen Neubau). Deshalb wird hier lieber nichts getan als das
    Falsche.
    """


def stale_keys(karte: IdentityMap | pd.DataFrame) -> set[str]:
    """Materialisierte Instrumente, die in dieser Karte **nicht mehr** vorkommen.

    Nimmt die frisch gebaute ``IdentityMap`` oder das geschriebene Manifest —
    dieselbe Frage, zwei Zeitpunkte: vor dem Schreiben (was würde verwaisen)
    und danach (was ist verwaist).

    Warum es die überhaupt gibt: ``materialise`` schreibt mit
    ``OVERWRITE_OR_IGNORE`` — es überschreibt, was es schreibt, und rührt den
    Rest nicht an. Ändert sich ein Schlüssel, bleibt die alte Partition liegen.

    Und Schlüssel **ändern sich**, solange der Lake wächst: der synthetische
    Schlüssel trägt den Segment-Index (``code.BBBY.US.s1``), und Segmentgrenzen
    werden in Lake-Handelstagen gemessen. Zwei Blöcke, zwischen denen noch
    nichts geladen ist, sehen wie ein Segment aus; füllt der Ladelauf die Jahre
    dazwischen, zerfällt es in ``s1`` und ``s2``. Dasselbe beim Wechsel
    ``synthetic → isin``, sobald die Instrument-Karte den Code kennt.

    Ohne Aufräumen wächst der Rest still mit, und ``materialised_keys`` — die
    Zahl, die „backtestbar" behauptet — zählt ihn mit.
    """
    if isinstance(karte, pd.DataFrame):
        gueltig = set() if karte.empty else set(karte["instrument"].astype(str))
    else:
        gueltig = {r["instrument"] for r in karte.rows}
    return materialised_keys() - gueltig


def prune_stale(imap: IdentityMap, *, partial: bool = False, dry_run: bool = False) -> list[str]:
    """Entfernt die Partitionen aus ``stale_keys``. Gibt die Schlüssel zurück.

    ``partial=True`` sagt: diese Karte wurde nur über einen **Ausschnitt** des
    Lakes gebaut (``--limit-days``). Dann heißt „steht nicht in der Karte"
    nicht „ist veraltet", sondern „lag außerhalb des Ausschnitts" — und
    Löschen würde genau die Instrumente treffen, die in Ordnung sind. Der Fall
    wird abgelehnt statt geraten.

    Eine leere Karte wird aus demselben Grund abgelehnt: sie unterscheidet
    „nichts ist mehr gültig" nicht von „der Bau ist schiefgegangen", und die
    zweite Lesart ist die wahrscheinlichere.

    ``dry_run`` beantwortet dieselbe Frage ohne zu löschen.
    """
    if partial:
        raise PruneRefusedError(
            "Karte aus einem Lake-Ausschnitt (--limit-days) beschreibt nicht den "
            "ganzen Bestand — was hier fehlt, ist nicht veraltet, sondern ungefragt."
        )
    if not imap.rows:
        raise PruneRefusedError(
            "Leere Karte — das ist kein Befund, sondern ein fehlgeschlagener Bau."
        )

    veraltet = sorted(stale_keys(imap))
    if dry_run:
        return veraltet

    for key in veraltet:
        storage.delete_tree(storage.cache_path(f"{RESOLVED_PREFIX}/instrument={key}"))
    if veraltet:
        _drop_materialised_cache()
        log.info(
            "Schicht 2 aufgeräumt: %d verwaiste Instrument-Partitionen entfernt.", len(veraltet)
        )
    return veraltet


@dataclass(frozen=True)
class Bruch:
    """Ein Instrument, dessen Kursdatei nicht hält, was das Manifest verspricht."""

    instrument: str
    #: ``n_bars`` aus dem Manifest — die Zusage.
    versprochen: int
    #: Zeilen in der materialisierten Kursdatei — was tatsächlich gelesen wird.
    vorhanden: int
    #: ``last`` aus dem Manifest.
    verspricht_bis: date
    #: Letzter Bar in der Datei. ``None``, wenn die Partition leer ist.
    reicht_bis: date | None

    @property
    def fehlend(self) -> int:
        """Positiv: die Datei hinkt nach. Negativ: das Manifest ist das ältere."""
        return self.versprochen - self.vorhanden


@dataclass(frozen=True)
class Materialisierungsbefund:
    """Wie viele Instrumente geprüft wurden, und welche davon auseinanderlaufen."""

    geprueft: int
    brueche: list[Bruch]

    @property
    def ok(self) -> bool:
        return not self.brueche


def stichprobe(karte: pd.DataFrame, *, n: int) -> list[str]:
    """Welche materialisierten Instrumente werden geprüft — deterministisch.

    Zwei Hälften, weil es zwei Fehlerbilder gibt:

    * **Die Hälfte mit den meisten Bars.** Der Rückstand aus #298 ist
      systematisch — Schicht 2 stammt aus einer Zeit, in der der Lake früher
      endete. Er trifft zuerst die Reihen, die am weitesten reichen, und dort
      ist er am größten. AAPL und SPY liegen in dieser Hälfte.
    * **Ein gleichmäßiger Querschnitt über den Rest.** Ein einzelner
      fehlgeschlagener Schreibvorgang ist nicht nach Bar-Zahl sortiert; eine
      reine Top-Liste würde ihn nie sehen.

    Deterministisch und nicht zufällig, damit zwei Läufe dieselbe Antwort
    vergleichbar machen: eine Zahl, die zwischen zwei Aufrufen springt, weil
    die Stichprobe wanderte, ist keine Auskunft über den Lake.
    """
    if karte.empty or n <= 0:
        return []
    vorhanden = materialised_keys()
    kandidaten = karte[karte["instrument"].astype(str).isin(vorhanden)]
    if kandidaten.empty:
        return []

    gereiht = kandidaten.sort_values(
        ["n_bars", "instrument"], ascending=[False, True]
    )["instrument"].astype(str).tolist()
    if len(gereiht) <= n:
        return gereiht

    kopf = gereiht[: n // 2]
    rest = sorted(set(gereiht) - set(kopf))
    schritt = max(1, len(rest) // max(1, n - len(kopf)))
    quer = rest[::schritt][: n - len(kopf)]
    return sorted(set(kopf) | set(quer))


def _datei_stand(schluessel: list[str]) -> dict[str, tuple[int, date | None]]:
    """Zeilenzahl und letzter Bar je Instrument — aus den Dateien selbst.

    Eine Abfrage über die betroffenen Partitionen statt einer je Instrument,
    und nur über die Spalte ``date``: DuckDB liest Parquet spaltenweise, der
    Rest der Datei wird nicht angefasst. ``instrument`` kommt aus dem
    Hive-Pfad, denn ``PARTITION_BY`` nimmt die Spalte aus den Daten heraus.
    """
    if not schluessel:
        return {}
    pfade = [
        storage.cache_path(f"{RESOLVED_PREFIX}/instrument={k}/*.parquet") for k in schluessel
    ]
    con = storage._duckdb_conn()
    try:
        df = con.execute(
            "SELECT instrument, COUNT(*) AS n, MAX(date) AS letzter "
            "FROM read_parquet(?, hive_partitioning=true) GROUP BY instrument",
            [pfade],
        ).df()
    finally:
        con.close()

    out: dict[str, tuple[int, date | None]] = {}
    for r in df.itertuples(index=False):
        letzter = pd.to_datetime(r.letzter).date() if pd.notna(r.letzter) else None
        out[str(r.instrument)] = (int(r.n), letzter)
    return out


def audit_materialised(karte: pd.DataFrame, *, n: int = 40) -> Materialisierungsbefund:
    """Hält die Kursdatei, was das Manifest über sie behauptet?

    **Warum diese Prüfung existiert.** Das Manifest und die materialisierten
    Kursdateien sind zwei Stände, die auseinanderlaufen können, ohne dass
    irgendwo etwas rot wird — und am 2026-08-26 haben sie das getan: für AAPL
    versprach die Karte 4.398 Bars bis 2013-09-23, die Datei hatte 1.683 bis
    2011-08-25 (#298). Sichtbar war nichts: `/market/<symbol>` liest die
    Coverage-Zeile aus dem Manifest und zeichnet die Kurve aus der Datei, also
    zeigte die Oberfläche eine Zusage neben einer Kurve, die sie nicht hält.

    ``build_resolved.py --status`` prüfte bis dahin nur die *Karte* gegen den
    Rohbestand — die Richtung „Datei gegen Karte" stellte niemand, obwohl die
    Zahl in beiden Quellen bereits steht.

    **Stichprobe, und das ausdrücklich.** Ein Vollabgleich wäre ein Roundtrip
    je Instrument; bei 62.152 Instrumenten gegen R2 ist das kein Statusbericht
    mehr. Der Fehler, um den es geht, ist ohnehin systematisch: eine veraltete
    Schicht 2 ist nicht bei einzelnen Reihen alt, sondern bei allen, deren
    Fenster über das damalige Lake-Ende reicht. Eine Stichprobe von einigen
    Dutzend findet das zuverlässig — was sie nicht kann, ist eine
    Unauffälligkeit auf den ganzen Bestand hochrechnen, und genau so wird sie
    auch berichtet.

    Nicht materialisierte Instrumente kommen nicht vor. Eine fehlende Datei ist
    kein Widerspruch, sondern eine Entscheidung (``--top`` statt ``--all``);
    verwaiste Schlüssel beantwortet ``stale_keys``.
    """
    if karte.empty:
        return Materialisierungsbefund(geprueft=0, brueche=[])

    keys = stichprobe(karte, n=n)
    if not keys:
        return Materialisierungsbefund(geprueft=0, brueche=[])

    stand = _datei_stand(keys)
    zeilen = karte.set_index(karte["instrument"].astype(str))

    brueche: list[Bruch] = []
    for k in keys:
        vorhanden, reicht_bis = stand.get(k, (0, None))
        zeile = zeilen.loc[k]
        versprochen = int(zeile["n_bars"])
        if vorhanden == versprochen:
            continue
        brueche.append(
            Bruch(
                instrument=k,
                versprochen=versprochen,
                vorhanden=vorhanden,
                verspricht_bis=zeile["last"],
                reicht_bis=reicht_bis,
            )
        )

    brueche.sort(key=lambda b: (-abs(b.fehlend), b.instrument))
    return Materialisierungsbefund(geprueft=len(keys), brueche=brueche)


# ---------------------------------------------------------------------------
# Ticker → CIK, mit Zeitbezug
#
# Dieselbe Falle wie bei den Kursen, eine Ebene höher. SECs
# `company_tickers.json` ist der **heutige** Stand: `BBBY` zeigt darin auf
# Overstock, weil die Firma die Marke aus der Insolvenzmasse kaufte. Wer für
# einen Backtest über 2007 danach fragt, bekommt die Bilanzzahlen der
# Nachfolgefirma auf den Kursen der alten — zwei Fehlzuordnungen, die sich
# gegenseitig plausibel machen.
#
# **Was hier geht und was nicht.** Die SEC liefert keine historische
# Ticker→CIK-Karte; `formerNames` in `submissions` führt frühere *Firmennamen*,
# nicht frühere Ticker. Eine tote Firma über ihren damaligen Ticker
# nachzuschlagen ist damit von EDGAR aus nicht möglich.
#
# Erkennen lässt sich der gefährliche Fall trotzdem — aus **unseren eigenen**
# Kursdaten: Schicht 2 weiß, dass `BBBY` zwei Segmente hat und welches davon
# abgelöst ist. Fällt der Stichtag in ein abgelöstes Segment, dann beschreibt
# die Heute-Karte per Definition eine andere Firma, und die Antwort ist
# `recycled` statt einer Zahl.
#
# Das ist dieselbe Haltung wie bei `resolve_symbols`: **keine Identität
# erfinden.** Lieber „unbekannt" als der falsche Konzern.


#: Auflösungsergebnisse, die ein Backtest verwenden darf.
USABLE_CIK_STATUS = frozenset({"confirmed", "unverified", "resolved_by_name"})

#: Lake-Pfad der Namenskarte aus `scripts/build_cik_map.py`.
CIK_HISTORY_PATH = "cik_history/_map.parquet"

def _history_for(schluessel: str) -> pd.DataFrame:
    """Nur die Zeilen zu **einem** Namensschlüssel — gezielt, nicht alles.

    **Warum das keine gecachte Volltabelle sein darf.** Die Karte hat 1.054.038
    Zeilen; sie als Modulvariable zu halten kostet im API-Prozess dreistellige
    Megabyte. Der Container hat 512 MB, und ein Prozess, der zu viel hält, wird
    nicht mit einer Fehlermeldung beendet, sondern hart getötet — im Log steht
    dann ein Neustart ohne Ursache, genau das Bild aus den Render-Logs vom
    2026-08-15.

    DuckDB liest die Parquet-Datei spaltenweise und filtert vor dem
    Materialisieren; zurück kommen die wenigen Zeilen, die der Schlüssel trifft.
    Derselbe Weg wie `storage.read_symbols` für Kurse.
    """
    spalten = ["cik", "name", "name_norm", "valid_from", "valid_to", "is_former", "tickers"]
    pfad = storage.cache_path(CIK_HISTORY_PATH)
    if not storage.exists(pfad):
        return pd.DataFrame(columns=spalten)

    con = storage._duckdb_conn()
    try:
        return con.execute(
            "SELECT cik, name, name_norm, valid_from, valid_to "
            "FROM read_parquet(?) WHERE name_norm = ?",
            [pfad, schluessel],
        ).df()
    finally:
        con.close()


def cik_by_name(
    name: str,
    as_of: date,
    *,
    history: pd.DataFrame | None = None,
) -> tuple[str | None, str]:
    """Firmenname zum Stichtag → CIK, plus Begründung. `None` heißt: kein Urteil.

    **Warum der Stichtag Teil der Frage ist.** Ein Name gehört nicht einer
    Firma, sondern einer Firma *in einem Zeitraum*. PineBridge Securities hiess
    bis 2009 AIG Equity Sales; wer den Namen ohne Datum nachschlägt, bekommt
    beide und muss raten.

    **Und warum Mehrdeutigkeit hier keine Auswahl auslöst.** Gemessen am
    2026-08-15 über 8.045 delistete Common Stocks: 641 Namen tragen mehr als
    einen Filer, und nur 66 davon lassen sich über den Zeitraum trennen. Der
    Rest sind **gleichzeitige** Co-Filer — Mutter und Finanztochter unter
    nahezu demselben Namen (`Noble Corporation plc`, fünf CIKs). Sich dort
    einen zu nehmen hiesse, mit gleicher Wahrscheinlichkeit die Bilanz der
    Holding oder die der Emissionstochter zu bekommen. Beides sieht plausibel
    aus, eines ist falsch — genau der Fehlertyp, gegen den diese Schicht steht.
    """
    # Lokal importiert wie `ticker_to_cik` weiter unten: `edgar` zieht httpx
    # nach, und resolve.py wird auch dort geladen, wo kein Netz im Spiel ist.
    from quantrace.providers.edgar import normalise_company_name

    schluessel = normalise_company_name(name)
    if not schluessel:
        return None, "Name ergibt keinen vergleichbaren Schlüssel."

    if history is None:
        treffer = _history_for(schluessel)
    else:
        treffer = history[history["name_norm"].astype(str) == schluessel]
    if treffer.empty:
        return None, (
            f"Kein SEC-Filer unter '{schluessel}'. Fehlt die Karte im Lake, baut "
            "`python scripts/build_cik_map.py --zip submissions.zip` sie."
        )

    stichtag = as_of.isoformat()
    passend = treffer[
        treffer["valid_from"].isna() | (treffer["valid_from"].astype(str).str[:10] <= stichtag)
    ]
    passend = passend[
        passend["valid_to"].isna() | (passend["valid_to"].astype(str).str[:10] >= stichtag)
    ]
    if passend.empty:
        return None, (
            f"'{schluessel}' ist bei der SEC bekannt, aber zu keinem Zeitpunkt "
            f"um den {as_of} — die Namensfenster liegen daneben."
        )

    kandidaten = sorted({str(c) for c in passend["cik"]})
    if len(kandidaten) > 1:
        return None, (
            f"{len(kandidaten)} Filer trugen am {as_of} den Namen '{schluessel}' "
            f"({', '.join(kandidaten[:4])}…). Meist Mutter und Finanztochter, die "
            "gemeinsam einreichen. Einen davon zu wählen wäre geraten."
        )

    zeile = passend.iloc[0]
    return str(kandidaten[0]), (
        f"Über den Firmennamen aufgelöst: '{zeile['name']}' war am {as_of} "
        f"bei der SEC unter CIK {kandidaten[0]} eingetragen."
    )


@dataclass(frozen=True)
class CikResolution:
    """Wem gehörte dieser Ticker am Stichtag — und wie sicher wissen wir das?

    ``status``
        ``confirmed``
            Der Stichtag fällt in das **aktive** Segment des Codes; die
            Heute-Karte beschreibt also dieselbe Firma.
        ``recycled``
            Der Stichtag fällt in ein **abgelöstes** Segment. Die Heute-Karte
            zeigt auf die Nachfolgefirma — ``cik`` bleibt ``None``.
        ``unknown``
            Kein CIK für diesen Ticker. Meist eine Firma, die nicht mehr
            existiert und deren Kürzel niemand übernommen hat (LEH).
        ``unverified``
            Schicht 2 ist nicht gebaut oder kennt den Code nicht — die
            Recycling-Prüfung war nicht möglich. Der CIK steht trotzdem da,
            aber als *ungeprüfte* Angabe.
    """

    code: str
    as_of: date
    cik: str | None
    status: str
    reason: str

    @property
    def usable(self) -> bool:
        return self.cik is not None and self.status in USABLE_CIK_STATUS


def _cik_ueber_namen(
    code: str,
    as_of: date,
    name_lookup: Callable[[str, date], str | None] | None,
    history: pd.DataFrame | None,
) -> CikResolution | None:
    """Der Namensweg — greift **nur**, wo der Ticker-Weg schweigt (#256).

    Gibt ``None`` zurück, wenn er nichts beitragen kann; dann bleibt es beim
    bisherigen Verhalten. Er kann den Ticker-Weg also nie überstimmen, nur eine
    Lücke füllen.

    Gemessen am 2026-08-15: über 8.045 delistete Common Stocks löst er 67,6 %
    eindeutig auf, wo heute **null** aufgelöst werden. Die 8 % mehrdeutigen
    bleiben ungelöst, und das ist Absicht — `cik_by_name` erklärt, warum ein
    Ratespiel dort teurer wäre als eine Lücke.

    ``resolved_by_name`` ist bewusst ein eigener Status und nicht ``confirmed``:
    die Zuordnung stammt aus einem Namensabgleich, nicht aus SECs eigener
    Ticker-Karte. Wer das Ergebnis später anzweifelt, soll am Status sehen,
    welchen Weg es genommen hat — dieselbe Disziplin wie ``identity_source``
    und ``Adjustment.status``.
    """
    if name_lookup is None:
        return None
    name = name_lookup(code, as_of)
    if not name:
        return None

    cik, begruendung = cik_by_name(name, as_of, history=history)
    if cik is None:
        log.debug("Namensweg für %s am %s ohne Ergebnis: %s", code, as_of, begruendung)
        return None

    return CikResolution(
        code=code, as_of=as_of, cik=cik, status="resolved_by_name", reason=begruendung
    )


def resolve_cik(
    code: str,
    as_of: date,
    *,
    cik_lookup: Callable[[str], str | None] | None = None,
    manifest: pd.DataFrame | None = None,
    name_lookup: Callable[[str, date], str | None] | None = None,
    cik_history: pd.DataFrame | None = None,
) -> CikResolution:
    """CIK eines Tickers **zum Stichtag** — oder die Begründung, warum keiner.

    ``cik_lookup`` ist injizierbar, damit Tests ohne Netz auskommen; per
    Vorgabe ist es ``edgar.ticker_to_cik``.
    """
    code = code.strip().upper()

    if cik_lookup is None:
        from quantrace.providers.edgar import ticker_to_cik

        cik_lookup = ticker_to_cik

    karte = read_manifest() if manifest is None else manifest
    segmente = karte[karte["code"].astype(str).str.upper() == code] if not karte.empty else karte

    # Der gefährliche Fall zuerst: liegt der Stichtag in einem abgelösten
    # Segment, ist die Heute-Karte über eine andere Firma. Erst gar keinen CIK
    # nachschlagen — eine Zahl, die man nicht benutzen darf, lädt zum Benutzen
    # ein.
    if not segmente.empty:
        passend = segmente[(segmente["first"] <= as_of) & (segmente["last"] >= as_of)]
        if not passend.empty and bool(passend.iloc[0]["superseded"]):
            zeile = passend.iloc[0]
            ueber_namen = _cik_ueber_namen(code, as_of, name_lookup, cik_history)
            if ueber_namen is not None:
                return ueber_namen
            return CikResolution(
                code=code,
                as_of=as_of,
                cik=None,
                status="recycled",
                reason=(
                    f"Am {as_of} gehörte '{code}' zu einem später abgelösten "
                    f"Segment ({zeile['first']}…{zeile['last']}, "
                    f"{zeile['instrument']}). SECs Ticker-Karte beschreibt die "
                    "Nachfolgefirma — deren Bilanz auf diesen Kursen wäre eine "
                    "erfundene Zuordnung."
                ),
            )

    cik = cik_lookup(code)
    if cik is None:
        ueber_namen = _cik_ueber_namen(code, as_of, name_lookup, cik_history)
        if ueber_namen is not None:
            return ueber_namen
        return CikResolution(
            code=code,
            as_of=as_of,
            cik=None,
            status="unknown",
            reason=(
                f"Kein CIK für '{code}' in SECs Ticker-Karte. Die Karte führt nur "
                "aktuelle Filer; eine Firma, die vor dem Stichtag verschwand und "
                "deren Kürzel niemand übernahm, steht dort nicht mehr."
            ),
        )

    if segmente.empty:
        return CikResolution(
            code=code,
            as_of=as_of,
            cik=cik,
            status="unverified",
            reason=(
                f"Schicht 2 kennt '{code}' nicht — ob das Kürzel bis heute "
                "derselben Firma gehört, ist damit ungeprüft. "
                "'python scripts/build_resolved.py --all' beantwortet das."
            ),
        )

    return CikResolution(
        code=code,
        as_of=as_of,
        cik=cik,
        status="confirmed",
        reason=f"'{code}' gehört am {as_of} zum aktiven Segment.",
    )


__all__ = [
    "CIK_HISTORY_PATH",
    "DEFAULT_ACTIVE_TOLERANCE_DAYS",
    "DEFAULT_GAP_TRADING_DAYS",
    "MANIFEST_PATH",
    "RESOLVED_PREFIX",
    "USABLE_CIK_STATUS",
    "AmbiguousCodeError",
    "CikResolution",
    "IdentityMap",
    "PruneRefusedError",
    "Segment",
    "build_identity_map",
    "cik_by_name",
    "code_calendar",
    "find_codes",
    "materialise",
    "materialised_keys",
    "prune_stale",
    "read_manifest",
    "resolve_cik",
    "resolve_symbols",
    "segment_codes",
    "segments_from_lake",
    "stale_keys",
    "write_manifest",
]
