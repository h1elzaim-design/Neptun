"""EODHD-Provider — Wertpapier-Stammdaten, geschlüsselt nach ISIN.

**Warum dieses Modul mit der Identität anfängt und nicht mit Kursen.**

Am 2026-08-10 gegen die echte API geprüft:

    tot     BBBY_old   Bed Bath & Beyond Inc     ISIN US0758961009
    tot     BBBYQ      Bed Bath & Beyond Inc.    ISIN US0758961009
    aktiv   BBBY       Bed Bath & Beyond, Inc.   ISIN US6903701018   ← andere Firma

Overstock.com kaufte die Marke aus der Insolvenzmasse und benannte sich um. Die
drei Einträge unterscheiden sich **um ein Komma**. Wer nach dem Ticker `BBBY`
fragt, bekommt Overstocks Kurshistorie ab 2002 — lückenlos, ohne Absturz im
April 2023, und damit als „Bed Bath & Beyond überlebte die Insolvenz" lesbar.

Genau das ist beim ersten Sondenlauf passiert.

Der Schaden wäre nicht bloß eine fehlende Pleite: der Totalverlust wird durch
den Kursverlauf einer fremden Firma **ersetzt**. Ein Backtest verbucht dort
einen Gewinn, wo Anleger alles verloren. Und weil `ticker → CIK` bei EDGAR nach
heutigem Stand mappt, lägen obendrein die Bilanzzahlen der Nachfolgefirma auf
den Kursen der alten. Zwei Fehlzuordnungen, die sich gegenseitig plausibel
machen, und keine wirft eine Fehlermeldung.

**Weder Ticker noch Name taugen als Schlüssel.** Hier hätte auch ein
Namensabgleich versagt. Die ISIN trennt sie, weil sie einem Wertpapier gehört
und nicht einem Platzschild.

EODHD selbst macht es richtig — die Tote liegt unter eigenen Codes mit eigener
ISIN. Der Fehler entsteht erst beim Zugriff. Dieses Modul macht den richtigen
Zugriff zum einfachen.

Braucht ``EODHD_API_KEY``. Der EOD-Tarif (19,99 €/Mon.) reicht; die beiden
Symbollisten liegen unter „Exchanges List API Data".
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx
import pandas as pd

from quantrace.instruments import (
    US_DIVIDENDS_PREFIX,
    US_EQUITY_PREFIX,
    US_SPLITS_PREFIX,
)

log = logging.getLogger(__name__)

_BASE = "https://eodhd.com/api"
_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0  # Sekunden: 2, 4, 8, 16

#: EODHD hängt an abgelöste Codes ein Suffix. Beide zeigen auf dieselbe ISIN,
#: sind also keine zwei Wertpapiere — nur zwei Schreibweisen desselben.
_SUPERSEDED_SUFFIXES = ("_old", "-OLD")


class QuotaExceededError(RuntimeError):
    """Tageskontingent erschöpft (HTTP 402) — weiterer Abruf wäre sinnlos.

    Ein Bulk-Request kostet 100 API-Calls; das Default-Limit ist 100.000/Tag
    (UTC-Mitternacht). Der Loader bricht bei dieser Ausnahme den ganzen Lauf ab,
    statt hunderte Tage mit demselben 402 zu verbrennen.
    """


def _api_key() -> str:
    key = os.environ.get("EODHD_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "EODHD_API_KEY fehlt. Token im EODHD-Dashboard unter 'API token'; "
            "der EOD-Tarif genügt für die Symbollisten."
        )
    return key


def _get_with_retry(client: httpx.Client, path: str, params: dict[str, Any]) -> Any:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.get(f"{_BASE}/{path}", params={**params, "api_token": _api_key()})
            if resp.status_code == 402:
                # Payment Required = Tageslimit (oder Abo). Kein Retry.
                raise QuotaExceededError(
                    f"EODHD {path}: HTTP 402 — Tageskontingent erschöpft "
                    f"(Bulk = 100 Calls/Request, Limit typisch 100.000/Tag, Reset UTC). "
                    f"Morgen fortsetzen; bereits geladene Tage bleiben im Lake."
                )
            if resp.status_code in (401, 403):
                # Kein Retry: das wird beim vierten Versuch nicht besser, und
                # jeder weitere Aufruf kostet Kontingent für nichts.
                raise RuntimeError(
                    f"EODHD {path}: HTTP {resp.status_code} — Token oder Tarif deckt das nicht ab."
                )
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = _BACKOFF_BASE**attempt
                log.warning("EODHD %s (status %s) — retry in %.0fs", path, resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except QuotaExceededError:
            raise
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            wait = _BACKOFF_BASE**attempt
            last_exc = exc
            log.warning("EODHD %s Netzfehler (%s) — retry in %.0fs", path, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"EODHD nicht erreichbar nach {_MAX_RETRIES} Versuchen: {path}") from last_exc


# ---------------------------------------------------------------------------
# Stammdaten


@dataclass(frozen=True)
class Security:
    """Ein Wertpapier — mit ISIN, wo es eine gibt.

    **``isin`` darf leer sein, und das ist eine Aussage, kein Datenfehler**
    (#297). Rund drei Viertel der toten US-Stammaktien tragen keine; wer sie
    verwirft, verwirft genau die Firmen, derentwegen der survivorship-freie
    Bulk überhaupt benutzt wird. Ob ein Eintrag eine hat, sagt
    ``identity_source`` — die schwächere Zusage wird markiert, nicht versteckt,
    dieselbe Disziplin wie bei ``resolve.identity_source`` und
    ``Adjustment.status``.

    Wer eine ISIN *braucht*, holt die Liste mit ``require_isin=True`` — dann
    ist ein leeres Feld ausgeschlossen, statt später aufzufallen.
    """

    isin: str
    code: str  # Ticker OHNE Börsensuffix, z.B. "AAPL"
    exchange: str
    name: str
    type: str
    currency: str
    #: False = delistet. Kommt aus der Liste, in der der Eintrag stand, nicht
    #: aus einem Feld — EODHD führt Tote und Lebende getrennt.
    active: bool

    @property
    def symbol(self) -> str:
        """Abfragekürzel für die Kurs-API, z.B. ``AAPL.US``."""
        return f"{self.code}.{self.exchange}"

    @property
    def superseded(self) -> bool:
        """Ein abgelöster Code (``BBBY_old``) — dieselbe ISIN, alte Schreibweise."""
        return self.code.endswith(_SUPERSEDED_SUFFIXES)

    @property
    def identity_source(self) -> str:
        """``isin`` oder ``code`` — worauf sich die Identität dieses Eintrags stützt.

        ``code`` heißt: der Eintrag beantwortet „gab es dieses Kürzel je an
        dieser Börse?" und sonst nichts. Er taugt als Zugehörigkeitstest, aber
        nicht als Schlüssel über die Zeit — dafür ist ADR-013 zuständig.
        """
        return "isin" if self.isin else "code"


def fetch_symbol_list(
    exchange: str = "US", *, delisted: bool = False, require_isin: bool = True
) -> list[Security]:
    """Symbolliste einer Börse. ``delisted=True`` liefert **nur** die Toten.

    Am 2026-08-10 geprüft: `delisted=1` enthält AAPL nicht, LEH schon — es ist
    also keine Gesamtliste, sondern der Friedhof. Für die USA: 59.190 tote
    gegen 51.650 lebende Einträge. Mehr als die Hälfte aller je gelisteten
    US-Ticker existiert nicht mehr.

    **Zwei Aufrufer, zwei Ansprüche** (#297). ``require_isin`` entscheidet,
    welcher gilt:

    ``True`` (Vorgabe)
        Für **Identität**. Ein Eintrag ohne stabilen Schlüssel ist keine
        Identität, und einer, der sie vortäuscht, ist schlechter als keiner —
        die Disziplin aus ADR-013. So liest ``build_instrument_map.py``.

    ``False``
        Für einen **Katalog**, dessen Frage „gab es dieses Kürzel je an dieser
        Börse?" lautet. Die beantwortet auch ein Eintrag ohne ISIN, und ihn zu
        verwerfen kostet genau die toten Firmen: LEH, BSC, MOT, KFT fehlen
        heute im Katalog, obwohl EODHD sie führt. Rund drei Viertel der toten
        US-Stammaktien haben keine ISIN — der Filter trifft also überwiegend
        die Seite, um derentwillen der Bulk benutzt wird. Wer so liest, muss
        ``Security.identity_source`` mitführen, sonst sieht die schwächere
        Zusage aus wie die starke.

    Die Vorgabe ist die strengere, weil nur einer der beiden Fehler teuer ist —
    dieselbe Asymmetrie wie beim ``purpose``-Default in ADR-014.
    """
    params: dict[str, Any] = {"fmt": "json"}
    if delisted:
        params["delisted"] = 1
    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        rows = _get_with_retry(client, f"exchange-symbol-list/{exchange}", params)

    out: list[Security] = []
    ohne_isin = 0
    for r in rows if isinstance(rows, list) else []:
        isin = str(r.get("Isin") or "").strip()
        code = str(r.get("Code") or "").strip()
        if not code:
            continue
        if not isin:
            # Warrants, Rechte, manche OTC-Papiere haben keine — und ein großer
            # Teil der toten Stammaktien. Ob sie durchkommen, entscheidet der
            # Aufrufer; gezählt werden sie in jedem Fall.
            ohne_isin += 1
            if require_isin:
                continue
        out.append(
            Security(
                isin=isin,
                code=code,
                exchange=str(r.get("Exchange") or exchange).strip(),
                name=str(r.get("Name") or "").strip(),
                type=str(r.get("Type") or "").strip(),
                currency=str(r.get("Currency") or "").strip(),
                active=not delisted,
            )
        )
    log.info(
        "EODHD %s (%s): %d Wertpapiere, davon %d ohne ISIN (%s).",
        exchange,
        "delistet" if delisted else "aktiv",
        len(out),
        ohne_isin,
        "übersprungen" if require_isin else "mit identity_source='code' geführt",
    )
    return out


@dataclass
class SecurityMaster:
    """Lebende und tote Wertpapiere einer Börse, nach ISIN geschlüsselt."""

    securities: list[Security]

    @classmethod
    def load(cls, exchange: str = "US", *, require_isin: bool = True) -> SecurityMaster:
        """Beide Listen holen und zusammenlegen. Zwei Requests.

        ``require_isin`` wird an ``fetch_symbol_list`` durchgereicht — die
        Vorgabe bleibt die strenge. Zur Bedeutung siehe dort (#297).
        """
        return cls(
            fetch_symbol_list(exchange, delisted=False, require_isin=require_isin)
            + fetch_symbol_list(exchange, delisted=True, require_isin=require_isin)
        )

    # -- Nachschlagen ------------------------------------------------------

    def by_isin(self, isin: str) -> list[Security]:
        """Alle Einträge zu einer ISIN — meist einer, bei abgelösten Codes mehrere.

        Ein leerer Suchbegriff findet **nichts**, statt alle ISIN-losen
        Einträge zurückzugeben: „keine ISIN" ist keine ISIN, und ein Treffer
        auf `""` wäre der Fehlertyp, gegen den ADR-013 gebaut ist.
        """
        if not isin:
            return []
        return [s for s in self.securities if s.isin == isin]

    def by_code(self, code: str) -> list[Security]:
        """Alle Wertpapiere, die dieses Kürzel je trugen.

        Gibt bewusst eine **Liste** zurück, nicht einen Treffer. Ein Ticker ist
        keine Identität; wer hier einen einzelnen Wert erwartet, hat das
        Problem noch nicht verstanden, das dieses Modul löst.
        """
        return [s for s in self.securities if s.code == code]

    def resolve(self, code: str, *, prefer_active: bool = True) -> Security | None:
        """Ein Kürzel auf **ein** Wertpapier auflösen — mit Warnung bei Mehrdeutigkeit.

        Für den Fall, dass man wirklich nur einen Ticker hat. Trägt das Kürzel
        mehrere ISINs, wird das geloggt: die Auswahl ist dann eine Annahme, und
        Annahmen gehören sichtbar gemacht.
        """
        treffer = self.by_code(code)
        if not treffer:
            return None
        # Leere ISINs zählen hier nicht mit: sie sagen „unbekannt", nicht
        # „ein anderes Papier". Sonst meldete jeder ISIN-lose Eintrag neben
        # einem belegten eine Mehrdeutigkeit, die es nicht gibt (#297).
        isins = {s.isin for s in treffer if s.isin}
        if len(isins) > 1:
            log.warning(
                "Kürzel %r trug %d verschiedene Wertpapiere (%s) — löse auf %s auf.",
                code,
                len(isins),
                ", ".join(sorted(isins)),
                "das aktive" if prefer_active else "das erste",
            )
        if prefer_active:
            aktive = [s for s in treffer if s.active]
            if aktive:
                return aktive[0]
        return treffer[0]

    # -- Befund ------------------------------------------------------------

    def recycled_codes(self) -> dict[str, list[Security]]:
        """Kürzel, die nacheinander **verschiedenen** Wertpapieren gehörten.

        Der Kern des Problems, als Zahl. Jeder Eintrag hier ist ein Ticker, bei
        dem eine Abfrage nach Kürzel die falsche Firma treffen kann.

        Abgelöste Schreibweisen desselben Papiers (``BBBY_old``) zählen nicht
        mit — die tragen dieselbe ISIN und sind kein Identitätswechsel.

        **ISIN-lose Einträge zählen ebenfalls nicht mit** (#297). Sie belegen
        keinen Wechsel, sondern eine Wissenslücke, und als eigener „Wert" im
        Vergleich hätten sie jeden Code mit einem solchen Eintrag als recycelt
        gemeldet. Diese Zahl soll ein Befund bleiben, kein Rauschmaß — was der
        Filter dadurch *nicht* sieht, ist bekannt und in #297 benannt.
        """
        nach_code: dict[str, list[Security]] = defaultdict(list)
        for s in self.securities:
            nach_code[s.code].append(s)
        return {
            code: sec
            for code, sec in nach_code.items()
            if len({s.isin for s in sec if s.isin}) > 1
        }

    def summary(self) -> dict[str, int]:
        aktiv = sum(1 for s in self.securities if s.active)
        recycelt = self.recycled_codes()
        return {
            "wertpapiere": len(self.securities),
            "aktiv": aktiv,
            "delistet": len(self.securities) - aktiv,
            "eindeutige_isins": len({s.isin for s in self.securities if s.isin}),
            # Die Gegenzahl gehört daneben, nicht in eine Fußnote: ohne sie
            # liest sich `eindeutige_isins` als Abdeckung des Katalogs.
            "ohne_isin": sum(1 for s in self.securities if not s.isin),
            "recycelte_kuerzel": len(recycelt),
            "abgeloeste_schreibweisen": sum(1 for s in self.securities if s.superseded),
        }


# ---------------------------------------------------------------------------
# Tagesquerschnitte


@dataclass(frozen=True)
class BulkFeed:
    """Ein Tagesquerschnitt des Bulk-Endpunkts.

    Drei Feeds teilen sich denselben Pfad und unterscheiden sich nur im
    ``type``-Parameter. Alle drei am 2026-08-10 gegen die echte API geprüft;
    die Spaltenlisten sind abgeschrieben, nicht vermutet.
    """

    #: Wert des ``type``-Parameters. Leer = Kurse (der Endpunkt kennt kein
    #: ``type=prices``; die Kurse sind der Default).
    param: str
    #: Lake-Wurzel. Eigene Wurzel je Feed, weil zwei Partitionsschemata unter
    #: einer Wurzel jede Hive-Abfrage brechen würden.
    prefix: str
    columns: tuple[str, ...]
    #: Spalten, die als Gleitkomma gelesen werden. Der Rest bleibt Text —
    #: insbesondere ``split``, das als ``"2.000000/1.000000"`` kommt.
    numeric: tuple[str, ...] = ()
    label: str = ""


#: Kurse. Kein ISIN-Feld — der Querschnitt kommt mit Codes, und genau deshalb
#: wird er **roh nach Datum** abgelegt statt beim Schreiben aufgelöst.
BULK_COLUMNS = (
    "code",
    "exchange_short_name",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)

#: Die drei Feeds des Bulk-Endpunkts.
#:
#: **Warum die beiden Corporate-Action-Feeds nicht optional sind.** Die
#: ``adjusted_close``-Spalte der Kurse ist zum *Abrufzeitpunkt* adjustiert.
#: Wer 1996–2010 heute lädt und 2026 im nächsten Jahr, bekommt zwei
#: Zeitabschnitte mit verschiedenen Split-Faktoren aneinandergeklebt: an
#: jedem Split dazwischen springt die Reihe. Der Sprung sieht aus wie eine
#: Rendite und ist keine. Mit gespeicherten Actions wird die Adjustierung
#: eine Funktion der Daten statt eine Funktion des Abrufdatums.
#:
#: Dazu der zweite Grund, der schwerer wiegt: **Dividenden sind kein
#: Beiwerk, sondern ein Teil der Rendite.** Eine reine Kursreihe unterschätzt
#: die Gesamtrendite systematisch bei genau den Titeln, die eine
#: Langfrist-Plattform prüfen würde. Und ``declarationDate`` sagt, wann die
#: Zahlung öffentlich wurde — das Ex-Datum sagt nur, wann sie den Kurs traf.
BULK_FEEDS: dict[str, BulkFeed] = {
    "prices": BulkFeed(
        param="",
        prefix=US_EQUITY_PREFIX,
        columns=BULK_COLUMNS,
        numeric=("open", "high", "low", "close", "adjusted_close", "volume"),
        label="Kurse",
    ),
    "splits": BulkFeed(
        param="splits",
        prefix=US_SPLITS_PREFIX,
        # `exchange`, nicht `exchange_short_name` — die API ist hier
        # uneinheitlich. Nicht angleichen: was ankommt, wird abgelegt.
        columns=("code", "exchange", "date", "split"),
        label="Splits",
    ),
    "dividends": BulkFeed(
        param="dividends",
        prefix=US_DIVIDENDS_PREFIX,
        columns=("code", "exchange", "date", "dividend", "currency", "declarationDate"),
        numeric=("dividend",),
        label="Dividenden",
    ),
}


def parse_split_ratio(raw: object) -> float | None:
    """``"2.000000/1.000000"`` → ``2.0``. ``None``, wenn unlesbar.

    Der Kursfaktor eines Splits, nicht das Verhältnis als Text. Bewusst
    **beim Lesen** und nicht beim Schreiben: im Lake liegt, was ankam.

    ``None`` statt ``1.0`` bei Unlesbarem. Ein nicht interpretierbarer Split
    stillschweigend als „kein Split" zu behandeln, wäre derselbe Fehler wie
    fehlende Kosten als 0,0 zu lesen — die Falle, in die `_score_realism`
    schon einmal gelaufen ist.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        if "/" in text:
            zaehler, nenner = text.split("/", 1)
            n = float(nenner)
            return float(zaehler) / n if n else None
        return float(text)
    except (TypeError, ValueError):
        return None


def fetch_bulk_day(day: date, exchange: str = "US", kind: str = "prices") -> pd.DataFrame:
    """Ein Tagesquerschnitt einer Börse. Ein Request.

    **Warum das der wichtigste Endpunkt des Tarifs ist.** Am 2026-08-10 gegen
    den 2008-09-12 geprüft — den letzten Handelstag vor Lehmans Insolvenz:
    26.090 Symbole, darunter LEH, BSC, WM und AIG. Der Querschnitt zeigt den
    Markt **wie er an dem Tag war**, nicht die heutigen Überlebenden mit
    historischen Kursen.

    Damit ist er die Grundlage für survivorship-freie Universen — und das
    ISIN-Loch (drei Viertel der toten Stammaktien haben keine) spielt dabei
    keine Rolle: innerhalb eines Tages ist ein Code eindeutig. Mehrdeutig wird
    er erst über die Zeit, und das ist ein Problem der Verkettung, nicht der
    Erhebung.

    Kosten: rund 7.560 Requests je Feed für 30 Jahre gegen 110.840 bei Abruf
    pro Symbol. Der Tarif erlaubt 100.000 am Tag.

    Leerer Frame an handelsfreien Tagen — kein Fehler, sondern die Auskunft
    „an diesem Tag wurde nicht gehandelt". Bei den Corporate Actions ist leer
    sogar der Normalfall: gemessen 2 Splits und 154 Dividenden an einem Tag.
    """
    try:
        feed = BULK_FEEDS[kind]
    except KeyError:
        raise ValueError(
            f"Unbekannter Feed {kind!r} — bekannt: {', '.join(sorted(BULK_FEEDS))}"
        ) from None

    params: dict[str, Any] = {"fmt": "json", "date": day.isoformat()}
    if feed.param:
        params["type"] = feed.param
    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        rows = _get_with_retry(client, f"eod-bulk-last-day/{exchange}", params)

    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=list(feed.columns))

    df = pd.DataFrame(rows)
    fehlend = [c for c in feed.columns if c not in df.columns]
    if fehlend:
        # Ein stillschweigend fehlendes Feld wäre eine Spalte voller NaN im
        # Lake — lieber laut sein, solange nur ein Tag betroffen ist.
        log.warning(
            "Bulk %s %s: Spalten fehlen in der Antwort: %s", kind, day, ", ".join(fehlend)
        )
        for c in fehlend:
            df[c] = pd.NA

    df = df[list(feed.columns)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for c in feed.numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # volume bleibt float. EODHD liefert an manchen Tagen Bruchteile
    # (gemessen 706/26.090 am 2008-09-12, z.B. 22321.80). Ein erzwungenes
    # Int64 mit safe-cast wirft den ganzen Tag weg — schlimmer als ein
    # nicht-ganzzahliges Volume im Lake. Schicht 1 speichert, was ankommt.
    return df
