"""SEC-EDGAR-Provider — Fundamentaldaten, die von sich aus point-in-time sind.

Warum das hier steht: bis jetzt hat QuantRace **keine** Fundamentaldaten. Der
naheliegende Weg wäre ein kommerzieller Anbieter — nur liefern die fast alle
den *heutigen* Stand einer Bilanz. Eine 2019 nachträglich korrigierte Zahl kommt
dort in der korrigierten Fassung, und ein Backtest über 2019 kennt damit etwas,
das es damals nicht gab. Dieselbe Fehlerklasse wie ``adjusted_close``, nur eine
Ebene höher und schwerer zu bemerken.

EDGAR hat dieses Problem nicht — **weil jede Zahl ihr Einreichungsdatum
mitbringt**:

    {"end": "2016-09-24", "val": 215639000000, "filed": "2018-11-05", "form": "10-K"}

Point-in-time ist damit keine Zusatzleistung, die man kaufen muss, sondern eine
Filterbedingung: ``filed <= Analysetag``. Genau das macht `as_of`.

**Restatements sind der eigentliche Grund für dieses Modul.** Derselbe Zeitraum
taucht mehrfach auf — einmal wie ursprünglich gemeldet, dann wie später
korrigiert. Wer naiv den letzten Eintrag nimmt, holt sich die Korrektur in einen
Backtest, der Jahre davor liegt. `as_of` löst das explizit: unter allen
Einreichungen bis zum Stichtag gewinnt pro Periode die **jüngste** — das ist,
was man an dem Tag geglaubt hätte, nicht was heute stimmt.

Grenzen, damit sie niemand suchen muss:

* **Nur US-Filer.** Europa läuft über ESEF/`filings.xbrl.org`, ein eigener Bau.
* **Nur was in XBRL steht**, also im Wesentlichen ab 2009 verlässlich.
* Die Konzepte heißen nicht überall gleich. Apple bucht Umsatz seit ASC 606
  unter ``RevenueFromContractWithCustomerExcludingAssessedTax``, davor unter
  ``Revenues`` und ``SalesRevenueNet``. `concept_series` führt diese Tags zu
  **einer** Reihe zusammen — sonst begänne die Umsatzhistorie 2019.

Kein API-Key nötig. Die SEC verlangt aber einen **User-Agent mit Kontakt**
(``SEC_USER_AGENT``) und begrenzt auf 10 Requests/Sekunde — beides ist hier
umgesetzt, ein Verstoß führt zu Sperren für die ganze IP.

Doku: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any

import httpx

log = logging.getLogger(__name__)

_BASE = "https://data.sec.gov"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0  # Sekunden: 2, 4, 8, 16

#: Die SEC erlaubt 10 Requests/Sekunde. Wir bleiben bewusst darunter — die Strafe
#: für Überschreiten ist eine IP-Sperre, und die träfe auch den Worker.
_MIN_INTERVAL_S = 0.15
_rate_lock = threading.Lock()
_last_request_at = 0.0

#: Rangfolge, keine Aufzählung: alle Tags werden zusammengeführt, aber bei
#: derselben Beobachtung gewinnt das früher gelistete. Die ASC-606-Konzepte
#: stehen vorn, weil sie die aktuelle Praxis abbilden.
CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    # --- Gewinn- und Verlustrechnung ---------------------------------------
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "cost_of_revenue": (
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfServices",
    ),
    #: Direkt gemeldet, wo vorhanden. Wer ihn aus `revenue - cost_of_revenue`
    #: rechnet, mischt zwei Reihen mit womöglich verschiedenen Einreichungs-
    #: zeitpunkten — deshalb hier ein eigenes Konzept statt einer Ableitung.
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "rnd_expense": ("ResearchAndDevelopmentExpense",),
    "sga_expense": (
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ),
    "interest_expense": ("InterestExpense", "InterestExpenseDebt"),
    "pretax_income": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ),
    "income_tax": ("IncomeTaxExpenseBenefit",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "eps_basic": ("EarningsPerShareBasic",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    # --- Bilanz -------------------------------------------------------------
    "assets": ("Assets",),
    "current_assets": ("AssetsCurrent",),
    "inventory": ("InventoryNet",),
    "ppe_net": ("PropertyPlantAndEquipmentNet",),
    "liabilities": ("Liabilities",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "short_term_debt": ("DebtCurrent", "ShortTermBorrowings", "LongTermDebtCurrent"),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "shares_outstanding": ("CommonStockSharesOutstanding", "dei:EntityCommonStockSharesOutstanding"),
    # --- Kapitalflussrechnung ----------------------------------------------
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "investing_cash_flow": ("NetCashProvidedByUsedInInvestingActivities",),
    "financing_cash_flow": ("NetCashProvidedByUsedInFinancingActivities",),
    "depreciation_amortization": (
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ),
    "capex": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
    "dividends_paid": ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends"),
    "buybacks": ("PaymentsForRepurchaseOfCommonStock",),
}

#: Die XBRL-Einheit je Kennzahl. Alles ohne Eintrag ist ``USD``.
#:
#: **Warum das eine eigene Karte braucht.** `companyfacts` schlüsselt jede
#: Reihe nach Einheit, und wer unter der falschen nachsieht, findet nichts —
#: keinen Fehler, eine leere Liste. Am 2026-08-13 gegen 23 Filer geprüft
#: (`scripts/audit_edgar_concepts.py`): `eps_basic`, `eps_diluted` und
#: `shares_outstanding` lieferten bei **null von 23** einen Wert, weil
#: `concept_series` fest ``USD`` annahm. Die drei Kennzahlen existierten seit
#: ihrer Aufnahme am 2026-08-12 nur auf dem Papier.
#:
#: Das ist derselbe teure Fehlertyp wie an drei anderen Stellen im Projekt: er
#: sieht aus wie „diese Firma meldet das eben nicht", nicht wie ein Defekt. Und
#: er hätte jede fundamentale Cross-Section auf EPS still leer gelassen.
#:
#: Die Einheit gehört deshalb zur **Kennzahl**, nicht zum Aufruf — genauso wie
#: die Periode in `DEFAULT_PERIODS`. Ein Aufrufer, der sie übergeben müsste,
#: müsste XBRL kennen; das ist genau die Kenntnis, die dieses Modul kapselt.
CONCEPT_UNITS: dict[str, str] = {
    "eps_basic": "USD/shares",
    "eps_diluted": "USD/shares",
    "shares_outstanding": "shares",
}

#: Tags, die **Komponenten** derselben Kennzahl sind statt Alternativen.
#:
#: Der Befund aus dem Prüflauf vom 2026-08-13 (#257): bei `cost_of_revenue`
#: weichen **114 von 114** Konflikten zwischen ``CostOfGoodsSold`` und
#: ``CostOfServices`` im Wert ab. Kein Wunder — das sind keine zwei Namen für
#: dieselbe Zahl, das sind zwei Summanden. Eine Firma, die Waren *und*
#: Dienstleistungen getrennt ausweist, hat beide, und eine Rangfolge nimmt einen
#: davon und nennt ihn „Cost of Revenue".
#:
#: Der Fehler ist der leise: der Wert ist **zu klein**, nicht falsch benannt.
#: Eine zu kleine Kostenzahl macht die Bruttomarge zu gut, und zwar genau bei
#: den gemischten Geschäftsmodellen — nicht zufällig verteilt, sondern
#: systematisch.
#:
#: **Warum das Summieren hier zulässig ist und bei `gross_profit` nicht.** Der
#: Einwand gegen Ableitungen lautet: zwei Reihen mit womöglich verschiedenen
#: Einreichungszeitpunkten zu mischen erzeugt eine Zahl, die es so nie gab.
#: Summiert wird deshalb ausschliesslich innerhalb **einer Einreichung** —
#: gleiche Periode, gleiches `filed`, gleiche `accession`. Das ist ein einziges
#: Dokument; ein Zeitversatz kann dort nicht entstehen. `gross_profit` bliebe
#: eine Ableitung über *verschiedene* Kennzahlen und bleibt deshalb ungerechnet.
#:
#: **Nur wenn kein Summen-Tag gewonnen hat.** ``CostOfRevenue`` und
#: ``CostOfGoodsAndServicesSold`` sind selbst schon Summen und stehen in der
#: Rangfolge vorn. Wer meldet, wird genommen; addiert wird erst, wenn nur die
#: Bestandteile da sind. Sonst zählte man doppelt.
#:
#: Bewusst **nicht** hier: `sga_expense` und `depreciation_amortization`. Dort
#: sind die nachrangigen Tags *Teilmengen* des vorderen (G&A ⊂ SG&A), keine
#: Summanden — und weil das breitere Tag die Rangfolge anführt, gewinnt bereits
#: heute das Richtige.
CONCEPT_COMPONENTS: dict[str, frozenset[str]] = {
    "cost_of_revenue": frozenset({"CostOfGoodsSold", "CostOfServices"}),
}

#: US-Börsenschluss in UTC. 16:00 ET sind 20:00 UTC in der Sommerzeit und
#: 21:00 UTC im Winter — wir nehmen die **frühere** Grenze. Ein Bericht, der im
#: Januar um 20:30 UTC (= 15:30 ET, also vor Schluss) angenommen wurde, gilt
#: damit erst am Folgetag. Das ist zu streng und genau richtig: die Kosten sind
#: ein Tag Verspätung, der Fehler in die andere Richtung wäre Look-ahead.
MARKET_CLOSE_UTC_HOUR = 20

#: Formulare, die für Research zählen. 8-K ist das Ereignis-Formular (Quartalszahlen,
#: Übernahmen, Vorstandswechsel), 10-Q/10-K die periodischen Berichte.
RESEARCH_FORMS = ("8-K", "10-Q", "10-K")


# ---------------------------------------------------------------------------
# HTTP


def _user_agent() -> str:
    """Kontakt-String für die SEC. Ohne Adresse antwortet die API mit 403.

    Die SEC verlangt ausdrücklich eine erreichbare Mailadresse. Ein Default
    ohne ``@`` (GitHub-URL, Noreply) sieht gesetzt aus und wird trotzdem
    abgelehnt — genau dann bleibt die Profilseite leer, ohne dass jemand
    einen Konfigurationsfehler vermutet.
    """
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        raise RuntimeError(
            "SEC_USER_AGENT muss 'Name erreichbare@email' sein — "
            "ohne Adresse antwortet die SEC mit 403, und die "
            "Fundamentaldaten bleiben leer."
        )
    return ua


def _throttle() -> None:
    """Mindestabstand zwischen zwei Requests, prozessweit."""
    global _last_request_at
    with _rate_lock:
        delta = time.monotonic() - _last_request_at
        if delta < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - delta)
        _last_request_at = time.monotonic()


def _get_with_retry(client: httpx.Client, url: str) -> Any:
    """GET mit Backoff. Gibt den geparsten JSON-Body zurück.

    404 ist **kein** Fehler, sondern eine Antwort: nicht jeder Filer hat jedes
    Konzept. Wer das als Ausnahme behandelt, zwingt jeden Aufrufer zu einem
    try/except um eine völlig normale Auskunft.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        _throttle()
        try:
            resp = client.get(url, headers={"User-Agent": _user_agent()})
            if resp.status_code == 404:
                return None
            if resp.status_code == 403:
                raise RuntimeError(
                    "SEC 403 — User-Agent abgelehnt. SEC_USER_AGENT braucht "
                    "eine erreichbare Mailadresse, keine GitHub-Noreply."
                )
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = _BACKOFF_BASE**attempt
                log.warning("EDGAR %s (status %s) — retry in %.0fs", url, resp.status_code, wait)
                last_exc = httpx.HTTPStatusError(
                    f"status {resp.status_code}", request=resp.request, response=resp
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            wait = _BACKOFF_BASE**attempt
            last_exc = exc
            log.warning("EDGAR %s Netzfehler (%s) — retry in %.0fs", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"EDGAR nicht erreichbar nach {_MAX_RETRIES} Versuchen: {url}") from last_exc


# ---------------------------------------------------------------------------
# Ticker → CIK


_ticker_map_cache: dict[str, str] | None = None


def load_ticker_map(*, client: httpx.Client | None = None, refresh: bool = False) -> dict[str, str]:
    """Ticker (Großbuchstaben) → zehnstellige CIK.

    Die Datei ist der **aktuelle** Stand der SEC, rund 10.400 Einträge. Sie
    kennt keine historischen Ticker: ein Kürzel, das heute einer anderen Firma
    gehört, zeigt auf diese.

    **`formerNames` ist nicht die Gegenprobe** — das stand hier lange und
    stimmt nicht. Der Block führt frühere **Firmennamen** mit `from`/`to`, nicht
    frühere **Ticker**. Eine Firma, die 2008 unter `LEH` handelte und heute
    nicht mehr existiert, ist über diese Datei gar nicht erreichbar: sie steht
    weder unter ihrem Ticker (den führt die Karte nicht mehr) noch lässt sie
    sich von einem heutigen Ticker aus finden.

    Der Weg zu toten Filern führt deshalb über den **Namen** zum Stichtag —
    `scripts/build_cik_map.py` baut die Karte dafür aus SECs
    `submissions`-Bulk, der auch tote CIKs samt `formerNames` enthält. Siehe
    #256; `normalise_company_name` ist die Normalisierung, an der ein naiver
    Abgleich sonst scheitert.
    """
    global _ticker_map_cache
    if _ticker_map_cache is not None and not refresh:
        return _ticker_map_cache

    owns_client = client is None
    client = client or httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        payload = _get_with_retry(client, _TICKERS_URL) or {}
    finally:
        if owns_client:
            client.close()

    mapping: dict[str, str] = {}
    for row in payload.values() if isinstance(payload, dict) else []:
        ticker = str(row.get("ticker", "")).strip().upper()
        cik = row.get("cik_str")
        if ticker and cik is not None:
            mapping[ticker] = str(int(cik)).zfill(10)

    _ticker_map_cache = mapping
    log.info("EDGAR: %d Ticker→CIK-Zuordnungen geladen.", len(mapping))
    return mapping


def ticker_to_cik(ticker: str, *, client: httpx.Client | None = None) -> str | None:
    """Zehnstellige CIK zu einem Ticker, oder ``None``."""
    return load_ticker_map(client=client).get(ticker.strip().upper())


#: Rechtsformen und Zusätze, die eine Firma nicht *identifizieren*. Sie werden
#: am Namensende abgeschnitten, damit „Bed Bath & Beyond Inc." und „Bed Bath and
#: Beyond, Inc" denselben Schlüssel ergeben.
#:
#: Bewusst **nicht** in dieser Liste: `Holdings`, `Group`, `Trust`, `Partners`.
#: Die sehen nach Beiwerk aus, unterscheiden aber reale Filer — „XY Holdings"
#: und „XY" sind regelmässig Mutter und Tochter mit eigener CIK und eigener
#: Bilanz. Wer sie wegkürzt, verschmilzt zwei Bilanzen zu einer.
_RECHTSFORMEN = (
    "incorporated", "inc", "corporation", "corp", "company", "co",
    "limited", "ltd", "llc", "llp", "lp", "plc", "sa", "nv", "ag", "the",
)

_NICHT_ALPHANUM = re.compile(r"[^a-z0-9]+")


def normalise_company_name(name: str) -> str:
    """Firmenname → Vergleichsschlüssel. Leer, wenn nichts übrig bleibt.

    **Warum das eine eigene Funktion mit eigenen Tests ist.** Der Weg zu toten
    Filern führt über den Namen (`load_ticker_map` erklärt, warum es keinen
    anderen gibt), und ein naiver Vergleich scheitert an Kleinigkeiten: die drei
    BBBY-Einträge im EODHD-Katalog unterscheiden sich um ein **Komma**.

    Zugleich ist zu scharfes Normalisieren der teurere Fehler. Ein Schlüssel,
    der zwei verschiedene Firmen zusammenfallen lässt, liefert eine Bilanz —
    die falsche —, während ein zu enger Schlüssel nur nichts liefert. Deshalb
    fallen ausschliesslich Rechtsformen, Satzzeichen und Gross-/Kleinschreibung
    weg, und `&` wird zu `and`; alles andere bleibt stehen.

    Der Aufrufer muss trotzdem mit Mehrdeutigkeit rechnen: verschiedene Filer
    dürfen denselben Schlüssel haben, und die Karte hält sie deshalb alle.
    """
    s = name.strip().lower().replace("&", " and ")
    s = _NICHT_ALPHANUM.sub(" ", s)
    teile = [t for t in s.split() if t]
    # Rechtsformen fallen nur am **Ende** weg. „Inc" mitten im Namen ist Teil
    # des Namens, und „The Co Operative Bank" verlöre sonst sein zweites Wort.
    while teile and teile[-1] in _RECHTSFORMEN:
        teile.pop()
    while teile and teile[0] in ("the",):
        teile.pop(0)
    return " ".join(teile)


# ---------------------------------------------------------------------------
# Fakten


@dataclass(frozen=True)
class Fact:
    """Ein XBRL-Datenpunkt mit dem Datum, an dem er öffentlich wurde."""

    concept: str
    period_start: date | None
    period_end: date
    value: float
    filed: date
    form: str
    accession: str
    unit: str
    #: True, wenn für dieselbe Periode eine frühere Einreichung mit einem
    #: **anderen** Wert existiert — die Zahl wurde also nachträglich korrigiert.
    restated: bool = False
    #: Die Summanden, wenn dieser Wert aus mehreren Tags **einer** Einreichung
    #: addiert wurde (siehe `CONCEPT_COMPONENTS`). Leer heißt: genau ein Tag
    #: gemeldet — der Normalfall. Eine zusammengesetzte Zahl soll nicht
    #: aussehen wie eine gemeldete.
    components: tuple[str, ...] = ()
    #: Annahmezeit der Einreichung (SEC `acceptanceDateTime`), falls bekannt.
    #: Steht **nicht** in `companyfacts` — nur in `submissions`, und dort nur
    #: für die jüngsten ~1000 Einreichungen. `None` heißt „unbekannt", nicht
    #: „gleichzeitig mit filed".
    accepted_at: datetime | None = None

    @property
    def usable_from(self) -> date:
        """Ab wann die Zahl in einem Backtest benutzt werden darf.

        `filed` ist ein **Datum ohne Uhrzeit**, und 86 % der 10-K/10-Q werden
        nach Börsenschluss angenommen (gemessen über AAPL, MSFT, JNJ, XOM, JPM:
        93 von 108). Wer `filed <= t` filtert, gibt einer Strategie also in der
        Mehrheit der Fälle Information, die am Tag t nicht handelbar war.

        Mit bekannter Annahmezeit wird genau unterschieden. Ohne — für ältere
        Einreichungen, die nicht mehr im `recent`-Block stehen — gilt der
        Folgetag. Höchstens ein Tag zu spät, nie einen zu früh.
        """
        if self.accepted_at is not None and self.accepted_at.hour < MARKET_CLOSE_UTC_HOUR:
            return self.filed
        return self.filed + timedelta(days=1)


def _parse_date(raw: object) -> date | None:
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def fetch_company_facts(cik: str, *, client: httpx.Client | None = None) -> dict[str, Any] | None:
    """Alle XBRL-Fakten eines Filers in einem Abruf.

    Ein Request statt einem pro Kennzahl — bei 10 Requests/Sekunde ist das der
    Unterschied zwischen Sekunden und Minuten für ein ganzes Universum.
    """
    cik = str(cik).strip().lstrip("CIK").zfill(10)
    owns_client = client is None
    client = client or httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        return _get_with_retry(client, f"{_BASE}/api/xbrl/companyfacts/CIK{cik}.json")
    finally:
        if owns_client:
            client.close()


def _period_kind(start: date | None, end: date) -> str:
    """``instant`` | ``quarterly`` | ``annual`` | ``other`` — nach Dauer.

    Bilanzposten (Assets, Equity, Cash) sind Stichtagswerte und haben kein
    ``start``; Stromgrößen (Umsatz, Gewinn, Cashflow) haben eines. Die Fenster
    sind großzügig, weil Geschäftsjahre nicht auf den Tag genau laufen — Apples
    Quartale schwanken zwischen 84 und 98 Tagen.
    """
    if start is None:
        return "instant"
    days = (end - start).days
    if 80 <= days <= 100:
        return "quarterly"
    if 340 <= days <= 380:
        return "annual"
    return "other"


def concept_series(
    facts: dict[str, Any] | None,
    concept: str,
    *,
    unit: str | None = None,
    taxonomy: str = "us-gaap",
    period: str = "any",
) -> list[Fact]:
    """Alle Einreichungen zu einer Kennzahl, chronologisch nach `filed`.

    ``unit`` kommt aus `CONCEPT_UNITS`, wenn es nicht übergeben wird — für die
    allermeisten Kennzahlen ``USD``, für Ergebnis je Aktie ``USD/shares`` und
    für Stückzahlen ``shares``. Ein fester Default wäre hier eine stille Falle:
    unter der falschen Einheit liefert `companyfacts` keinen Fehler, sondern
    eine leere Reihe (siehe `CONCEPT_UNITS`).

    `period` filtert nach Periodendauer: ``"annual"``, ``"quarterly"``,
    ``"instant"`` (Bilanzstichtag) oder ``"any"``.

    **Für Stromgrößen ist das keine Feinheit, sondern Pflicht.** EDGAR liefert
    Quartals- und Jahreswerte in derselben Reihe. Wer ungefiltert den jeweils
    jüngsten Eintrag nimmt, vergleicht Äpfel mit Birnen: Apples Umsatz war zum
    Stichtag 2013-01-01 „36 Mrd" (ein Quartal) und 2023-01-01 „394 Mrd" (ein
    Jahr) — ein Faktor 11, der wie Wachstum aussieht und keins ist. Ein Backtest
    darauf hat keinen Fehler, nur ein falsches Ergebnis.

    `concept` darf ein Alias aus `CONCEPT_ALIASES` sein (``"revenue"``) oder ein
    XBRL-Tag (``"NetIncomeLoss"``).

    **Alle Alias-Tags werden zusammengeführt, nicht nur das erste.** Das ist der
    Unterschied zwischen einer nutzbaren und einer unbrauchbaren Reihe: Apple
    bucht Umsatz seit ASC 606 unter
    ``RevenueFromContractWithCustomerExcludingAssessedTax``, davor unter
    ``Revenues``. Wer nur das erste Tag mit Daten nimmt, bekommt eine Reihe, die
    2018 beginnt — und ein Backtest über zehn Jahre steht ohne Zahlen da, ohne
    dass eine Fehlermeldung darauf hinweist.

    Bei Überschneidungen gewinnt das **früher gelistete** Tag: die Reihenfolge in
    `CONCEPT_ALIASES` ist eine Rangfolge, keine Aufzählung. Verglichen wird über
    (Periode, Einreichungsdatum) — dieselbe Zahl unter zwei Tags im selben
    Filing ist ein Duplikat, keine zweite Beobachtung.

    Enthält bewusst **alle** Einreichungen, auch mehrfache pro Periode. Das
    Aussortieren macht `as_of`, weil erst dort der Stichtag bekannt ist.
    """
    if not facts:
        return []
    tags = CONCEPT_ALIASES.get(concept, (concept,))
    einheit = unit if unit is not None else CONCEPT_UNITS.get(concept, "USD")
    all_facts: dict[str, Any] = facts.get("facts", {}) or {}

    komponenten = CONCEPT_COMPONENTS.get(concept, frozenset())

    merged: dict[tuple[date | None, date, date], Fact] = {}
    #: (Beobachtung, Dokument) → Tag → Wert. Nur für Komponenten-Tags gefüllt;
    #: die `accession` steht im Schlüssel, damit später ausschliesslich
    #: innerhalb **einer** Einreichung summiert wird.
    komponenten_werte: dict[tuple[tuple[date | None, date, date], str], dict[str, float]] = {}

    for tag in tags:  # Rangfolge: früher gelistet gewinnt
        tax, _, bare = tag.partition(":")
        if not bare:
            tax, bare = taxonomy, tag
        entries = ((all_facts.get(tax) or {}).get(bare) or {}).get("units", {}).get(einheit)
        if not entries:
            continue

        for e in entries:
            end = _parse_date(e.get("end"))
            filed = _parse_date(e.get("filed"))
            val = e.get("val")
            if end is None or filed is None or val is None:
                continue
            start = _parse_date(e.get("start"))
            if period != "any" and _period_kind(start, end) != period:
                continue
            key = (start, end, filed)
            accn = str(e.get("accn", ""))

            if bare in komponenten:
                komponenten_werte.setdefault((key, accn), {})[bare] = float(val)

            if key in merged:  # höherrangiges Tag hat diese Beobachtung schon
                continue
            merged[key] = Fact(
                concept=bare,
                period_start=start,
                period_end=end,
                value=float(val),
                filed=filed,
                form=str(e.get("form", "")),
                accession=accn,
                unit=einheit,
            )

    if komponenten:
        merged = _summiere_komponenten(merged, komponenten_werte, komponenten)

    return sorted(merged.values(), key=lambda f: (f.filed, f.period_end))


def _summiere_komponenten(
    merged: dict[tuple[date | None, date, date], Fact],
    komponenten_werte: dict[tuple[tuple[date | None, date, date], str], dict[str, float]],
    komponenten: frozenset[str],
) -> dict[tuple[date | None, date, date], Fact]:
    """Bestandteile derselben Einreichung addieren — siehe `CONCEPT_COMPONENTS`.

    Angefasst wird eine Beobachtung nur, wenn **kein** Summen-Tag sie gewonnen
    hat: gewinnt ``CostOfRevenue``, ist die Summe bereits gemeldet und ein
    Addieren wäre Doppelzählung. Erst wenn nur Bestandteile da sind — und davon
    mehr als einer, im selben Dokument — wird addiert.

    Der zusammengesetzte Wert trägt seine Herkunft sichtbar: `concept` nennt die
    Summanden, `components` macht sie maschinenlesbar. Eine Zahl, die aus zwei
    Zeilen entstanden ist, soll nicht aussehen wie eine gemeldete.
    """
    for key, fact in list(merged.items()):
        if fact.concept not in komponenten:
            continue  # ein Summen-Tag hat gewonnen — nichts zu tun
        teile = komponenten_werte.get((key, fact.accession), {})
        if len(teile) < 2:
            continue  # nur ein Bestandteil gemeldet: das *ist* der Wert
        namen = tuple(sorted(teile))
        merged[key] = replace(
            fact,
            concept="+".join(namen),
            value=float(sum(teile.values())),
            components=namen,
        )
    return merged


def as_of(facts: list[Fact], when: date | str) -> list[Fact]:
    """Was am Stichtag über jede Periode bekannt war — eine Zahl pro Periode.

    Zwei Schritte, und der zweite ist der, den man vergisst:

    1. **Filtern:** alles wegwerfen, was am Stichtag noch nicht *benutzbar* war.
       Maßgeblich ist `Fact.usable_from`, nicht `filed` — 86 % der Berichte
       werden nach Börsenschluss angenommen und sind am Einreichungstag nicht
       handelbar.
    2. **Auflösen:** pro Periode bleibt oft mehr als ein Eintrag übrig — die
       Erstmeldung und spätere Korrekturen. Es gewinnt die **jüngste
       Einreichung bis zum Stichtag**. Das ist, was man an dem Tag geglaubt
       hätte; nicht der Erstwert, und erst recht nicht die heutige Wahrheit.

    Das Ergebnis ist nach Periodenende sortiert und markiert korrigierte
    Perioden über `Fact.restated` — nützlich, weil Restatements selbst ein
    Signal sind (und ein Grund, einer Zahl zu misstrauen).
    """
    cutoff = when if isinstance(when, date) else datetime.strptime(str(when)[:10], "%Y-%m-%d").date()

    by_period: dict[tuple[date | None, date], list[Fact]] = {}
    for f in facts:
        if f.usable_from <= cutoff:
            by_period.setdefault((f.period_start, f.period_end), []).append(f)

    resolved: list[Fact] = []
    for candidates in by_period.values():
        candidates.sort(key=lambda f: f.filed)
        winner = candidates[-1]
        # "Korrigiert" heißt: ein früherer Eintrag nannte einen anderen Wert.
        # Eine Wiederholung desselben Werts ist keine Korrektur.
        was_restated = any(abs(c.value - winner.value) > 1e-9 for c in candidates[:-1])
        resolved.append(
            Fact(**{**winner.__dict__, "restated": was_restated}) if was_restated else winner
        )

    return sorted(resolved, key=lambda f: f.period_end)


def latest_as_of(facts: list[Fact], when: date | str) -> Fact | None:
    """Die zuletzt bekannte Periode zum Stichtag — der übliche Einzelabruf."""
    resolved = as_of(facts, when)
    return resolved[-1] if resolved else None


# ---------------------------------------------------------------------------
# Filings (die qualitative Seite)


@dataclass(frozen=True)
class Filing:
    """Ein eingereichtes Dokument — die institutionelle Primärquelle."""

    form: str
    filed: date
    accession: str
    primary_document: str
    report_date: date | None
    url: str


def fetch_submissions(cik: str, *, client: httpx.Client | None = None) -> dict[str, Any] | None:
    """Rohes Submissions-Dokument: Stammdaten + die jüngsten Einreichungen.

    Enthält auch ``formerNames`` — die Spur, über die eine umbenannte oder
    abgewickelte Firma wiederzufinden ist. Lehman Brothers steht dort bis heute.
    """
    cik = str(cik).strip().lstrip("CIK").zfill(10)
    owns_client = client is None
    client = client or httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        return _get_with_retry(client, f"{_BASE}/submissions/CIK{cik}.json")
    finally:
        if owns_client:
            client.close()


def acceptance_map(submissions: dict[str, Any] | None) -> dict[str, datetime]:
    """Accession-Nummer → Annahmezeit, aus dem `recent`-Block der Submissions.

    Deckt nur die jüngsten ~1000 Einreichungen ab. Ältere Fakten behalten
    `accepted_at=None` und fallen in `usable_from` auf den Folgetag zurück —
    das ist der sichere Rand, kein Mangel.
    """
    if not submissions:
        return {}
    recent = ((submissions.get("filings") or {}).get("recent")) or {}
    accns = recent.get("accessionNumber") or []
    times = recent.get("acceptanceDateTime") or []

    out: dict[str, datetime] = {}
    for accn, raw in zip(accns, times, strict=False):
        if not accn or not raw:
            continue
        try:
            out[str(accn)] = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            continue
    return out


def attach_acceptance(facts: list[Fact], acceptance: dict[str, datetime]) -> list[Fact]:
    """Annahmezeiten an die Fakten hängen, soweit bekannt.

    Fakten ohne Treffer bleiben unverändert (`accepted_at=None`) und gelten
    damit ab dem Folgetag.
    """
    if not acceptance:
        return facts
    return [
        f if f.accession not in acceptance else Fact(**{**f.__dict__, "accepted_at": acceptance[f.accession]})
        for f in facts
    ]


def parse_filings(
    submissions: dict[str, Any] | None,
    *,
    forms: tuple[str, ...] = RESEARCH_FORMS,
    since: date | None = None,
) -> list[Filing]:
    """Einreichungen aus dem Submissions-Dokument, neueste zuerst.

    Deckt nur den ``recent``-Block ab (die letzten ~1000 Einreichungen). Ältere
    liegen in separaten Dateien unter ``filings.files`` — für Ereignisstudien
    über lange Zeiträume muss die nachgeladen werden. Bewusst noch nicht
    gebaut: erst wenn eine Hypothese es braucht.
    """
    if not submissions:
        return []
    recent = ((submissions.get("filings") or {}).get("recent")) or {}
    cik_raw = str(submissions.get("cik", "")).lstrip("0") or "0"

    cols = ("form", "filingDate", "accessionNumber", "primaryDocument", "reportDate")
    series = [recent.get(c) or [] for c in cols]
    if not series[0]:
        return []

    wanted = {f.upper() for f in forms} if forms else None
    out: list[Filing] = []
    for form, filed_raw, accn, doc, report_raw in zip(*series, strict=False):
        if wanted and str(form).upper() not in wanted:
            continue
        filed = _parse_date(filed_raw)
        if filed is None or (since and filed < since):
            continue
        accn_plain = str(accn).replace("-", "")
        out.append(
            Filing(
                form=str(form),
                filed=filed,
                accession=str(accn),
                primary_document=str(doc),
                report_date=_parse_date(report_raw),
                url=f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{accn_plain}/{doc}",
            )
        )
    return sorted(out, key=lambda f: f.filed, reverse=True)


# ---------------------------------------------------------------------------
# Bequemer Einstieg


#: Welche Periodendauer je Kennzahl gemeint ist. Stromgrößen als Jahreswert
#: (vergleichbar über Firmen mit verschobenem Geschäftsjahr), Bilanzposten als
#: Stichtag. Ohne diese Zuordnung müsste jeder Aufrufer sie selbst kennen — und
#: einer würde sie vergessen.
#: **Stromgrößen** (GuV, Kapitalfluss) als Jahreswert, **Bestandsgrößen**
#: (Bilanz) zum Stichtag. Die Zuordnung folgt der Rechnungslegung, nicht der
#: Bequemlichkeit: eine Bilanzposition hat keine Periodendauer, eine
#: Ertragsposition hat keinen Stichtag.
DEFAULT_PERIODS: dict[str, str] = {
    # Stromgrößen — Jahreswert, damit Firmen mit verschobenem Geschäftsjahr
    # vergleichbar bleiben.
    "revenue": "annual",
    "cost_of_revenue": "annual",
    "gross_profit": "annual",
    "operating_income": "annual",
    "rnd_expense": "annual",
    "sga_expense": "annual",
    "interest_expense": "annual",
    "pretax_income": "annual",
    "income_tax": "annual",
    "net_income": "annual",
    "eps_basic": "annual",
    "eps_diluted": "annual",
    "operating_cash_flow": "annual",
    "investing_cash_flow": "annual",
    "financing_cash_flow": "annual",
    "depreciation_amortization": "annual",
    "capex": "annual",
    "dividends_paid": "annual",
    "buybacks": "annual",
    # Bestandsgrößen — Stichtag.
    "assets": "instant",
    "current_assets": "instant",
    "inventory": "instant",
    "ppe_net": "instant",
    "liabilities": "instant",
    "current_liabilities": "instant",
    "short_term_debt": "instant",
    "long_term_debt": "instant",
    "equity": "instant",
    "cash": "instant",
    "shares_outstanding": "instant",
}


def fundamentals_as_of(
    ticker: str,
    when: date | str,
    concepts: tuple[str, ...] = ("revenue", "net_income", "assets", "equity"),
    *,
    client: httpx.Client | None = None,
) -> dict[str, Fact | None]:
    """Kennzahlen eines Tickers, wie sie am Stichtag bekannt waren.

    Der Einzeiler für den Normalfall. Ein Abruf pro Firma, danach reine
    Rechnerei — deshalb für ein ganzes Universum verträglich. Die Periodendauer
    kommt aus `DEFAULT_PERIODS`, damit Stromgrößen nicht still zwischen Quartal
    und Jahr springen.
    """
    cik = ticker_to_cik(ticker, client=client)
    if cik is None:
        log.warning("EDGAR: kein CIK für Ticker %r.", ticker)
        return dict.fromkeys(concepts)

    facts = fetch_company_facts(cik, client=client)
    return {
        c: latest_as_of(concept_series(facts, c, period=DEFAULT_PERIODS.get(c, "any")), when)
        for c in concepts
    }
