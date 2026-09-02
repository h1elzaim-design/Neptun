"""Welche Währung notiert ein Kürzel wirklich? (#297 Punkt 2)

**Das Problem.** Der US-Bulk-Feed enthält Papiere, die nicht in Dollar
notieren. `HSBA` steht 2010 bei 689,50 — das ist HSBC in London, in **Pence**.
Der Screen rechnet die Zahl als Dollar und setzt HSBC damit vor AAPL und GOOG.
Ebenso `CELSIA` (kolumbianische Pesos), `PVD`, `SBER` (Rubel), `HVN`.

**Warum der Katalog es nicht weiss.** EODHDs Symbollisten für die USA sagen
es nicht. Am 2026-09-02 abgefragt: in der aktiven Liste tragen alle 51.118
Einträge ``Currency: USD``, im Friedhof 59.923 von 59.924 — und dort steht
auch ``Country: USA`` für HSBC (London), Harvey Norman (Sydney) und
Petrovietnam (Ho-Chi-Minh-Stadt). Die Quelle hat das Feld, aber nicht die
Antwort. Ein Seed, der es nur durchreicht, kann daran nichts ändern.

**Woher die Antwort dann kommt.** Aus den Listen der *anderen* Börsen. EODHD
führt 70 davon, und dort stimmt die Währung: `HSBA` steht in der LSE-Liste
als ``GBX`` mit demselben Namen. Ein Kürzel, das

1. **nicht** in der aktiven US-Liste steht,
2. an einer Fremdbörse in einer anderen Währung als USD notiert,
3. und dort denselben Firmennamen trägt,

ist mit hoher Sicherheit dieses Fremdlisting — und nicht das US-Papier.

**Alle drei Bedingungen sind nötig, und die erste ist die wichtigste.** `BP`
steht in der LSE-Liste als GBX und heisst dort "BP PLC" — aber BP hat auch ein
echtes NYSE-ADR unter demselben Kürzel. Es als GBX zu markieren würde ein
liquides US-Papier aus jedem Universum werfen und seine Kurse verfälschen. Die
aktive US-Liste trennt die Fälle: `BP` steht drin, `HSBA` nicht.

**Die Richtung des Zweifels.** Wer nichts findet, ändert nichts. Eine falsche
Währungsangabe verfälscht Kurse und Dollarvolumina eines Papiers, das
womöglich echt ist; eine fehlende lässt nur den Zustand, den es ohnehin schon
gibt. Deshalb der strenge Namensabgleich und keine Heuristik über den
Kursbereich: 95 der 1.453 Code-Überschneidungen mit der LSE erfüllen ihn,
und das ist die richtige Grössenordnung — es sind Fremdkörper, keine Mehrheit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

#: Von Hand belegte Fremdlistings — für die Börsen, die EODHD gar nicht führt.
#: Begründung und Aufnahmekriterium stehen in der Datei selbst.
MANUAL_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "data_sources" / "foreign_listings.yaml"
)

#: Notierungen in Bruchteilen der Hauptwährung. EODHD führt London in ``GBX``
#: (Pence) und Johannesburg in ``ZAC`` (Cent) — beide sind ein Hundertstel
#: ihrer Hauptwährung, und wer das übersieht, rechnet um den Faktor 100 falsch.
#: Der Fehler wäre hier besonders tückisch: er macht Kurse *plausibler*, nicht
#: absurder, und fällt deshalb nicht auf.
SUBUNITS: dict[str, tuple[str, float]] = {
    "GBX": ("GBP", 0.01),
    "GBp": ("GBP", 0.01),
    "ZAC": ("ZAR", 0.01),
    "ILA": ("ILS", 0.01),
}

#: So viele führende Zeichen des Firmennamens müssen übereinstimmen.
#:
#: Nicht der ganze Name: dieselbe Firma heisst bei EODHD an zwei Börsen selten
#: buchstabengleich ("Shell plc" gegen "Shell PLC ADR"). Nicht weniger: bei zehn
#: Zeichen trifft "Bank of Am…" auf ein Dutzend verschiedener Firmen, und eine
#: Fehlzuordnung kostet hier ein echtes Papier.
NAME_PREFIX_LEN = 20

_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalise_name(name: str | None) -> str:
    """Firmenname auf das Vergleichbare reduziert.

    Satzzeichen und Gross-/Kleinschreibung fallen weg — "HSBC Holdings PLC"
    und "HSBC HOLDINGS PLC." sollen gleich sein. Die Rechtsform bleibt drin:
    sie unterscheidet Firmen, die sich sonst gleichen.
    """
    return _NORM_RE.sub("", (name or "").lower())[:NAME_PREFIX_LEN]


@dataclass(frozen=True)
class CurrencyFinding:
    """Ein Kürzel, das nachweislich nicht in Dollar notiert."""

    code: str
    currency: str
    exchange: str
    name: str
    #: Hauptwährung und Faktor, falls die Notierung eine Untereinheit ist.
    #: ``("GBP", 0.01)`` für Pence. ``None``, wenn die Währung selbst gilt.
    subunit: tuple[str, float] | None = None

    @property
    def base_currency(self) -> str:
        """Die Währung, für die es einen Wechselkurs gibt (``GBX`` → ``GBP``)."""
        return self.subunit[0] if self.subunit else self.currency

    @property
    def to_base(self) -> float:
        """Faktor auf die Hauptwährung. ``0.01`` für Pence, sonst ``1.0``."""
        return self.subunit[1] if self.subunit else 1.0


def resolve_currencies(
    *,
    us_active_codes: set[str],
    catalog: dict[str, str],
    foreign_listings: dict[str, list[tuple[str, str, str]]],
) -> list[CurrencyFinding]:
    """Welche Kürzel des Katalogs notieren in Fremdwährung?

    Parameters
    ----------
    us_active_codes
        Kürzel aus EODHDs **aktiver** US-Symbolliste. Wer hier steht, ist ein
        echtes US-Papier und wird nie umgewidmet — das ist der `BP`-Schutz.
    catalog
        Kürzel → Firmenname, so wie der Instrumentenkatalog sie führt.
    foreign_listings
        Kürzel → Liste von ``(Börsencode, Währung, Name)`` aus den Listen der
        anderen Börsen.

    Returns
    -------
    Nur Kürzel, bei denen alle drei Bedingungen zutreffen. Mehrdeutige — das
    Kürzel notiert an zwei Fremdbörsen in verschiedenen Währungen und beide
    Namen passen — bleiben **aussen vor**: welche der beiden gilt, ist damit
    nicht entschieden, und raten wäre schlechter als schweigen.
    """
    treffer: list[CurrencyFinding] = []
    for code, katalogname in catalog.items():
        if code in us_active_codes:
            continue  # echtes US-Papier — nie umwidmen
        kandidaten = foreign_listings.get(code)
        if not kandidaten:
            continue
        ziel = normalise_name(katalogname)
        if not ziel:
            continue
        passend = {
            (waehrung, boerse, name)
            for boerse, waehrung, name in kandidaten
            if waehrung and waehrung.upper() != "USD" and normalise_name(name) == ziel
        }
        if not passend:
            continue
        waehrungen = {w.upper() for w, _, _ in passend}
        if len(waehrungen) > 1:
            log.info(
                "%s notiert an mehreren Fremdbörsen in verschiedenen Währungen (%s) "
                "— nicht zugeordnet.",
                code,
                ", ".join(sorted(waehrungen)),
            )
            continue
        waehrung, boerse, name = sorted(passend)[0]
        treffer.append(
            CurrencyFinding(
                code=code,
                currency=waehrung,
                exchange=boerse,
                name=name,
                subunit=SUBUNITS.get(waehrung) or SUBUNITS.get(waehrung.upper()),
            )
        )
    return sorted(treffer, key=lambda t: t.code)


@lru_cache(maxsize=1)
def manual_listings(path: str | None = None) -> dict[str, CurrencyFinding]:
    """Die von Hand belegten Fremdlistings, Kürzel → Befund.

    **Warum es sie neben `resolve_currencies` gibt.** Das Skript liest die
    Symbollisten der 69 Börsen, die EODHD führt — und findet damit nichts, was
    dort fehlt. Vietnam, Kolumbien, Russland und Thailand fehlen, und genau
    deren Papiere stehen 2020 auf den vordersten Rängen jedes US-Screens
    (`PVD`, `LDG`, `VPI`, `HVN`, `KDH`, `ISA`, `TTB`).

    **Warum eine Liste und kein Filter.** Am 2026-09-02 wurden vier
    datengetriebene Tests durchgemessen; keiner trennt. Der schärfste — nie
    adjustiert *und* Kurs über 300 $ — hätte `BRK-A` (99.600 $) und `ADBE`
    (334 $) mitgenommen, also Adobe aus jedem Universum geworfen. Die fehlende
    Information ist aus Kursen nicht rekonstruierbar; sie muss von aussen
    kommen. Eine Liste, in der jede Zeile ihren Beleg trägt, ist dann
    ehrlicher als eine Regel, die daneben greift.
    """
    import yaml

    pfad = Path(path) if path else MANUAL_PATH
    if not pfad.exists():
        return {}
    daten = yaml.safe_load(pfad.read_text()) or {}
    out: dict[str, CurrencyFinding] = {}
    for eintrag in daten.get("listings") or []:
        code = str(eintrag.get("code") or "").strip()
        waehrung = str(eintrag.get("currency") or "").strip().upper()
        # **Ohne Beleg kein Eintrag.** Eine geratene Waehrung macht den Kurs
        # plausibel und den Fehler damit unsichtbar — teurer als gar keine
        # Angabe, weil danach niemand mehr hinsieht.
        if not code or not waehrung or not str(eintrag.get("evidence") or "").strip():
            log.warning(
                "foreign_listings: Eintrag ohne Code, Waehrung oder Beleg — ignoriert: %r", eintrag
            )
            continue
        out[code] = CurrencyFinding(
            code=code,
            currency=waehrung,
            exchange="(manuell)",
            name=str(eintrag.get("name") or ""),
            subunit=SUBUNITS.get(waehrung),
        )
    return out


def all_findings(
    resolved: list[CurrencyFinding], *, manual: dict[str, CurrencyFinding] | None = None
) -> list[CurrencyFinding]:
    """Automatisch aufgeloeste und von Hand belegte Befunde, zusammengefuehrt.

    **Das Automatische gewinnt.** Es steht auf EODHDs eigener Symbolliste einer
    Boerse, an der das Papier wirklich notiert; ein Handeintrag ist eine
    Ableitung aus dem Firmennamen. Wo beide etwas sagen, ist die Quelle die
    bessere Auskunft — und ein Widerspruch gehoert ins Log, weil dann einer
    von beiden falsch ist.
    """
    hand = dict(manual_listings() if manual is None else manual)
    zusammen = {t.code: t for t in resolved}
    for code, befund in hand.items():
        vorhanden = zusammen.get(code)
        if vorhanden is None:
            zusammen[code] = befund
        elif vorhanden.currency.upper() != befund.currency.upper():
            log.warning(
                "%s: Boersenliste sagt %s, foreign_listings.yaml sagt %s — "
                "die Boersenliste gilt. Einer der beiden ist falsch.",
                code,
                vorhanden.currency,
                befund.currency,
            )
    return sorted(zusammen.values(), key=lambda t: t.code)
