"""Das Register der Datenbefunde — melden, entscheiden, anwenden.

**Warum es das gibt.** Am 2026-09-06 hat eine Heuristik aus einem
Einheitenfehler bei ``CTAS`` einen 4:1-Split gemacht und damit einen Sprung
von +303 % in eine adjustierte Reihe geschrieben. Alle Tests waren grün, die
CI war grün, und der PR-Text argumentierte ausführlich, warum genau das nicht
passieren kann.

Der Grund war kein Programmierfehler, sondern eine unmögliche Behauptung: ein
Einheitenfehler um Faktor 4 zeigt **Kurs mal vier, Faktor durch vier, stetigen
adjusted_close** — und ein echter 1:4-Reverse-Split zeigt exakt dasselbe. Aus
den Daten allein ist das nicht zu trennen.

Daraus folgt die Aufgabenteilung dieses Moduls:

``erkennen``
    Automatisch, jederzeit, ohne Urteil. Was auffällt, wird **gemeldet**.
``entscheiden``
    Ein Mensch — oder eine spätere, belegte Quelle. Steht in
    ``data/corrections/*.yaml`` und ist versioniert wie Code.
``anwenden``
    Der Lesepfad liest die Entscheidungen. Nichts wird geraten.

Eine Korrektur im Register ist damit etwas anderes als eine Heuristik: sie
trägt einen Namen, ein Datum und eine Begründung, und wer sie später anzweifelt
findet beides wieder.

## Was ein Befund ist

Ein Befund ist **eine Beobachtung**, kein Urteil: „an diesem Tag springt die
Faktorkurve, ohne dass eine Aktion im Feed steht". Ob das eine kaputte Zeile
oder eine fehlende Aktion ist, entscheidet der Befund nicht.

## Was eine Entscheidung ist

``verwerfen``
    Die Zeilen sind keine Beobachtung. Sie werden zur Lücke — dieselbe
    Behandlung wie Nullbars (#322) und Einheitenfehler (#324).
``aktion``
    Es hat eine Corporate Action gegeben, die im Feed fehlt. Sie wird mit dem
    angegebenen Faktor angewandt. **Nur mit Beleg** — ein Verweis auf eine
    Quelle gehört ins Feld ``quelle``.
``akzeptieren``
    Der Befund ist bekannt und harmlos. Er taucht in Berichten nicht mehr auf,
    bleibt aber im Register stehen.

Verwandt: #322 · #324 · #327
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Wo die Entscheidungen liegen. Im Repo und nicht im Lake: sie sind
#: Beurteilungen, keine Daten — sie gehören in die Historie, in Reviews und
#: neben den Code, der sie anwendet.
DEFAULT_CORRECTIONS_DIR = _REPO_ROOT / "data" / "corrections"

#: Wohin der naechtliche Scan seinen Bericht schreibt und woher die API ihn
#: liest. Eine Datei und kein Endpunkt, der selbst scannt: ein Scan ueber den
#: vollen Bestand dauert Minuten und liest 7 Mio. Zeilen — das gehoert nicht
#: in einen Seitenaufruf.
DEFAULT_REPORT_PATH = _REPO_ROOT / "reports" / "datenbefunde.json"

#: Die drei Entscheidungen. Mehr braucht es nicht, und weniger reicht nicht.
ENTSCHEIDUNGEN: tuple[str, ...] = ("verwerfen", "aktion", "akzeptieren")


@dataclass(frozen=True)
class Befund:
    """Eine Beobachtung an einer Reihe — ohne Urteil darüber, was sie bedeutet."""

    instrument: str
    code: str
    tag: date
    #: Woran es aufgefallen ist: ``einheiten_ausschlag``, ``einheiten_stufe``, …
    art: str
    #: Alles, was zum Entscheiden nötig ist. Bewusst frei: der nächste
    #: Detektor bringt andere Zahlen mit, und ein starres Schema zwänge ihn,
    #: seine Belege wegzulassen.
    belege: dict[str, Any] = field(default_factory=dict)

    def schluessel(self) -> str:
        """Die Adresse, unter der eine Entscheidung dazu steht."""
        return f"{self.code}@{self.tag.isoformat()}"


@dataclass(frozen=True)
class Entscheidung:
    """Was mit einem Befund geschehen soll. Von Hand gesetzt, nicht geraten.

    **Eine Zelle oder ein Abschnitt.** Entweder steht ``tag`` — dann gilt der
    Eintrag für genau diese Zelle, wie bei ``CTAS@2005-05-25`` — oder ``von``
    und ``bis`` plus ``art``: dann gilt er für alle Befunde *dieser Art* in
    diesem Fenster.

    **Warum es die zweite Form braucht.** Der erste volle Lauf am 2026-09-08
    fand 240 Papiere mit zwei Kursniveaus, im Median 145 betroffene Zeilen je
    Papier. Zelle für Zelle wären das Zehntausende Einträge — und niemand
    trägt sie ein, also bliebe alles unentschieden.

    **Warum nicht einfach ein Datumsbereich ohne ``art``.** Bei ``DIC``
    wechseln kaputte und gesunde Zeilen täglich ab (160 Abschnitte). Ein
    blosses ``von``/``bis`` mit ``verwerfen`` nähme die gesunde Hälfte mit.
    Was zutrifft, ist der Satz *„in diesem Fenster bedeutet ein
    `kursniveau`-Befund: verwerfen"* — der Mensch entscheidet Papier, Fenster
    und Bedeutung, die Invariante nur noch, welche Zeile es trifft.

    Damit bleibt die Trennlinie, für die es dieses Modul gibt: **die
    Automatik erkennt, ein Mensch entscheidet.** Ein Abschnitt ist eine
    grössere Aussage als eine Zelle, keine andere Art von Aussage.
    """

    code: str
    was: str
    #: Die Zelle, wenn es eine einzelne ist. Schliesst ``von``/``bis`` aus.
    tag: date | None = None
    #: Der Abschnitt, wenn es einer ist — beide Grenzen einschliesslich.
    von: date | None = None
    bis: date | None = None
    #: **Pflicht beim Abschnitt.** Die ``Befund.art``, für die er gilt. Ohne
    #: sie wäre der Eintrag eine Aussage über *alle* Auffälligkeiten im
    #: Fenster — auch über die, die es beim Eintragen noch nicht gab.
    art: str = ""
    #: Bei ``aktion``: der Split-Faktor im Tiingo-Sinn (2,0 für einen 2:1-Split).
    faktor: float | None = None
    #: **Pflicht bei ``aktion``.** Woher die Gewissheit kommt — ein Prospekt,
    #: eine Pressemitteilung, ein zweiter Anbieter. Ohne Beleg ist eine
    #: eingetragene Aktion nichts anderes als die Heuristik, die diesen ganzen
    #: Apparat nötig gemacht hat.
    quelle: str = ""
    grund: str = ""

    @property
    def ist_abschnitt(self) -> bool:
        return self.tag is None

    def schluessel(self) -> str:
        """Die Adresse im Register — eindeutig für beide Formen."""
        if self.tag is not None:
            return f"{self.code}@{self.tag.isoformat()}"
        return f"{self.code}@{self.von}..{self.bis}#{self.art}"

    def deckt(self, befund: Befund) -> bool:
        """Gilt diese Entscheidung für diesen Befund?"""
        if befund.code != self.code:
            return False
        if self.tag is not None:
            return befund.tag == self.tag
        # Beide Grenzen einschliesslich: wer ein Fenster aufschreibt, meint
        # die Tage, die er hingeschrieben hat.
        return (
            befund.art == self.art
            and self.von is not None
            and self.bis is not None
            and self.von <= befund.tag <= self.bis
        )


def _als_datum(wert: Any) -> date:
    if isinstance(wert, date):
        return wert
    return date.fromisoformat(str(wert))


def lade_entscheidungen(pfad: Path | None = None) -> dict[str, Entscheidung]:
    """Alle Entscheidungen aus ``data/corrections/*.yaml``, nach Schlüssel.

    Fehlt das Verzeichnis, gibt es keine Entscheidungen — und das ist kein
    Fehler, sondern der Normalfall für ein frisches Checkout.
    """
    ordner = pfad or DEFAULT_CORRECTIONS_DIR
    if not ordner.exists():
        return {}

    aus: dict[str, Entscheidung] = {}
    for datei in sorted(ordner.glob("*.yaml")):
        roh = yaml.safe_load(datei.read_text()) or {}
        for eintrag in roh.get("entscheidungen") or []:
            e = _entscheidung_aus(eintrag, datei)
            if e.schluessel() in aus:
                # Zwei Entscheidungen zur selben Zelle sind ein Widerspruch,
                # kein Vorrang. Wer eine ändern will, ändert sie.
                raise ValueError(
                    f"{datei.name}: {e.schluessel()} ist doppelt entschieden"
                )
            aus[e.schluessel()] = e
    return aus


def _entscheidung_aus(eintrag: dict[str, Any], datei: Path) -> Entscheidung:
    fehlend = [k for k in ("code", "was") if k not in eintrag]
    if fehlend:
        raise ValueError(f"{datei.name}: Eintrag ohne {fehlend}: {eintrag}")

    # **Zelle oder Abschnitt — nicht beides und nicht keines.** Ein Eintrag mit
    # `tag` *und* `von` liesse offen, welches gilt, und die stille Antwort
    # wäre eine Heuristik an genau der Stelle, an der dieses Modul keine will.
    hat_tag = "tag" in eintrag
    hat_bereich = "von" in eintrag or "bis" in eintrag
    if hat_tag and hat_bereich:
        raise ValueError(
            f"{datei.name}: {eintrag['code']} traegt 'tag' und 'von'/'bis'. "
            "Eine Entscheidung gilt fuer eine Zelle oder fuer einen Abschnitt."
        )
    if not hat_tag and not hat_bereich:
        raise ValueError(
            f"{datei.name}: {eintrag['code']} traegt weder 'tag' noch "
            "'von'/'bis' — es fehlt, worauf sich die Entscheidung bezieht."
        )
    if hat_bereich:
        if "von" not in eintrag or "bis" not in eintrag:
            raise ValueError(
                f"{datei.name}: {eintrag['code']} braucht 'von' **und** 'bis'. "
                "Ein halboffener Abschnitt waechst mit dem Lake weiter, und "
                "was er morgen deckt, hat heute niemand angesehen."
            )
        if not str(eintrag.get("art") or "").strip():
            # Siehe `Entscheidung.art`: ohne sie gilt der Eintrag auch fuer
            # Auffaelligkeiten, die es beim Eintragen noch nicht gab.
            raise ValueError(
                f"{datei.name}: der Abschnitt fuer {eintrag['code']} braucht "
                "eine 'art' (z. B. 'kursniveau'). Ohne sie entscheidet er "
                "ueber Befunde, die niemand gesehen hat."
            )
        if _als_datum(eintrag["von"]) > _als_datum(eintrag["bis"]):
            raise ValueError(
                f"{datei.name}: {eintrag['code']} hat 'von' nach 'bis'."
            )

    was = str(eintrag["was"])
    if was not in ENTSCHEIDUNGEN:
        raise ValueError(
            f"{datei.name}: '{was}' ist keine Entscheidung "
            f"(bekannt: {', '.join(ENTSCHEIDUNGEN)})"
        )
    faktor = eintrag.get("faktor")
    quelle = str(eintrag.get("quelle") or "")
    if was == "aktion":
        if faktor is None or float(faktor) <= 0:
            raise ValueError(
                f"{datei.name}: 'aktion' fuer {eintrag['code']} braucht einen "
                "positiven Faktor"
            )
        if not quelle.strip():
            # Siehe `Entscheidung.quelle`: ohne Beleg ist das die Heuristik,
            # gegen die dieses Register gebaut ist — nur mit mehr Zeremonie.
            raise ValueError(
                f"{datei.name}: 'aktion' fuer {eintrag['code']} braucht eine "
                "'quelle'. Eine eingetragene Corporate Action ohne Beleg ist "
                "geraten, nur langsamer."
            )
    return Entscheidung(
        code=str(eintrag["code"]),
        was=was,
        tag=_als_datum(eintrag["tag"]) if hat_tag else None,
        von=_als_datum(eintrag["von"]) if hat_bereich else None,
        bis=_als_datum(eintrag["bis"]) if hat_bereich else None,
        art=str(eintrag.get("art") or ""),
        faktor=float(faktor) if faktor is not None else None,
        quelle=quelle,
        grund=str(eintrag.get("grund") or ""),
    )


def entscheidung_fuer(
    befund: Befund, entscheidungen: dict[str, Entscheidung]
) -> Entscheidung | None:
    """Was zu diesem Befund entschieden ist — Zelle vor Abschnitt.

    **Die Zelle gewinnt.** Wer einen einzelnen Tag ausdrücklich anders
    entscheidet als den Abschnitt, um den er herum liegt, meint genau das: die
    Ausnahme ist die spätere, genauere Aussage. Andersherum wäre der
    Einzeleintrag wirkungslos, und niemand sähe warum.
    """
    genau = entscheidungen.get(befund.schluessel())
    if genau is not None:
        return genau
    for e in entscheidungen.values():
        if e.ist_abschnitt and e.deckt(befund):
            return e
    return None


def offene(befunde: list[Befund], entscheidungen: dict[str, Entscheidung]) -> list[Befund]:
    """Befunde, zu denen noch niemand etwas gesagt hat.

    Das ist die Liste, die ein Bericht zeigen muss. Alles andere ist erledigt —
    und ``akzeptieren`` heisst *auch* erledigt, sonst wüchse die Liste ewig und
    niemand sähe den nächsten echten Befund darin.
    """
    return [b for b in befunde if entscheidung_fuer(b, entscheidungen) is None]


def schreibe_bericht(
    befunde: list[Befund],
    entscheidungen: dict[str, Entscheidung],
    pfad: Path | None = None,
) -> Path:
    """Der Bericht, den die API liest. Siehe `DEFAULT_REPORT_PATH`."""
    import json
    from datetime import UTC, datetime

    ziel = pfad or DEFAULT_REPORT_PATH
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(
        json.dumps(
            {
                "erstellt": datetime.now(UTC).isoformat(),
                "n_gesamt": len(befunde),
                "n_entschieden": len(entscheidungen),
                "befunde": [
                    {
                        "instrument": b.instrument,
                        "code": b.code,
                        "tag": b.tag.isoformat(),
                        "art": b.art,
                        "belege": b.belege,
                    }
                    for b in offene(befunde, entscheidungen)
                ],
            },
            indent=2,
        )
    )
    return ziel


def lies_bericht(pfad: Path | None = None) -> dict[str, Any] | None:
    """Der letzte Bericht, oder ``None``, wenn noch keiner gelaufen ist.

    ``None`` und nicht ein leerer Bericht: „noch nie gescannt" ist etwas
    anderes als „nichts gefunden", und die API muss beides unterscheiden
    koennen. Genau dieselbe Regel wie bei `DataCoverage` (#307).
    """
    import json

    ziel = pfad or DEFAULT_REPORT_PATH
    if not ziel.exists():
        return None
    try:
        return json.loads(ziel.read_text())
    except (OSError, ValueError) as exc:  # pragma: no cover - defekte Datei
        log.warning("Befundbericht unter %s nicht lesbar: %s", ziel, exc)
        return None


__all__ = [
    "DEFAULT_CORRECTIONS_DIR",
    "DEFAULT_REPORT_PATH",
    "ENTSCHEIDUNGEN",
    "Befund",
    "Entscheidung",
    "entscheidung_fuer",
    "lade_entscheidungen",
    "lies_bericht",
    "offene",
    "schreibe_bericht",
]
