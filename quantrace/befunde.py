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
    """Was mit einem Befund geschehen soll. Von Hand gesetzt, nicht geraten."""

    code: str
    tag: date
    was: str
    #: Bei ``aktion``: der Split-Faktor im Tiingo-Sinn (2,0 für einen 2:1-Split).
    faktor: float | None = None
    #: **Pflicht bei ``aktion``.** Woher die Gewissheit kommt — ein Prospekt,
    #: eine Pressemitteilung, ein zweiter Anbieter. Ohne Beleg ist eine
    #: eingetragene Aktion nichts anderes als die Heuristik, die diesen ganzen
    #: Apparat nötig gemacht hat.
    quelle: str = ""
    grund: str = ""

    def schluessel(self) -> str:
        return f"{self.code}@{self.tag.isoformat()}"


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
    fehlend = [k for k in ("code", "tag", "was") if k not in eintrag]
    if fehlend:
        raise ValueError(f"{datei.name}: Eintrag ohne {fehlend}: {eintrag}")
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
        tag=_als_datum(eintrag["tag"]),
        was=was,
        faktor=float(faktor) if faktor is not None else None,
        quelle=quelle,
        grund=str(eintrag.get("grund") or ""),
    )


def offene(befunde: list[Befund], entscheidungen: dict[str, Entscheidung]) -> list[Befund]:
    """Befunde, zu denen noch niemand etwas gesagt hat.

    Das ist die Liste, die ein Bericht zeigen muss. Alles andere ist erledigt —
    und ``akzeptieren`` heisst *auch* erledigt, sonst wüchse die Liste ewig und
    niemand sähe den nächsten echten Befund darin.
    """
    return [b for b in befunde if b.schluessel() not in entscheidungen]


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
    "lade_entscheidungen",
    "lies_bericht",
    "offene",
    "schreibe_bericht",
]
