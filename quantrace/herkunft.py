"""Herkunftsstempel: womit ein Ergebnis gerechnet wurde (#342, ADR-017).

**Warum es das gibt.** Am 2026-09-06 wurden fünf Wege gefixt, auf denen kaputte
Kurse in Kennzahlen wanderten (#312, #322, #323, #324). Die Konsequenz stand in
``docs/STATUS.md``: *„Alle bisherigen Ergebnisse sind gelöscht, sie sind auf
anderen Daten gerechnet."* Die Entscheidung war richtig. Alternativlos war sie
nur, weil sich nicht feststellen liess, **welche** Ergebnisse die betroffenen
Papiere überhaupt berührt hatten.

Mit einem Stempel wird daraus eine Abfrage: „diese Läufe tragen einen
Korrektur-Stand vor X und berühren ein korrigiertes Papier — die sind veraltet,
der Rest steht."

Der zweite Grund wiegt schwerer: DSR, PBO und FDR korrigieren dafür, *wie viel
probiert wurde*. Ein Lauf ohne Herkunft lässt sich nachträglich nicht in diese
Zählung einordnen.

## Die Felder

``code``
    Inhaltshash über die Dateien, die rechnen (``quantrace/``, ``strategies/``,
    ``config/``, ``pyproject.toml``). **Nicht** der Commit: auf ``main`` landet
    jeder Vault-Save als Commit, und der Worker committet seine Ergebnisse in
    den eigenen Klon. Zwei Läufe mit identischem Code trügen sonst
    verschiedene Stempel — und ein Stempel, der sich ohne Grund bewegt, sagt
    nichts mehr.
``commit``
    Wo man den Code findet — Fundstelle, **kein** Vergleichsschlüssel. Mit
    ``+geaendert``, wenn die rechnenden Pfade vom Commit abweichen: dann ist
    der Commit nur die Nachbarschaft, und das soll man sehen.
``lake_bis``
    Bis wohin die Karte von Schicht 2 reicht — der Stand, gegen den gelesen
    wurde. ``None``, wenn nicht aus dem Lake gelesen wurde.
``korrekturen``
    Hash über alle Entscheidungen in ``data/corrections/`` — grob: jede neue
    Entscheidung bewegt ihn.
``korrekturen_betroffen``
    Derselbe Hash, nur über die Entscheidungen zu Papieren **dieses** Laufs.
    Das ist der feine Stempel aus #342 b): eine Entscheidung zu ``DIC``
    entwertet nur die Läufe, die ``DIC`` gelesen haben.
``universum``
    Hash der Universe-YAML. Sie ändert sich (``usable_window``, Neubau von
    ``us_top500_liquid``), der Name nicht.
``rechner``
    ``azure`` | ``local`` | ``daddy`` | ``heroku`` — wer gerechnet hat.

## Was bewusst **nicht** im Hash steht

Zeitstempel und Pfade. Zwei Läufe auf identischem Stand müssen identische
Stempel tragen (Abnahme 3 in #342), sonst ist „gleich" nicht mehr prüfbar.
Gehasht wird deshalb der **gelesene Inhalt** (YAML nach dem Parsen, kanonisch
serialisiert): ein geänderter Kommentar in einer Korrektur-Datei entwertet
nichts, eine geänderte Entscheidung schon.

## Was der Stempel nicht tut

Er markiert nichts als veraltet. #342 a) schlägt vor, erst nur zu stempeln —
eine Automatik ohne Erfahrung zu bauen ist das Muster aus #329, wo eine gut
gemeinte Heuristik einen Sprung von +303 % in ``CTAS`` geschrieben hat.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Was rechnet. ``data/universes`` und ``data/corrections`` haben eigene
#: Felder; ``agents/`` und ``api/`` rechnen nicht, sie rendern und verteilen.
CODE_PFADE: tuple[str, ...] = ("quantrace", "strategies", "config", "pyproject.toml")

#: Dateien, die in ``CODE_PFADE`` liegen, aber nichts ausrechnen. Bytecode
#: wird aus dem Quelltext erzeugt, und wer ihn mithasht, bekommt je Python-
#: Version einen anderen Stempel für denselben Code.
_IGNORIERT = ("__pycache__", ".pyc", ".pyo", ".DS_Store")

#: Hex-Stellen je Hash. 12 statt der 6 aus dem Ticket-Beispiel: bei tausenden
#: Läufen ist eine Kollision bei 6 Stellen (16,7 Mio.) nicht mehr
#: ausgeschlossen, und eine Kollision hiesse hier „gleicher Stand", wo keiner ist.
_STELLEN = 12


class Herkunft(BaseModel):
    """Der Stempel. Geht unverändert ins Ergebnis-JSON, in Postgres und ins
    Vault-Frontmatter — **eine** Form an allen drei Stellen, damit sie
    übereinstimmen können (#342, Abnahme 1)."""

    model_config = ConfigDict(frozen=True)

    code: str
    commit: str | None = None
    lake_bis: str | None = None
    korrekturen: str
    korrekturen_betroffen: str | None = None
    universum: str | None = None
    rechner: str

    def als_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _kurz(daten: bytes) -> str:
    return hashlib.sha256(daten).hexdigest()[:_STELLEN]


def _kanonisch(obj: Any) -> bytes:
    """Stabile Bytes für ein geparstes YAML — Schlüsselreihenfolge egal."""
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode()


def _dateien(wurzel: Path, pfade: tuple[str, ...]) -> list[Path]:
    aus: list[Path] = []
    for rel in pfade:
        p = wurzel / rel
        if p.is_file():
            aus.append(p)
        elif p.is_dir():
            aus.extend(
                f
                for f in p.rglob("*")
                if f.is_file() and not any(teil in str(f) for teil in _IGNORIERT)
            )
    return sorted(aus)


def code_hash(wurzel: Path | None = None, pfade: tuple[str, ...] = CODE_PFADE) -> str:
    """Inhaltshash der rechnenden Dateien — Pfad und Bytes je Datei.

    Der Pfad gehört dazu (relativ zur Wurzel): eine umbenannte Datei ist ein
    anderer Import. Der absolute Pfad nicht — derselbe Code in ``/app`` und in
    ``~/QuantRace`` ist derselbe Code.
    """
    wurzel = wurzel or _REPO_ROOT
    h = hashlib.sha256()
    for f in _dateien(wurzel, pfade):
        h.update(f.relative_to(wurzel).as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:_STELLEN]


def _git(wurzel: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=str(wurzel),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def commit(wurzel: Path | None = None) -> str | None:
    """Der Commit, mit ``+geaendert``, wenn die rechnenden Pfade abweichen.

    ``None`` ohne Git — ein Container ohne ``.git`` rechnet trotzdem, und
    ``code`` trägt die Identität auch dann.
    """
    wurzel = wurzel or _REPO_ROOT
    env = os.environ.get("QUANTRACE_CODE_VERSION", "").strip()
    if env:
        return env
    sha = _git(wurzel, "rev-parse", "--short=12", "HEAD")
    if not sha:
        return None
    geaendert = _git(
        wurzel, "status", "--porcelain", "--untracked-files=no", "--", *CODE_PFADE
    )
    return f"{sha}+geaendert" if geaendert else sha


def _entscheidungen_roh(ordner: Path) -> list[tuple[str, dict[str, Any]]]:
    """(Dateiname, Eintrag) für jede Entscheidung — ungeprüft.

    Bewusst nicht über ``befunde.lade_entscheidungen``: die validiert und wirft,
    und ein ungültiges Register ist ein Fehler des Lesepfads, nicht des
    Stempels. Der Stempel soll beschreiben, was dalag.
    """
    if not ordner.exists():
        return []
    aus: list[tuple[str, dict[str, Any]]] = []
    for datei in sorted(ordner.glob("*.yaml")):
        roh = yaml.safe_load(datei.read_text(encoding="utf-8")) or {}
        for eintrag in roh.get("entscheidungen") or []:
            if isinstance(eintrag, dict):
                aus.append((datei.name, eintrag))
    return aus


def _korrektur_hash(eintraege: list[tuple[str, dict[str, Any]]]) -> str:
    # Sortiert, damit die Reihenfolge der Einträge in der Datei keine Rolle
    # spielt: umsortieren ändert nichts an dem, was der Lesepfad anwendet.
    return _kurz(b"\n".join(sorted(_kanonisch([n, e]) for n, e in eintraege)))


def korrektur_hashes(
    symbole: list[str] | None = None, ordner: Path | None = None
) -> tuple[str, str | None]:
    """(alle, betroffen) — der grobe und der feine Korrektur-Stand.

    ``betroffen`` filtert auf die Codes des Laufs. Ein Suffix für geteilte
    Kürzel (``ACL__S1``) gehört nicht zum Code, unter dem entschieden wird.
    """
    from quantrace.befunde import DEFAULT_CORRECTIONS_DIR

    eintraege = _entscheidungen_roh(ordner or DEFAULT_CORRECTIONS_DIR)
    alle = _korrektur_hash(eintraege)
    if symbole is None:
        return alle, None
    codes = {str(s).split("__", 1)[0].upper() for s in symbole}
    betroffen = [(n, e) for n, e in eintraege if str(e.get("code", "")).upper() in codes]
    return alle, _korrektur_hash(betroffen)


def universum_hash(name: str, ordner: Path | None = None) -> str | None:
    """Hash des geparsten Universe-YAML, oder ``None``, wenn es keines gibt."""
    pfad = (ordner or (_REPO_ROOT / "data" / "universes")) / f"{name}.yaml"
    if not pfad.exists():
        return None
    return _kurz(_kanonisch(yaml.safe_load(pfad.read_text(encoding="utf-8"))))


def rechner() -> str:
    """Wer rechnet. ``QUANTRACE_RECHNER`` gewinnt — auf Daddy steht dort
    ``daddy`` (``docs/DADDY.md``), weil die Maschine sich sonst nicht von einem
    beliebigen Laptop unterscheidet."""
    gesetzt = os.environ.get("QUANTRACE_RECHNER", "").strip().lower()
    if gesetzt:
        return gesetzt
    # Beide setzt Azure selbst: Apps und Jobs, ohne unser Zutun.
    if os.environ.get("CONTAINER_APP_NAME") or os.environ.get("CONTAINER_APP_JOB_NAME"):
        return "azure"
    if os.environ.get("DYNO"):
        return "heroku"
    return "local"


def lake_bis(provider: str | None) -> str | None:
    """Der Karten-Stand von Schicht 2 — nur, wenn aus dem Lake gelesen wurde.

    Die Karte ist zu diesem Zeitpunkt meist schon im Speicher (der Lesepfad hat
    sie eben gelesen, ``read_manifest`` hält sie 60 s). Scheitert der Zugriff,
    steht ``None`` da: „unbekannt" ist ehrlicher als ein Stand, den niemand
    gemessen hat — und ein Stempel darf keinen Lauf verhindern, der bereits
    gerechnet hat.
    """
    if provider != "eodhd":
        return None
    try:
        from quantrace import resolve

        karte = resolve.read_manifest()
        if karte.empty or "last" not in karte.columns:
            return None
        return str(max(karte["last"]))
    except Exception as exc:  # noqa: BLE001 - Stempel ist Protokoll, kein Gate
        log.warning("Herkunft: Karten-Stand nicht lesbar (%s)", exc)
        return None


def stempel(
    *,
    universe: str | None = None,
    symbole: list[str] | None = None,
    provider: str | None = None,
    wurzel: Path | None = None,
) -> Herkunft:
    """Den Stempel für einen Lauf ermitteln — deterministisch.

    Aufrufen, **nachdem** die Daten geladen sind und bevor gerechnet wird: dann
    beschreibt er genau den Stand, der gelesen wurde, und nicht den, der nach
    einem stundenlangen Sweep zufällig daliegt.
    """
    wurzel = wurzel or _REPO_ROOT
    alle, betroffen = korrektur_hashes(symbole, ordner=wurzel / "data" / "corrections")
    return Herkunft(
        code=code_hash(wurzel),
        commit=commit(wurzel),
        lake_bis=lake_bis(provider),
        korrekturen=alle,
        korrekturen_betroffen=betroffen,
        universum=universum_hash(universe, ordner=wurzel / "data" / "universes")
        if universe
        else None,
        rechner=rechner(),
    )


def aus_ergebnis(pfad: Path) -> dict[str, Any] | None:
    """Den Stempel aus einem Ergebnis-JSON lesen — für Postgres.

    Die Run-Zeile liest ihn **aus derselben Datei**, aus der auch die Vault-Note
    gerendert wird. Zwei getrennte Ermittlungen könnten auseinanderlaufen — der
    Stempel in der Note und der in der Datenbank beschrieben dann verschiedene
    Läufe, und keiner merkt es.
    """
    try:
        payload = json.loads(pfad.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    wert = payload.get("gerechnet_mit") if isinstance(payload, dict) else None
    return wert if isinstance(wert, dict) else None


def aus_neuen_ergebnissen(ordner: Path, seit_ts: float) -> dict[str, Any] | None:
    """Der Stempel des ersten Ergebnis-JSON, das seit ``seit_ts`` geschrieben wurde.

    Dieselbe Auswahl wie beim Veröffentlichen (API-Subprozess und Worker): alle
    ``*.json`` im Ausgabeordner mit ``mtime >= seit_ts - 1`` — eine Sekunde
    Spielraum für die Auflösung des Dateisystems. Ein Lauf schreibt genau eine
    Datei; stehen mehrere da, tragen sie denselben Stempel, weil sie aus
    demselben Prozess kommen.
    """
    if not ordner.exists():
        return None
    for p in sorted(ordner.glob("*.json")):
        try:
            if p.stat().st_mtime < seit_ts - 1.0:
                continue
        except OSError:
            continue
        wert = aus_ergebnis(p)
        if wert is not None:
            return wert
    return None
