"""Wohin eine Note geschrieben wird — echtes Research oder Testlauf.

Entscheidung vom 2026-08-09. Bis dahin teilten sich Maschinentests und Research
denselben Vault, ununterscheidbar. Das ist genau die Verwechslung, die den
Schnitt vom selben Tag nötig gemacht hat: 115 Notes, alle als Test entstanden,
alle so abgelegt als wären sie Ergebnisse.

Ab jetzt schreibt ein Testlauf unter ``Trading Research/_smoke/`` in dieselbe
Ordnerstruktur, echtes Research wie bisher direkt unter ``Trading Research/``.

**Warum ein Geschwister-Ordner und kein Frontmatter-Feld.** Die Leser im Repo
globben non-rekursiv — ``load_context`` in ``agents/_vault_context.py`` und der
Approved-Scan in ``quantrace/paper/registry.py`` machen beide
``folder.glob("*.md")``. Ein Geschwister-Root ist damit **automatisch**
unsichtbar für jeden RAG-Kontext, jede Kandidatenliste und jede
Trial-Zählung. Ein Frontmatter-Feld müsste an jeder Lesestelle geprüft werden,
und die eine vergessene Stelle wäre wieder genau der Fehler von vorher. Hier
kann man das Filtern nicht vergessen, weil es keins gibt.

**Der Default ist ``smoke``, nicht ``research``.** Ein Lauf, der nichts sagt,
ist ein Testlauf. Research ist eine bewusste Angabe. Dieselbe Richtung wie
``Tool.costly`` im Agent Mode: die folgenreiche Zusage muss man aussprechen,
die harmlose gilt von selbst. Die umgekehrte Voreinstellung hat den Vault in
den Zustand gebracht, den der Schnitt aufgeräumt hat.

Praktisch heißt das für ``purpose``: was nicht ``"research"`` ist, ist
``"smoke"`` — auch Tippfehler, ``None`` und leere Strings. Ein unlesbarer Wert
darf nicht in die Research-Spur fallen.
"""

from __future__ import annotations

import logging
from typing import Literal

log = logging.getLogger(__name__)

#: Wurzel des Obsidian-Vaults, relativ zum Repo-Root.
VAULT_ROOT = "Trading Research"

#: Unterordner für Maschinentests. Der Unterstrich ist kein Zierrat: Obsidian
#: und die Leser hier behandeln ``_``-Präfixe bereits als "nicht für den
#: normalen Durchgang" (siehe ``_vault_context`` und ``02 Strategien/_experimental``).
SMOKE_SEGMENT = "_smoke"

Purpose = Literal["smoke", "research"]

#: Siehe Modul-Docstring: unmarkiert heißt Testlauf.
DEFAULT_PURPOSE: Purpose = "smoke"

RESEARCH: Purpose = "research"
SMOKE: Purpose = "smoke"


def coerce_purpose(raw: object) -> Purpose:
    """Beliebige Eingabe → ``"research"`` oder ``"smoke"``.

    Nur der exakte String ``"research"`` (ohne Rücksicht auf Groß-/Kleinschreibung
    und umgebende Leerzeichen) ergibt die Research-Spur. Alles andere — ``None``,
    ``""``, ``"reserach"``, eine Zahl — wird zu ``"smoke"``.

    Das ist bewusst asymmetrisch. Ein Testlauf, der versehentlich als Research
    abgelegt wird, verschmutzt die Mehrfachtest-Rechnung des echten Research.
    Ein Research-Lauf, der versehentlich als Test abgelegt wird, liegt im
    falschen Ordner und wird verschoben. Nur einer der beiden Fehler ist teuer.
    """
    if isinstance(raw, str) and raw.strip().lower() == RESEARCH:
        return RESEARCH
    if raw not in (None, "", SMOKE) and not (
        isinstance(raw, str) and raw.strip().lower() == SMOKE
    ):
        log.warning("purpose=%r unbekannt — Lauf gilt als Testlauf (%s).", raw, SMOKE)
    return SMOKE


def vault_root(purpose: object = DEFAULT_PURPOSE) -> str:
    """Vault-Wurzel für diese Spur, relativ zum Repo-Root.

    ``"Trading Research"`` für Research, ``"Trading Research/_smoke"`` für Tests.
    """
    if coerce_purpose(purpose) is RESEARCH:
        return VAULT_ROOT
    return f"{VAULT_ROOT}/{SMOKE_SEGMENT}"


def note_dir(folder: str, purpose: object = DEFAULT_PURPOSE) -> str:
    """Zielordner einer Note, z. B. ``Trading Research/_smoke/03 Backtests``."""
    return f"{vault_root(purpose)}/{folder}"


def note_path(folder: str, title: str, purpose: object = DEFAULT_PURPOSE) -> str:
    """Vollständiger Pfad einer Note, relativ zum Repo-Root.

    ``title`` wird nur von Pfadtrennern befreit — die Slug-Bildung bleibt beim
    Aufrufer, weil die Konventionen je Ordner verschieden sind (Datumspräfix bei
    Backtests, blanker Family-Name bei Strategien).
    """
    safe = title.replace("/", "-").strip()
    return f"{note_dir(folder, purpose)}/{safe}.md"


def is_smoke_path(rel_path: str) -> bool:
    """Liegt dieser Pfad in der Testspur?

    Erwartet einen Pfad relativ zum Repo-Root, wie ihn ``note_path`` liefert.
    """
    normalised = str(rel_path).replace("\\", "/").strip("/")
    return normalised.startswith(f"{VAULT_ROOT}/{SMOKE_SEGMENT}/") or normalised == (
        f"{VAULT_ROOT}/{SMOKE_SEGMENT}"
    )


def purpose_of_path(rel_path: str) -> Purpose:
    """Spur aus einem Pfad ablesen — die Umkehrung von ``note_path``."""
    return SMOKE if is_smoke_path(rel_path) else RESEARCH
