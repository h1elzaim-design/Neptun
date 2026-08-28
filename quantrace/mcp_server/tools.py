"""Die aufrufbare Oberfläche — Definitionen und Dispatch, ohne SDK.

Bewusst getrennt von ``server.py``: hier steht, *was* aufrufbar ist und was ein
Aufruf zurückgibt; dort nur, wie es über das Protokoll geht. Der Schnitt ist
kein Stil, sondern eine Testbarkeitsentscheidung — diese Datei läuft ohne das
``mcp``-Paket, und damit prüft CI die Grenze, ohne die Abhängigkeit zu
installieren.

**Das Gate ist `callable_manifest.yaml`, nicht dieses Modul.** Die Tabelle unten
beschreibt Handler; welche davon existieren, entscheidet die Datei. Ohne
Manifest ist nichts aufrufbar (fail-closed) — die teurere Voreinstellung wäre
„alles", und sie sähe im Betrieb genauso aus wie eine korrekte Konfiguration.

Warum überhaupt eine zweite Registry neben ``agents/agent_mode/tools.py``: die
dortige ist für einen **Betreiber** gebaut. Acht ihrer Tools lesen den Vault,
vier schreiben hinein, zwei rechnen auf unserer Hardware. Nach außen gedreht
wäre sie ein Leck, kein Produkt. Begründung und die Analyse aller 25 Tools:
``docs/MCP_BOUNDARY.md``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: Repo-Wurzel, wenn das Paket editierbar installiert ist.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Wo das Manifest gesucht wird, wenn die Umgebung nichts vorgibt.
_MANIFEST_ORTE = (
    _REPO_ROOT / "callable_manifest.yaml",
    Path(__file__).resolve().parent / "callable_manifest.yaml",
)

#: Vorrang vor allem anderen. Ein Betreiber soll eine **engere** Liste fahren
#: können, ohne das Paket zu verändern.
_MANIFEST_ENV = "QUANTRACE_CALLABLE_MANIFEST"


class ToolNotAllowedError(RuntimeError):
    """Ein Name, den das Manifest nicht führt.

    Kein ``KeyError``: der Unterschied zwischen „Tippfehler" und „das ist auf
    dieser Oberfläche nicht vorgesehen" gehört in die Antwort. Dieselbe
    Unterscheidung trifft die interne Registry aus demselben Grund.
    """


class ResponseTooLargeError(RuntimeError):
    """Die Antwort überschreitet die Obergrenze des Tools.

    **Nicht abgeschnitten, sondern abgelehnt.** Eine gekürzte Antwort sieht aus
    wie eine vollständige — derselbe Fehlertyp, den `_score_realism` mit
    fehlenden Kosten begangen hat: ein stiller Wert, der wie ein echter
    aussieht. Und ein Deckel, der still kürzt, ist obendrein kein Deckel gegen
    Extraktion: wer den Lake über tausend Aufrufe abziehen will, stört sich
    nicht an fehlenden Zeilen am Ende.
    """


class ManifestMissingError(RuntimeError):
    """Kein Manifest gefunden — also ist nichts aufrufbar.

    Fail-closed: ein Server, der ohne Liste alles anbietet, ist im Log nicht von
    einem korrekt konfigurierten zu unterscheiden.
    """


@dataclass(frozen=True, slots=True)
class CallableTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    #: Obergrenze der JSON-Antwort in Bytes. Kommt aus dem Manifest, nicht von
    #: hier: die Zahl ist eine Betriebsentscheidung, keine Eigenschaft des
    #: Handlers.
    max_bytes: int = 65536


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


_STR: dict[str, Any] = {"type": "string"}


# --------------------------------------------------------------------------
# Handler. Alle vier rufen ausschließlich öffentlichen `quantrace/`-Code —
# kein Import aus `agents/`. Das ist die Bedingung dafür, dass dieser Server
# selbst öffentlich sein kann: er beschreibt die Maschine, nicht das Labor.
# Die Importe stehen in den Funktionen, damit ein Prozess, der nur die Liste
# braucht, nicht den halben Graph-Compiler lädt.
# --------------------------------------------------------------------------


def _platform_capabilities() -> Any:
    # `include_lake=False` ist hier keine Sparsamkeit, sondern die Grenze:
    # Abdeckung und Instrumentenzahl sind eine Auskunft über *unseren*
    # Bestand. Was bleibt, kommt aus Code und Konfiguration.
    from quantrace import feasibility

    return feasibility.capabilities(include_lake=False)


def _assess_feasibility(
    needs_data: list[str] | None = None,
    needs_nodes: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    universe: str | None = None,
    data_from: str | None = None,
    data_to: str | None = None,
) -> Any:
    """Urteilen über **die Daten des Aufrufers**, nicht über unsere.

    ``data_from``/``data_to`` sind der Kern dieser Fassung: der Aufrufer sagt,
    welche Historie er hat, und bekommt ein Urteil darüber. Ohne die Angabe
    wird das Zeitfenster gar nicht geprüft — lieber eine ausdrücklich
    unvollständige Antwort als eine, die unseren Ladestand als seine Grenze
    ausgibt.
    """
    from datetime import date

    from quantrace import feasibility

    def _d(s: str | None) -> date | None:
        return date.fromisoformat(s) if s else None

    von, bis = _d(data_from), _d(data_to)
    if (von is None) != (bis is None):
        return {
            "feasible": False,
            "blockers": [
                "data_from und data_to gehören zusammen — mit nur einer Grenze "
                "lässt sich kein Fenster prüfen. Beide angeben oder beide weglassen."
            ],
            "caveats": [],
            "missing_nodes": [],
            "missing_data": [],
        }

    urteil = feasibility.assess(
        needs_data=needs_data,
        needs_nodes=needs_nodes,
        start=_d(start) if von else None,
        end=_d(end) if von else None,
        universe=universe,
        data_window=(von, bis) if von and bis else (date.min, date.max),
        judge_availability=False,
    ).to_dict()

    if not von and (start or end):
        urteil["caveats"] = [
            *urteil.get("caveats", []),
            "Das Zeitfenster wurde NICHT geprüft: ohne data_from/data_to ist "
            "unbekannt, welche Historie du hast. Bausteine und Datenklassen "
            "sind geprüft, die Jahre nicht.",
        ]
    return urteil


def _list_graph_nodes() -> Any:
    from quantrace.graph import PRESETS, catalog_payload

    return {"nodes": catalog_payload(), "presets": sorted(PRESETS)}


def _validate_graph_strategy(graph: dict[str, Any]) -> Any:
    from quantrace.graph import GraphSpec, validate_graph

    spec = GraphSpec.model_validate(graph)
    res = validate_graph(spec)
    return {
        "ok": bool(getattr(res, "ok", not getattr(res, "errors", []))),
        "errors": list(getattr(res, "errors", []) or []),
        "warnings": list(getattr(res, "warnings", []) or []),
    }


#: Was ein Handler *könnte*. Ob er es darf, sagt das Manifest.
_HANDLER: dict[str, tuple[str, dict[str, Any], Callable[..., Any]]] = {
    "platform_capabilities": (
        "Was die Plattform HEUTE hergibt: Graph-Knoten nach Familie, "
        "Datenklassen mit Verfügbarkeit und Begründung, das geladene "
        "Datenfenster und die Universen. Hol das, BEVOR du eine Strategie "
        "entwirfst — es beschreibt den Ist-Stand, nicht das Prinzip.",
        _schema({}),
        _platform_capabilities,
    ),
    "assess_feasibility": (
        "Ist eine Strategie-Idee mit dem heutigen Stand rechenbar? Prüft "
        "Bausteine, Datenklassen und ob das Datenfenster den Zeitraum trägt. "
        "Antwortet mit feasible plus blockers (macht sie unrechenbar) und "
        "caveats (macht sie eingeschränkt rechenbar). Ein unbekannter "
        "Schlüssel ist ein Blocker, kein ignoriertes Feld — frag vorher "
        "platform_capabilities nach den gültigen Namen. Für die Prüfung des "
        "Zeitfensters gib mit data_from/data_to an, welche Historie DU hast; "
        "ohne die beiden bleibt das Fenster ungeprüft.",
        _schema(
            {
                "needs_data": {
                    "type": "array",
                    "items": _STR,
                    "description": "Schlüssel aus platform_capabilities.data, z.B. 'eod_prices'",
                },
                "needs_nodes": {
                    "type": "array",
                    "items": _STR,
                    "description": "Knotentypen aus dem Graph-Katalog, z.B. 'indicator.rsi'",
                },
                "start": {**_STR, "description": "ISO-Datum, Beginn des Backtest-Fensters"},
                "end": {**_STR, "description": "ISO-Datum, Ende des Backtest-Fensters"},
                "universe": {**_STR, "description": "Name eines Universums, z.B. 'us_core_etfs'"},
                "data_from": {
                    **_STR,
                    "description": "ISO-Datum: ab wann DU Kursdaten hast (nicht wir)",
                },
                "data_to": {**_STR, "description": "ISO-Datum: bis wann DU Kursdaten hast"},
            }
        ),
        _assess_feasibility,
    ),
    "list_graph_nodes": (
        "Der Knotenkatalog der Graph-IR: was sich überhaupt bauen lässt, plus "
        "die Presets. Typisierte Signal-Bausteine — ein Look-ahead-Fehler ist "
        "damit strukturell ausgeschlossen, nicht durch Disziplin.",
        _schema({}),
        _list_graph_nodes,
    ),
    "validate_graph_strategy": (
        "Einen Graphen gegen den Compiler prüfen, ohne ihn zu speichern. Der "
        "billige Weg, Fehler vor dem Rechnen zu erfahren — inklusive der "
        "Ablehnung von Graphen, die Zukunft referenzieren.",
        _schema(
            {"graph": {"type": "object", "description": "Der Graph als GraphSpec-JSON"}},
            ["graph"],
        ),
        _validate_graph_strategy,
    ),
}


@dataclass(frozen=True, slots=True)
class Manifest:
    callable_names: tuple[str, ...]
    denied_names: tuple[str, ...]
    caps: dict[str, int]
    quelle: Path


def manifest_path() -> Path:
    """Das Manifest finden — oder scheitern, nie auf ein weiteres ausweichen.

    **Die Umgebungsvariable ist bindend, nicht der erste von mehreren
    Versuchen.** Zeigt sie ins Leere, ist das ein Fehler und kein Grund, auf
    die Datei im Repo zurückzufallen: wer eine engere Liste gesetzt hat und
    einen Tippfehler im Pfad, bekäme sonst still die volle Oberfläche. Genau
    die Sorte Rückfall, die im Betrieb aussieht wie eine korrekte
    Konfiguration.
    """
    gesetzt = os.environ.get(_MANIFEST_ENV, "").strip()
    if gesetzt:
        p = Path(gesetzt)
        if not p.is_file():
            raise ManifestMissingError(
                f"{_MANIFEST_ENV}={gesetzt!r} zeigt auf keine Datei. Kein "
                "Rückfall auf das Repo-Manifest: eine engere Liste, die nicht "
                "existiert, darf nicht als weitere durchgehen."
            )
        return p

    for p in _MANIFEST_ORTE:
        if p.is_file():
            return p
    raise ManifestMissingError(
        "Kein callable_manifest.yaml gefunden. Ohne Manifest ist nichts "
        f"aufrufbar — setz {_MANIFEST_ENV} oder leg die Datei in die "
        "Repo-Wurzel. Die Begründung für diese Strenge steht in "
        "docs/MCP_BOUNDARY.md."
    )


def load_manifest(path: Path | None = None) -> Manifest:
    p = path or manifest_path()
    roh = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    eintraege = roh.get("callable") or []
    namen: list[str] = []
    caps: dict[str, int] = {}
    for e in eintraege:
        name = str(e["name"])
        namen.append(name)
        if e.get("max_bytes"):
            caps[name] = int(e["max_bytes"])
    return Manifest(
        callable_names=tuple(namen),
        denied_names=tuple(str(x) for x in (roh.get("denied") or [])),
        caps=caps,
        quelle=p,
    )


def registry(path: Path | None = None) -> dict[str, CallableTool]:
    """Die Tools, die dieses Manifest freigibt.

    Ein Name im Manifest, für den es keinen Handler gibt, ist ein Fehler und
    kein leiser Ausfall: sonst könnte eine Zeile im Manifest jahrelang
    behaupten, etwas sei angeboten, das niemand implementiert hat.
    """
    m = load_manifest(path)
    verboten = set(m.denied_names)
    out: dict[str, CallableTool] = {}
    for name in m.callable_names:
        if name in verboten:
            raise ValueError(
                f"{m.quelle.name}: '{name}' steht in `callable` UND in `denied`. "
                "Eine Grenze, die sich selbst widerspricht, ist keine."
            )
        if name not in _HANDLER:
            raise ValueError(
                f"{m.quelle.name}: '{name}' hat keinen Handler in "
                f"{__name__}. Entweder tippfehler oder ein Versprechen ohne Deckung."
            )
        beschreibung, schema, handler = _HANDLER[name]
        out[name] = CallableTool(
            name=name,
            description=beschreibung,
            parameters=schema,
            handler=handler,
            max_bytes=m.caps.get(name, 65536),
        )
    return out


def call(name: str, arguments: dict[str, Any] | None = None, *, path: Path | None = None) -> str:
    """Ein Tool aufrufen und die Antwort als JSON zurückgeben.

    Die Obergrenze wird **nach** dem Serialisieren geprüft und führt zu einer
    Ablehnung, nicht zu einer Kürzung (siehe ``ResponseTooLargeError``).
    """
    tools = registry(path)
    tool = tools.get(name)
    if tool is None:
        raise ToolNotAllowedError(
            f"'{name}' ist auf dieser Oberfläche nicht aufrufbar. "
            f"Verfügbar: {', '.join(sorted(tools)) or '—'}."
        )
    ergebnis = tool.handler(**(arguments or {}))
    text = json.dumps(ergebnis, ensure_ascii=False, default=str)
    groesse = len(text.encode("utf-8"))
    if groesse > tool.max_bytes:
        raise ResponseTooLargeError(
            f"Antwort von '{name}' ist {groesse:,} Bytes, erlaubt sind "
            f"{tool.max_bytes:,}. Frag enger an — gekürzt zurückzugeben wäre "
            "eine Antwort, die vollständig aussieht und keine ist."
        )
    return text
