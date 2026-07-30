"""Validator + Compiler der Graph-IR (#179).

`validate_graph` prüft die Struktur GEGEN den Katalog — Typen, Pflicht-Ports,
Parameter, Zyklen, genau eine Signal-Senke. Fehler sind Klartext mit Node-ID,
damit ein Editor sie inline am Knoten zeigen kann. `compile_graph` liefert ein
ausführbares Objekt; gerechnet werden nur die Vorfahren der Senke (nicht
angeschlossene Knoten sind erlaubt — Editor-Drafts — und landen als Warnung).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantrace.graph.nodes import CATALOG, NodeDef
from quantrace.graph.schema import SIGNAL, GraphNode, GraphSpec
from quantrace.models import MarketData


class GraphValidationError(ValueError):
    """Sammel-Fehler der Graph-Validierung; `errors` trägt die Einzelmeldungen."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Graph invalide:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: topologische Reihenfolge der Vorfahren der Senke (inkl. Senke, hinten)
    order: list[str] = field(default_factory=list)
    sink: str | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def _coerced_params(node: GraphNode, ndef: NodeDef, errors: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    known = {p.name for p in ndef.params}
    for name in node.params:
        if name not in known:
            errors.append(f"{node.id}: unbekannter Parameter {name!r} für {ndef.type}")
    for p in ndef.params:
        if p.name in node.params:
            try:
                out[p.name] = p.coerce(node.params[p.name])
            except ValueError as e:
                errors.append(f"{node.id}: {e}")
        elif p.required:
            errors.append(f"{node.id}: Pflichtparameter {p.name!r} fehlt")
        else:
            out[p.name] = p.default
    return out


def validate_graph(spec: GraphSpec) -> ValidationResult:
    res = ValidationResult()
    by_id = {n.id: n for n in spec.nodes}

    # 1. Typen, Ports, Parameter
    coerced: dict[str, dict[str, Any]] = {}
    for node in spec.nodes:
        ndef = CATALOG.get(node.type)
        if ndef is None:
            res.errors.append(f"{node.id}: unbekannter Node-Typ {node.type!r}")
            continue
        coerced[node.id] = _coerced_params(node, ndef, res.errors)

        ports = dict(ndef.inputs)
        for port, src_id in node.inputs.items():
            if port not in ports:
                res.errors.append(f"{node.id}: {ndef.type} hat keinen Port {port!r}")
                continue
            src = by_id.get(src_id)
            if src is None:
                res.errors.append(f"{node.id}.{port}: referenziert unbekannten Knoten {src_id!r}")
                continue
            if src_id == node.id:
                res.errors.append(f"{node.id}.{port}: Selbstreferenz")
                continue
            src_def = CATALOG.get(src.type)
            if src_def is not None and src_def.output != ports[port]:
                res.errors.append(
                    f"{node.id}.{port}: erwartet {ports[port]!r}, "
                    f"{src_id} liefert {src_def.output!r}"
                )
        for port, _ptype in ndef.inputs:
            if port not in node.inputs:
                res.errors.append(f"{node.id}: Pflicht-Port {port!r} ist nicht verbunden")

    # 2. Genau eine Signal-Senke
    sinks = [n.id for n in spec.nodes if CATALOG.get(n.type) and CATALOG[n.type].output == SIGNAL]
    if len(sinks) != 1:
        res.errors.append(
            f"Graph braucht genau einen signal.*-Knoten als Senke, hat {len(sinks)}"
            + (f" ({', '.join(sinks)})" if sinks else "")
        )
    else:
        res.sink = sinks[0]

    if res.errors:
        return res

    # 3. Zyklen-Check + topologische Ordnung, nur über die Vorfahren der Senke.
    order: list[str] = []
    state: dict[str, int] = {}  # 0=unbesucht implizit, 1=auf dem Stack, 2=fertig

    def visit(nid: str, path: list[str]) -> None:
        if state.get(nid) == 2:
            return
        if state.get(nid) == 1:
            cycle = path[path.index(nid) :] + [nid]
            res.errors.append("Zyklus: " + " → ".join(cycle))
            return
        state[nid] = 1
        for src_id in by_id[nid].inputs.values():
            visit(src_id, path + [nid])
        state[nid] = 2
        order.append(nid)

    visit(res.sink, [])
    if res.errors:
        return res

    res.order = order
    unused = sorted(set(by_id) - set(order))
    if unused:
        res.warnings.append(
            "Nicht mit der Senke verbunden (wird nicht gerechnet): " + ", ".join(unused)
        )
    # Coerced-Params zurück in die Spec-Objekte spiegeln, damit compile sie nutzt.
    for node in spec.nodes:
        if node.id in coerced:
            node.params = coerced[node.id]
    return res


@dataclass
class CompiledGraph:
    """Ausführbarer Graph: `run(data)` → (entries, exits) wie generate_signals."""

    spec: GraphSpec
    order: list[str]
    sink: str
    warnings: list[str]

    def run(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        values: dict[str, Any] = {}
        for nid in self.order:
            node = self.spec.node(nid)
            ndef = CATALOG[node.type]
            inputs = {port: values[src] for port, src in node.inputs.items()}
            values[nid] = ndef.fn(inputs, node.params, data)
        return values[self.sink]


def compile_graph(spec: GraphSpec | dict[str, Any]) -> CompiledGraph:
    """Validieren + kompilieren. Wirft GraphValidationError bei jedem Verstoß."""
    if isinstance(spec, dict):
        spec = GraphSpec.model_validate(spec)
    res = validate_graph(spec)
    if not res.ok:
        raise GraphValidationError(res.errors)
    assert res.sink is not None
    return CompiledGraph(spec=spec, order=res.order, sink=res.sink, warnings=res.warnings)
