"""Graph-IR-Schema — die serialisierbare Form einer Strategie (#179).

Eine Graph-Strategie ist ein typisierter DAG: Quellen (`source.*`) liefern
Serien, Indikatoren/Mathe transformieren sie, Logik-Knoten machen daraus
Bool-Zustände, und genau EIN `signal.*`-Knoten (die Senke) übersetzt das in
(entries, exits). Das Schema hier ist reine Struktur — Typprüfung, Zyklen,
Parameter-Validierung passieren in `quantrace.graph.compiler.validate_graph`
gegen den Node-Katalog.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

#: Port-/Werttypen im Graphen. "signal" existiert nur als Output der Senke.
SERIES = "series"
BOOL = "bool"
SIGNAL = "signal"


class GraphNode(BaseModel):
    """Ein Knoten: Typ aus dem Katalog + Params + benannte Eingangs-Ports."""

    id: str = Field(..., pattern=r"^[a-z0-9_]{1,64}$")
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    #: Port-Name → id des liefernden Knotens.
    inputs: dict[str, str] = Field(default_factory=dict)


class GraphSpec(BaseModel):
    """Der komplette Graph. Kanten sind implizit über `inputs` definiert."""

    version: int = 1
    nodes: list[GraphNode]

    @field_validator("nodes")
    @classmethod
    def _ids_unique(cls, v: list[GraphNode]) -> list[GraphNode]:
        seen: set[str] = set()
        for n in v:
            if n.id in seen:
                raise ValueError(f"Doppelte Node-ID: {n.id!r}")
            seen.add(n.id)
        return v

    def node(self, node_id: str) -> GraphNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)
