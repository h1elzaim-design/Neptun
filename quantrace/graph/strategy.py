"""GraphStrategy — die Brücke von der Graph-IR in den bestehenden Stack (#179).

Eine Graph-Strategie ist für den Rest des Systems eine ganz normale Strategie:

    StrategySpec(
        class_path="quantrace.graph:GraphStrategy",
        params={"graph": {...}, "fast.window": 10},
        param_space={"fast.window": [5, 10, 20]},   # Sweeps über Knoten-Params
        ...
    )

Damit laufen Sweep, Walk-Forward, DSR/PBO/Bootstrap, Evaluation und Vault-Notes
unverändert — sie kennen nur StrategySpec und (entries, exits).

Parameter-Overrides sind dotted: ``"<node_id>.<param>"``. Unbekannte Knoten
oder Parameter sind ein harter Fehler — nie stillschweigend ignorieren, sonst
sweept man ein Grid, das gar nicht wirkt.
"""

from __future__ import annotations

from typing import Any

from quantrace.graph.compiler import compile_graph
from quantrace.graph.nodes import CATALOG
from quantrace.graph.schema import GraphSpec
from quantrace.strategy import Strategy


def apply_param_overrides(spec: GraphSpec, overrides: dict[str, Any]) -> GraphSpec:
    """Dotted Overrides ("node.param" → Wert) auf eine Kopie der Spec anwenden."""
    if not overrides:
        return spec
    out = spec.model_copy(deep=True)
    by_id = {n.id: n for n in out.nodes}
    for key, value in overrides.items():
        node_id, sep, param = key.partition(".")
        if not sep or not param:
            raise ValueError(
                f"Override {key!r}: erwartet das Format '<node_id>.<param>' "
                f"(z.B. 'fast.window')"
            )
        node = by_id.get(node_id)
        if node is None:
            raise ValueError(f"Override {key!r}: Graph hat keinen Knoten {node_id!r}")
        ndef = CATALOG.get(node.type)
        if ndef is not None and param not in {p.name for p in ndef.params}:
            raise ValueError(f"Override {key!r}: {node.type} hat keinen Parameter {param!r}")
        node.params[param] = value
    return out


class GraphStrategy(Strategy):
    """Führt eine Graph-IR aus. `graph` ist Pflicht, alles andere sind Overrides."""

    defaults: dict[str, Any] = {}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        graph = self.params.get("graph")
        if not isinstance(graph, dict):
            raise ValueError(
                "GraphStrategy braucht params['graph'] (dict im GraphSpec-Format) — "
                "siehe quantrace/graph/schema.py"
            )
        overrides = {k: v for k, v in self.params.items() if k != "graph"}
        spec = apply_param_overrides(GraphSpec.model_validate(graph), overrides)
        self._compiled = compile_graph(spec)

    def generate_signals(self, data):  # type: ignore[override]
        return self._compiled.run(data)
