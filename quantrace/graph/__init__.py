"""Graph-IR: Strategien als deklarativer, serialisierbarer Signal-DAG (#179).

Öffentliche Oberfläche:
    GraphSpec / GraphNode      — das Schema (schema.py)
    CATALOG / catalog_payload  — das geschlossene Node-Vokabular (nodes.py)
    validate_graph / compile_graph / GraphValidationError — compiler.py
    GraphStrategy              — Strategy-Adapter für den bestehenden Stack
    PRESETS                    — bestehende Templates als Referenz-Graphen

Look-Ahead ist strukturell ausgeschlossen (siehe nodes.py-Docstring);
Execution-Lag bleibt Sache des Runners, wie bei allen Templates.
"""

from quantrace.graph.compiler import (
    CompiledGraph,
    GraphValidationError,
    ValidationResult,
    compile_graph,
    validate_graph,
)
from quantrace.graph.nodes import CATALOG, catalog_payload
from quantrace.graph.presets import PRESETS
from quantrace.graph.schema import GraphNode, GraphSpec
from quantrace.graph.strategy import (
    DEFAULT_MAX_COMBOS,
    GraphStrategy,
    apply_param_overrides,
    derive_param_space,
)
from quantrace.graph.vault import (
    UNVERIFIED,
    GraphSpecNotFoundError,
    build_spec,
    is_graph_spec,
    list_graph_specs,
    load_graph,
)

__all__ = [
    "CATALOG",
    "DEFAULT_MAX_COMBOS",
    "CompiledGraph",
    "GraphNode",
    "GraphSpec",
    "GraphSpecNotFoundError",
    "GraphStrategy",
    "GraphValidationError",
    "PRESETS",
    "UNVERIFIED",
    "ValidationResult",
    "apply_param_overrides",
    "build_spec",
    "catalog_payload",
    "compile_graph",
    "derive_param_space",
    "is_graph_spec",
    "list_graph_specs",
    "load_graph",
    "validate_graph",
]
