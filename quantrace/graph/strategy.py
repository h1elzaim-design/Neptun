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


# ---------------------------------------------------------------------------
# Sweep-Grid aus dem Graphen ableiten
# ---------------------------------------------------------------------------

#: Wie viele Kombinationen ein abgeleitetes Grid höchstens aufspannt.
#:
#: Die Grenze ist kein Geschmack, sondern Rechenzeit: jede Kombination ist ein
#: vollständiger Backtest über das Universum. Der Wert liegt bewusst **genau
#: über** 3^4 — vier Parameter à drei Stützstellen ist der typische erzeugte
#: Graph (zwei Indikatorfenster, zwei Schwellen), und ihn zu kappen hieße, den
#: Exit-Schwellwert aus dem Sweep zu werfen, weil er im Graphen zufällig hinten
#: steht. Ab fünf Parametern (3^5 = 243) wird gekappt, und das ist richtig so.
DEFAULT_MAX_COMBOS = 81

#: Stützstellen je Parameter. Drei ist das Minimum, mit dem ein Sweep etwas
#: über die *Form* sagt statt nur über zwei Punkte — kleiner, Mitte, größer.
_FAKTOREN = (0.5, 1.0, 1.5)

#: Schrittweite für einen Parameter, der auf 0 steht. Multiplikative
#: Stützstellen ergeben dort dreimal dieselbe Null; Schwellwerte um den
#: Mittelwert (`exit_z: 0.0`) sind genau dieser Fall.
_NULL_SCHRITT = 0.5


def _stuetzstellen(wert: Any, pdef: Any) -> list[Any]:
    """Drei Werte um `wert` herum — typgerecht, ohne die Untergrenze zu reißen."""
    try:
        basis = float(wert)
    except (TypeError, ValueError):
        return []  # nicht-numerisch: kein Grid, kein Rateversuch

    ist_int = getattr(pdef, "kind", "float") == "int"
    if basis == 0.0:
        roh = [-_NULL_SCHRITT, 0.0, _NULL_SCHRITT] if not ist_int else [0, 1, 2]
    else:
        roh = [basis * f for f in _FAKTOREN]

    minimum = getattr(pdef, "min", None)
    out: list[Any] = []
    for v in roh:
        v = int(round(v)) if ist_int else round(v, 6)
        if minimum is not None and v < minimum:
            continue
        if v not in out:
            out.append(v)
    return sorted(out)


def derive_param_space(
    graph: GraphSpec | dict[str, Any], *, max_combos: int = DEFAULT_MAX_COMBOS
) -> dict[str, list[Any]]:
    """Ein Sweep-Grid aus den Knoten-Parametern eines Graphen ableiten.

    **Warum das überhaupt existiert.** Ein `param_space` ist für einen Sweep
    keine Kür, sondern die Voraussetzung: ohne Grid gibt es nichts zu sweepen,
    und die Pipeline lehnt den Lauf ab (`pipeline_runner._command_for`). Eine
    vom Agenten erzeugte Graph-Strategie hatte bis hierher keins — sie war
    einzeln rechenbar und im Sweep tot, was erst am fehlgeschlagenen Run
    auffiel.

    Abgeleitet wird **deterministisch aus den gesetzten Werten**, nicht vom
    Modell geraten: je Parameter das Halbe, der Wert selbst und das Anderthalbe
    (`_FAKTOREN`), typgerecht gerundet und an der Untergrenze des Parameters
    abgeschnitten. Das ist bewusst konservativ — es ist ein Startpunkt für den
    ersten Sweep, kein Forschungsbereich. Wer einen anderen will, überschreibt
    ihn (Builder, API, oder von Hand in der Note).

    **Die Kappung ist Teil des Ergebnisses, nicht ein Detail.** Das volle
    Kreuzprodukt wächst exponentiell mit der Knotenzahl; jenseits von
    `max_combos` fallen Parameter **von hinten** wieder heraus, also die der
    spät im Graphen stehenden Knoten. Die Reihenfolge ist die des Graphen und
    damit stabil: derselbe Graph ergibt dasselbe Grid, heute und beim nächsten
    Lauf.

    Leeres Dict ist ein legitimes Ergebnis — ein Graph ohne numerische
    Parameter (reine Kreuzung zweier Quellen) hat nichts zu sweepen, und ein
    Grid zu erfinden wäre schlimmer als keins.
    """
    spec = graph if isinstance(graph, GraphSpec) else GraphSpec.model_validate(graph)

    raum: dict[str, list[Any]] = {}
    for node in spec.nodes:
        ndef = CATALOG.get(node.type)
        if ndef is None:
            continue
        for pdef in ndef.params:
            wert = node.params.get(pdef.name, pdef.default)
            if wert is None:
                continue
            werte = _stuetzstellen(wert, pdef)
            if len(werte) > 1:  # ein einziger Wert ist kein Grid
                raum[f"{node.id}.{pdef.name}"] = werte

    # Von hinten kürzen, bis das Kreuzprodukt in `max_combos` passt.
    def kombis(d: dict[str, list[Any]]) -> int:
        n = 1
        for v in d.values():
            n *= len(v)
        return n

    while raum and kombis(raum) > max_combos:
        raum.pop(next(reversed(raum)))

    return raum
