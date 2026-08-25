"""Graph-Specs aus dem Vault laden (#188).

Der visuelle Builder (#180) speichert eine Graph-Strategie als Note unter
``Trading Research/02 Strategien/<slug>.md`` mit dem Graphen im **Frontmatter**
(`graph:`). Dieses Modul macht daraus eine ausführbare ``StrategySpec`` — der
zweite Auflösungspfad neben ``strategy_registry``, damit die Registry das
bleiben kann, was sie ist: die statische Whitelist der Code-Strategien.

Der Graph wird beim Laden **immer erneut validiert**. Eine von Hand kaputt
editierte Note (Zyklus, offener Port, Tippfehler im Node-Typ) darf nicht in
einen Backtest laufen — der Fehler kommt als Klartext, nicht als Traceback aus
der Rechenschleife.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from quantrace.graph.compiler import GraphValidationError, validate_graph
from quantrace.graph.schema import GraphSpec
from quantrace.models import StrategySpec, Timeframe

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPEC_FOLDER = "Trading Research/02 Strategien"
GRAPH_CLASS_PATH = "quantrace.graph:GraphStrategy"

#: Der Status generierter Strategien, bis ein Mensch sie befördert (#267).
#: Steht im Frontmatter der Note, nicht in einem Kommentar — die Lesepfade
#: fragen das Feld ab, und nur ein Feld lässt sich abfragen.
UNVERIFIED = "unverified"

# Slug wird in einen Vault-Pfad interpoliert — vor jedem Dateizugriff prüfen
# (gleiche Begründung wie _validate_family in api/routers/strategies.py).
_SLUG_RE = re.compile(r"^[a-z0-9_]{1,64}$")


class GraphSpecNotFoundError(FileNotFoundError):
    pass


def validate_slug(slug: str) -> str:
    slug = (slug or "").strip()
    if not _SLUG_RE.match(slug):
        raise ValueError(
            f"Ungültiger Graph-Spec-Slug {slug!r} — erlaubt ist ^[a-z0-9_]{{1,64}}$"
        )
    return slug


def spec_path(slug: str, vault_root: Path | None = None) -> Path:
    root = vault_root or REPO_ROOT
    return root / SPEC_FOLDER / f"{validate_slug(slug)}.md"


def _read_frontmatter(path: Path) -> dict[str, Any]:
    import frontmatter as fm_lib

    post = fm_lib.load(str(path))
    return dict(post.metadata or {})


def is_graph_spec(slug: str, vault_root: Path | None = None) -> bool:
    """True, wenn die Note existiert und einen `graph:`-Block trägt."""
    try:
        path = spec_path(slug, vault_root)
    except ValueError:
        return False
    if not path.exists():
        return False
    try:
        return isinstance(_read_frontmatter(path).get("graph"), dict)
    except Exception:
        return False


def list_graph_specs(
    vault_root: Path | None = None,
    *,
    include_unverified: bool = False,
) -> list[dict[str, Any]]:
    """Graph-Specs im Vault — für Dropdowns und Validierung.

    **`unverified` ist per Vorgabe nicht dabei, und die Asymmetrie ist der
    Punkt** (#267). Eine generierte Strategie soll auswählbar sein, aber nur
    *ausdrücklich*: wer sie beim Namen nennt, bekommt sie; wer „alle
    Strategien" aufzählt, nicht. Aufzählen ist der versehentliche Pfad — ein
    Sweep, ein Agentenlauf, eine Katalogrechnung —, und genau dort darf eine
    ungeprüfte Strategie nicht mitlaufen und später mit einer Zahl neben
    handgeschriebenen stehen, ohne dass man ihr die Herkunft ansieht.

    Dieselbe Bauart wie `purpose: smoke` in ADR-014: der teure Fall muss
    ausgesprochen werden, der harmlose gilt von selbst. Die drei Aufrufer, die
    heute ausdrücklich auswählen (UI-Dropdown, Slug-Prüfung,
    `param_space`-Check), setzen ``include_unverified=True``.

    `load_graph(slug)` bleibt davon unberührt: ein Slug **ist** die
    ausdrückliche Auswahl.
    """
    root = vault_root or REPO_ROOT
    folder = root / SPEC_FOLDER
    if not folder.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.md")):
        if path.name.startswith("_") or not _SLUG_RE.match(path.stem):
            continue
        try:
            fm = _read_frontmatter(path)
        except Exception:
            continue
        graph = fm.get("graph")
        if not isinstance(graph, dict):
            continue
        if not include_unverified and str(fm.get("status") or "").strip() == UNVERIFIED:
            continue
        out.append(
            {
                "slug": path.stem,
                "family": str(fm.get("family") or path.stem),
                "status": fm.get("status"),
                "n_nodes": len(graph.get("nodes") or []),
                "param_space": fm.get("param_space") or {},
            }
        )
    return out


def load_graph(slug: str, vault_root: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(graph_dict, frontmatter) einer Graph-Spec — validiert, aber nicht kompiliert."""
    path = spec_path(slug, vault_root)
    if not path.exists():
        raise GraphSpecNotFoundError(
            f"Keine Strategie-Note '{slug}' unter {SPEC_FOLDER}/ gefunden."
        )
    fm = _read_frontmatter(path)
    graph = fm.get("graph")
    if not isinstance(graph, dict):
        raise ValueError(
            f"Note '{slug}' trägt keinen `graph:`-Block im Frontmatter — das ist "
            "keine Graph-Strategie (Code-Strategien laufen über --strategy)."
        )

    try:
        parsed = GraphSpec.model_validate(graph)
    except Exception as e:
        raise ValueError(f"Graph in '{slug}' ist strukturell ungültig: {e}") from e

    res = validate_graph(parsed)
    if not res.ok:
        raise GraphValidationError(res.errors)
    return graph, fm


def build_spec(
    slug: str,
    universe: str,
    timeframe: Timeframe = Timeframe.DAILY,
    params: dict[str, Any] | None = None,
    *,
    vault_root: Path | None = None,
) -> tuple[str, StrategySpec]:
    """Graph-Spec aus dem Vault → (strategy_id, StrategySpec) für den Runner.

    Analog zu ``strategy_registry.build_spec``. Overrides sind dotted
    (``"<node_id>.<param>"``) und landen neben `graph` in ``params`` —
    ``GraphStrategy`` löst sie beim Instanziieren auf.
    """
    graph, fm = load_graph(slug, vault_root)
    merged = {"graph": graph, **(params or {})}
    strategy_class = str(fm.get("strategy_class") or "custom")
    if strategy_class not in {
        "trend_following",
        "mean_reversion",
        "momentum",
        "cross_sectional",
        "volatility",
        "custom",
    }:
        strategy_class = "custom"

    spec = StrategySpec(
        strategy_id=slug,
        name=str(fm.get("name") or slug),
        class_path=GRAPH_CLASS_PATH,
        strategy_class=strategy_class,  # type: ignore[arg-type]
        universe=universe,
        timeframe=timeframe,
        params=merged,
        param_space=dict(fm.get("param_space") or {}),
        description=str(fm.get("description") or ""),
    )
    return slug, spec
