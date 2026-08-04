"""Portfolio-Registry: Ziel-Gewichte aus `05 Approved Candidates/` ableiten.

Schließt die Lücke zwischen dem Approval-Loop (Mensch schiebt Evaluations nach
`05 Approved Candidates/`) und dem Rebalance-Planner (braucht symbol-genaue
Ziel-Gewichte): statt die Gewichte im `/paper`-Dry-Run von Hand einzutippen,
liest die Registry die approved Notes und baut daraus einen Ziel-Vektor.

Aufbau (rein, kein Netz, kein git — nur Filesystem-Reads im Repo):

1. Jede Note in `Trading Research/05 Approved Candidates/` mit
   ``type: approved_strategy`` und ``go_live_stage`` in {paper, live_small,
   live_full} (controlled vocabulary, VAULT_CONVENTIONS §3) ist ein Kandidat.
   Der Mensch hat die Note dorthin bewegt — das ist die Autorität;
   Agent-Verdicts/Guardrail-Verstöße werden aber als Flags mitgeliefert.
2. Universe-Auflösung über den `related`-Wikilink auf die Backtest-Note
   (`03 Backtests/<slug>.md` → Frontmatter `universe`), Symbole aus
   `data/universes/<universe>.yaml`.
3. Sleeve-Gewichte über die Kandidaten. Zwei Wege:
   - **ohne Return-Pfade** (Default): ``equal`` (1/n) oder ``score``
     (proportional zum Governance-Score) — keine Risikosicht, weil keine
     Historie vorliegt.
   - **mit Return-Pfaden** (``sleeve_returns``): echte Portfolio-Konstruktion
     über :func:`quantrace.portfolio.construct_portfolio` — Kovarianz mit
     Ledoit-Wolf-Shrinkage plus die gewählte Sizing-Methode (risk_parity,
     min_variance, …), Constraints und optionales Vol-Targeting. Bei
     ``weighting='score'`` wird der Governance-Score zum **Risikobudget**
     statt zum Kapitalgewicht. Fehlen Pfade oder ist die gemeinsame Historie
     zu kurz, fällt die Registry mit Warnung auf 1/n zurück — nie stillschweigend.
4. **Innerhalb** eines Sleeves: Gleichgewichtung über die Universe-Symbole.
   Das ist bewusst die *neutrale* Allokation — die signal-konditionierte
   Version (Positionen aus dem tatsächlichen Strategie-Signal) kommt mit dem
   Daily-Rebalance-Cron (siehe ``daily_plan.py``) und ersetzt dann nur diesen
   Schritt.

Die Governance-Invariante bleibt unberührt: die Registry *plant* nur —
Ausführung läuft weiter über den gated Executor.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from quantrace.portfolio.construction import (
    Constraints,
    PortfolioConstruction,
    construct_portfolio,
)
from quantrace.portfolio.risk_model import MIN_OBS

log = logging.getLogger(__name__)

APPROVED_DIR = "Trading Research/05 Approved Candidates"
BACKTESTS_DIR = "Trading Research/03 Backtests"
UNIVERSES_DIR = "data/universes"

# Controlled vocabulary aus docs/VAULT_CONVENTIONS.md §3 (05 Approved Candidates).
_STAGES = ("paper", "live_small", "live_full")

# `related: ['[[03 Backtests/2026-06-12_grid_global_macro]]']`
_BACKTEST_LINK = re.compile(r"\[\[03 Backtests/([^\]|#]+)")
# Body-Bullets aus dem Evaluation-Agent: "- `fast` = 10"
_PARAM_BULLET = re.compile(r"^- `(?P<key>[^`]+)` = (?P<val>.+)$", re.MULTILINE)


@dataclass(slots=True)
class ApprovedCandidate:
    """Eine approved Note, aufgelöst bis auf Universe + Symbole."""

    slug: str
    note_path: str
    strategy: str | None
    params: dict[str, Any]
    universe: str | None
    symbols: list[str]
    score: float | None
    sharpe: float | None
    verdict: str | None
    guardrails_passed: bool | None
    go_live_stage: str | None
    approved_by: str | None
    approval_date: str | None
    #: Slug der verlinkten Backtest-Note — der Schlüssel, unter dem der
    #: persistierte Return-Pfad des Sleeves liegt.
    backtest_slug: str | None = None
    sleeve_weight: float = 0.0
    flags: list[str] = field(default_factory=list)

    @property
    def deployable(self) -> bool:
        """Trägt Symbole → kann in den Ziel-Vektor eingehen."""
        return bool(self.symbols)


#: Innerhalb der Sleeves gleichgewichtet — neutrale Allokation ohne Signal.
#: Ein automatisierter Konsument (Daily-Cron) MUSS dieses Feld prüfen, bevor
#: er target_weights als echte Allokation behandelt.
WEIGHTING_BASIS_NEUTRAL = "neutral_equal_within_sleeve"
#: Reserviert für den Daily-Cron: Gewichte aus dem tatsächlichen Strategie-Signal.
WEIGHTING_BASIS_SIGNAL = "strategy_signal"


@dataclass(slots=True)
class PortfolioRegistry:
    """Ziel-Portfolio aus dem Vault: Kandidaten + kombinierter Gewichts-Vektor."""

    candidates: list[ApprovedCandidate]
    target_weights: dict[str, float]
    weighting: str
    # Maschinell prüfbarer Marker, WIE die Symbol-Gewichte zustande kamen —
    # nicht nur eine Prosa-Warning. Solange hier NEUTRAL steht, sind die
    # Gewichte ein Platzhalter (kein Signal) und dürfen nicht automatisch
    # submittet werden.
    weighting_basis: str = WEIGHTING_BASIS_NEUTRAL
    #: Tatsächlich verwendete Sizing-Methode der Sleeve-Gewichte. ``equal``
    #: heißt: kein Risikomodell im Spiel (keine Pfade oder Fallback).
    sizing: str = "equal"
    #: Vollständiger Konstruktions-Report (Risikobeiträge, Vol, Shrinkage),
    #: sofern ein Risikomodell gerechnet werden konnte.
    construction: PortfolioConstruction | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def gross_weight(self) -> float:
        return sum(self.target_weights.values())

    @property
    def n_deployable(self) -> int:
        return sum(1 for c in self.candidates if c.deployable)


def _parse_scalar(raw: str) -> Any:
    """Best-effort Typisierung eines Param-Werts aus dem Note-Body."""
    txt = raw.strip()
    try:
        return int(txt)
    except ValueError:
        pass
    try:
        return float(txt)
    except ValueError:
        pass
    if txt.lower() in ("true", "false"):
        return txt.lower() == "true"
    return txt


def _load_note(path: Path) -> tuple[dict[str, Any], str]:
    """Frontmatter + Body einer Vault-Note. Wirft nicht — Fehler → ({}, '')."""
    try:
        import frontmatter

        post = frontmatter.load(str(path))
        return dict(post.metadata or {}), post.content or ""
    except Exception as e:  # kaputtes YAML o.ä. — Note überspringen, nicht crashen
        log.warning("registry: Note %s nicht lesbar: %s", path.name, e)
        return {}, ""


def _related_backtest_slug(fm: dict[str, Any]) -> str | None:
    related = fm.get("related") or []
    if isinstance(related, str):
        related = [related]
    for entry in related:
        m = _BACKTEST_LINK.search(str(entry))
        if m:
            return m.group(1).strip()
    return None


def _resolve_universe(fm: dict[str, Any], repo_root: Path) -> str | None:
    """Universe der Kandidatin: eigenes Frontmatter, sonst die Backtest-Note."""
    universe = fm.get("universe")
    if universe:
        return str(universe)
    slug = _related_backtest_slug(fm)
    if not slug:
        return None
    bt_path = repo_root / BACKTESTS_DIR / f"{slug}.md"
    if not bt_path.exists():
        return None
    bt_fm, _ = _load_note(bt_path)
    universe = bt_fm.get("universe")
    return str(universe) if universe else None


def _universe_symbols(universe: str, repo_root: Path) -> list[str]:
    path = repo_root / UNIVERSES_DIR / f"{universe}.yaml"
    if not path.exists():
        return []
    try:
        meta = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        log.warning("registry: Universe-YAML %s nicht lesbar: %s", path.name, e)
        return []
    symbols = meta.get("symbols") or []
    return [str(s).strip() for s in symbols if str(s).strip()]


def _params_from_body(body: str) -> dict[str, Any]:
    """Best-Run-Parameter aus dem `## Auto-generated`-Body der Evaluation-Note.

    Der Evaluation-Agent schreibt sie als "- `key` = value"-Bullets unter
    "### Best Run" — dasselbe Format, das :meth:`EvaluationResult.to_markdown`
    erzeugt, daher stabil parsebar.
    """
    section = body.split("### Best Run", 1)
    if len(section) < 2:
        return {}
    # Bis zur nächsten H3/H2 schneiden, damit keine fremden Bullets matchen.
    chunk = re.split(r"\n#{2,3} ", section[1], maxsplit=1)[0]
    return {
        m.group("key"): _parse_scalar(m.group("val"))
        for m in _PARAM_BULLET.finditer(chunk)
    }


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_registry(
    repo_root: Path,
    *,
    weighting: str = "equal",
    stages: tuple[str, ...] = _STAGES,
    sizing: str = "equal",
    sleeve_returns: Mapping[str, Sequence[float]] | None = None,
    constraints: Constraints | None = None,
    current_weights: Mapping[str, float] | None = None,
    periods_per_year: float = 252.0,
    min_obs: int = MIN_OBS,
) -> PortfolioRegistry:
    """Baut die Portfolio-Registry aus dem Vault.

    Parameters
    ----------
    repo_root:
        Repo-Wurzel (enthält ``Trading Research/`` und ``data/universes/``).
    weighting:
        ``equal`` — 1/n über die deploybaren Kandidaten (Default).
        ``score`` — proportional zum Governance-Score; Kandidaten ohne
        positiven Score fallen auf equal zurück (mit Warnung). Mit einer
        risikomodell-basierten ``sizing``-Methode wird der Score zum
        **Risikobudget** statt zum Kapitalgewicht.
    stages:
        Zugelassene ``go_live_stage``-Werte (VAULT_CONVENTIONS §3:
        paper | live_small | live_full). Default: alle drei.
    sizing:
        Eine aus :data:`quantrace.portfolio.SIZING_METHODS`. Alles außer
        ``equal`` braucht ``sleeve_returns`` — sonst Fallback auf 1/n.
    sleeve_returns:
        ``key → (T,) periodische Returns`` des jeweiligen Sleeves,
        **zeitaligned**. Key ist der Kandidaten-Slug (``eval_<backtest>``)
        oder der Backtest-Slug — beides wird aufgelöst.
    constraints:
        Box/Brutto/Turnover/Vol-Targeting für die Konstruktion.
    current_weights:
        Aktuelle Sleeve-Gewichte (für Turnover-Deckel und -Ausweis).
    """
    if weighting not in ("equal", "score"):
        raise ValueError(f"weighting must be 'equal' or 'score', got {weighting!r}")

    approved_dir = repo_root / APPROVED_DIR
    warnings: list[str] = []
    candidates: list[ApprovedCandidate] = []

    if not approved_dir.is_dir():
        warnings.append(f"{APPROVED_DIR}/ existiert nicht — leeres Portfolio.")
        return PortfolioRegistry([], {}, weighting, warnings=warnings)

    for path in sorted(approved_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        fm, body = _load_note(path)
        if fm.get("type") != "approved_strategy":
            continue

        stage = fm.get("go_live_stage")
        if stage not in stages:
            warnings.append(
                f"{path.stem}: go_live_stage={stage!r} nicht in {list(stages)} — übersprungen."
            )
            continue

        universe = _resolve_universe(fm, repo_root)
        symbols = _universe_symbols(universe, repo_root) if universe else []

        cand = ApprovedCandidate(
            slug=path.stem,
            note_path=f"{APPROVED_DIR}/{path.name}",
            strategy=fm.get("strategy"),
            params=_params_from_body(body),
            universe=universe,
            symbols=symbols,
            score=_safe_float(fm.get("score")),
            sharpe=_safe_float(fm.get("sharpe")),
            verdict=fm.get("verdict"),
            guardrails_passed=fm.get("guardrails_passed"),
            go_live_stage=stage,
            approved_by=fm.get("approved_by"),
            approval_date=str(fm.get("approval_date") or "") or None,
            backtest_slug=_related_backtest_slug(fm),
        )

        # Flags: der Mensch hat approved — das zählt. Aber Widersprüche zum
        # Agent-Verdict gehören sichtbar in den Plan, nicht unter den Teppich.
        if cand.guardrails_passed is False:
            cand.flags.append("guardrails_failed (human override)")
        if cand.verdict and cand.verdict != "approve":
            cand.flags.append(f"agent_verdict={cand.verdict} (human override)")
        if universe is None:
            cand.flags.append("universe_unresolved")
            warnings.append(
                f"{path.stem}: Universe nicht auflösbar (weder Frontmatter noch "
                "Backtest-Note) — trägt keine Gewichte bei."
            )
        elif not symbols:
            cand.flags.append("universe_symbols_missing")
            warnings.append(
                f"{path.stem}: keine Symbole für Universe '{universe}' in "
                f"{UNIVERSES_DIR}/ — trägt keine Gewichte bei."
            )

        candidates.append(cand)

    deployable = [c for c in candidates if c.deployable]
    if not deployable:
        if candidates:
            warnings.append("Kein Kandidat deploybar — target_weights leer.")
        return PortfolioRegistry(candidates, {}, weighting, warnings=warnings)

    # Sleeve-Gewichte über die Kandidaten: erst der Risikomodell-Pfad,
    # sonst die historienfreie Heuristik (equal / score).
    sleeves, used_sizing, construction = _sleeve_weights(
        deployable,
        weighting=weighting,
        sizing=sizing,
        sleeve_returns=sleeve_returns,
        constraints=constraints,
        current_weights=current_weights,
        periods_per_year=periods_per_year,
        min_obs=min_obs,
        warnings=warnings,
    )

    for cand, sleeve in zip(deployable, sleeves, strict=True):
        cand.sleeve_weight = sleeve

    # Symbol-Vektor: Sleeve gleichverteilt über die Universe-Symbole,
    # Überlappungen (gleiches Symbol in mehreren Universen) summieren sich.
    target_weights: dict[str, float] = {}
    for cand in deployable:
        w = cand.sleeve_weight / len(cand.symbols)
        for sym in cand.symbols:
            target_weights[sym] = target_weights.get(sym, 0.0) + w

    warnings.append(
        "Innerhalb der Sleeves: Gleichgewichtung über die Universe-Symbole "
        "(neutrale Allokation). Signal-konditionierte Gewichte kommen mit dem "
        "Daily-Rebalance-Cron."
    )

    return PortfolioRegistry(
        candidates=candidates,
        target_weights={k: round(v, 8) for k, v in sorted(target_weights.items())},
        weighting=weighting,
        sizing=used_sizing,
        construction=construction,
        warnings=warnings,
    )


def _returns_for(
    cand: ApprovedCandidate, sleeve_returns: Mapping[str, Sequence[float]]
) -> Sequence[float] | None:
    """Return-Pfad eines Kandidaten — per Kandidaten- oder Backtest-Slug.

    Der Aufrufer darf beides als Key benutzen: den Slug der Evaluation-Note
    (``eval_<backtest>``) oder den der verlinkten Backtest-Note. Letzteres ist
    der Weg, auf dem der Vault die Pfade liefert.
    """
    for key in (
        cand.slug,
        cand.backtest_slug,
        cand.slug[5:] if cand.slug.startswith("eval_") else None,
    ):
        if key and key in sleeve_returns:
            return sleeve_returns[key]
    return None


def _feasible_constraints(
    constraints: Constraints | None, n: int, warnings: list[str]
) -> Constraints | None:
    """Konzentrationsgrenzen an ein kleines Buch anpassen.

    Eine Governance-Grenze wie ``max_weight=0.40`` ist für ein gewachsenes
    Buch gedacht, mit zwei Sleeves aber schlicht unerfüllbar (2 × 0.4 < 1.0).
    Statt daran zu scheitern wird sie auf das Minimum gelockert, das
    ``Σw = gross`` überhaupt zulässt — mit Warnung, damit sichtbar bleibt,
    dass die Grenze gerade nicht bindet.
    """
    if constraints is None or n < 1:
        return constraints

    from dataclasses import replace

    updated = constraints
    even = constraints.gross / n
    if n * constraints.max_weight < constraints.gross - 1e-12:
        warnings.append(
            f"max_weight={constraints.max_weight:.0%} ist bei {n} Sleeves nicht "
            f"erfüllbar — auf {even:.0%} (Gleichgewichtung) gelockert."
        )
        updated = replace(updated, max_weight=even)
    if n * constraints.min_weight > constraints.gross + 1e-12:
        warnings.append(
            f"min_weight={constraints.min_weight:.0%} ist bei {n} Sleeves nicht "
            f"erfüllbar — auf {even:.0%} gesenkt."
        )
        updated = replace(updated, min_weight=even)
    if updated.min_weight > updated.max_weight:
        updated = replace(updated, min_weight=updated.max_weight)
    return updated


def _heuristic_sleeves(
    deployable: list[ApprovedCandidate], weighting: str, warnings: list[str]
) -> list[float]:
    """Historienfreie Sleeve-Gewichte: 1/n oder score-proportional."""
    n = len(deployable)
    if weighting != "score":
        return [1.0 / n] * n
    scores = [c.score if (c.score or 0.0) > 0 else None for c in deployable]
    if any(s is None for s in scores):
        warnings.append(
            "score-Gewichtung: mindestens ein Kandidat ohne positiven Score — "
            "Fallback auf equal."
        )
        return [1.0 / n] * n
    total = sum(s for s in scores if s is not None)
    return [float(s) / total for s in scores if s is not None]


def _sleeve_weights(
    deployable: list[ApprovedCandidate],
    *,
    weighting: str,
    sizing: str,
    sleeve_returns: Mapping[str, Sequence[float]] | None,
    constraints: Constraints | None,
    current_weights: Mapping[str, float] | None,
    periods_per_year: float,
    min_obs: int,
    warnings: list[str],
) -> tuple[list[float], str, PortfolioConstruction | None]:
    """Sleeve-Gewichte + die tatsächlich verwendete Methode.

    Der Risikomodell-Pfad greift nur, wenn *jeder* deploybare Kandidat einen
    Return-Pfad mitbringt: eine Kovarianz über eine Teilmenge wäre stillschweigend
    ein anderes Portfolio. Jeder Fallback wird als Warnung ausgewiesen — die
    Registry darf nie so tun, als läge ein Risikomodell vor, wenn keins da ist.
    """
    def fallback() -> list[float]:
        return _heuristic_sleeves(deployable, weighting, warnings)

    if sizing == "equal":
        return fallback(), "equal", None
    if not sleeve_returns:
        warnings.append(
            f"sizing='{sizing}' angefragt, aber keine Sleeve-Return-Pfade übergeben — "
            "Fallback auf die historienfreie Gewichtung."
        )
        return fallback(), "equal", None

    series = {c.slug: _returns_for(c, sleeve_returns) for c in deployable}
    missing = [slug for slug, s in series.items() if s is None or len(s) == 0]
    if missing:
        warnings.append(
            f"sizing='{sizing}': kein Return-Pfad für {missing} — ein Risikomodell über "
            "nur einen Teil des Buchs wäre irreführend, daher Fallback auf die "
            "historienfreie Gewichtung. (Alt-Ergebnisse ohne persistierten Pfad: "
            "Winner einmal re-runnen.)"
        )
        return fallback(), "equal", None

    budgets: dict[str, float] | None = None
    if weighting == "score":
        if all((c.score or 0.0) > 0 for c in deployable):
            budgets = {c.slug: float(c.score or 0.0) for c in deployable}
            warnings.append(
                "weighting='score' mit Risikomodell: der Governance-Score ist das "
                "**Risikobudget** (Anteil am Portfoliorisiko), nicht das Kapitalgewicht."
            )
        else:
            warnings.append(
                "weighting='score': mindestens ein Kandidat ohne positiven Score — "
                "gleiche Risikobudgets."
            )

    try:
        construction = construct_portfolio(
            {slug: list(s) for slug, s in series.items() if s is not None},
            method=sizing,
            constraints=_feasible_constraints(constraints, len(deployable), warnings),
            risk_budgets=budgets,
            current_weights=current_weights,
            periods_per_year=periods_per_year,
            min_obs=min_obs,
        )
    except ValueError as e:
        warnings.append(
            f"Portfolio-Konstruktion ('{sizing}') nicht möglich: {e} — Fallback auf die "
            "historienfreie Gewichtung."
        )
        return fallback(), "equal", None

    warnings.extend(construction.warnings)
    return (
        [construction.weights[c.slug] for c in deployable],
        sizing,
        construction,
    )


__all__ = [
    "ApprovedCandidate",
    "PortfolioRegistry",
    "load_registry",
]
