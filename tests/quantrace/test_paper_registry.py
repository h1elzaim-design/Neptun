"""Portfolio-Registry: `05 Approved Candidates/` → Ziel-Gewichte.

Baut einen synthetischen Vault in tmp_path (Approved-Notes + Backtest-Notes +
Universe-YAMLs) und prüft Auflösung, Sleeve-Gewichtung, Overlaps und Flags.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantrace.paper import load_registry

APPROVED = "Trading Research/05 Approved Candidates"
BACKTESTS = "Trading Research/03 Backtests"
UNIVERSES = "data/universes"


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _approved_note(
    *,
    backtest_slug: str,
    strategy: str = "sma_crossover",
    score: float = 0.82,
    sharpe: float = 1.4,
    verdict: str = "approve",
    guardrails_passed: bool = True,
    stage: str = "paper",
    params_bullets: str = "- `fast` = 10\n- `slow` = 200",
) -> str:
    return f"""---
type: approved_strategy
strategy: {strategy}
score: {score}
sharpe: {sharpe}
verdict: {verdict}
guardrails_passed: {str(guardrails_passed).lower()}
go_live_stage: {stage}
approved_by: test-reviewer
approval_date: '2026-06-12'
related:
- '[[03 Backtests/{backtest_slug}]]'
---

## Auto-generated

### Verdict

**{verdict}** — final score {score}

### Best Run

Parameters:
{params_bullets}

Metrics:
- Sharpe: {sharpe}
"""


def _backtest_note(universe: str) -> str:
    return f"""---
type: backtest_report
universe: {universe}
---

## Auto-generated
"""


def _universe_yaml(name: str, symbols: list[str]) -> str:
    sym_lines = "\n".join(f"  - {s}" for s in symbols)
    return f"name: {name}\nprovider: tiingo\nsymbols:\n{sym_lines}\n"


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    _write(tmp_path, f"{UNIVERSES}/uni_a.yaml", _universe_yaml("uni_a", ["SPY", "QQQ"]))
    _write(tmp_path, f"{UNIVERSES}/uni_b.yaml", _universe_yaml("uni_b", ["QQQ", "IWM", "DIA", "GLD"]))
    _write(tmp_path, f"{BACKTESTS}/2026-06-01_grid_a.md", _backtest_note("uni_a"))
    _write(tmp_path, f"{BACKTESTS}/2026-06-02_grid_b.md", _backtest_note("uni_b"))
    return tmp_path


def test_single_candidate_full_nav_equal_within_sleeve(vault: Path):
    _write(vault, f"{APPROVED}/eval_a.md", _approved_note(backtest_slug="2026-06-01_grid_a"))

    reg = load_registry(vault)

    assert reg.n_deployable == 1
    assert reg.target_weights == {"QQQ": 0.5, "SPY": 0.5}
    assert reg.gross_weight == pytest.approx(1.0)
    cand = reg.candidates[0]
    assert cand.universe == "uni_a"
    assert cand.sleeve_weight == pytest.approx(1.0)
    assert cand.params == {"fast": 10, "slow": 200}
    assert cand.flags == []


def test_two_candidates_equal_sleeves_and_overlap_sums(vault: Path):
    _write(vault, f"{APPROVED}/eval_a.md", _approved_note(backtest_slug="2026-06-01_grid_a"))
    _write(
        vault,
        f"{APPROVED}/eval_b.md",
        _approved_note(backtest_slug="2026-06-02_grid_b", strategy="dual_momentum", score=0.6),
    )

    reg = load_registry(vault)

    assert reg.n_deployable == 2
    # Sleeve A: 0.5 auf {SPY, QQQ} → je 0.25. Sleeve B: 0.5 auf 4 Symbole → je 0.125.
    # QQQ liegt in beiden → 0.25 + 0.125.
    assert reg.target_weights["SPY"] == pytest.approx(0.25)
    assert reg.target_weights["QQQ"] == pytest.approx(0.375)
    assert reg.target_weights["GLD"] == pytest.approx(0.125)
    assert reg.gross_weight == pytest.approx(1.0)


def test_score_weighting(vault: Path):
    _write(
        vault, f"{APPROVED}/eval_a.md",
        _approved_note(backtest_slug="2026-06-01_grid_a", score=0.9),
    )
    _write(
        vault, f"{APPROVED}/eval_b.md",
        _approved_note(backtest_slug="2026-06-02_grid_b", score=0.3),
    )

    reg = load_registry(vault, weighting="score")

    sleeves = {c.slug: c.sleeve_weight for c in reg.candidates}
    assert sleeves["eval_a"] == pytest.approx(0.75)  # 0.9 / 1.2
    assert sleeves["eval_b"] == pytest.approx(0.25)
    assert reg.gross_weight == pytest.approx(1.0)


def test_score_weighting_falls_back_on_nonpositive_score(vault: Path):
    _write(vault, f"{APPROVED}/eval_a.md", _approved_note(backtest_slug="2026-06-01_grid_a", score=0.9))
    _write(vault, f"{APPROVED}/eval_b.md", _approved_note(backtest_slug="2026-06-02_grid_b", score=0.0))

    reg = load_registry(vault, weighting="score")

    assert all(c.sleeve_weight == pytest.approx(0.5) for c in reg.candidates if c.deployable)
    assert any("Fallback auf equal" in w for w in reg.warnings)


def test_vault_stage_vocabulary_accepted(vault: Path):
    """VAULT_CONVENTIONS §3: go_live_stage ∈ {paper, live_small, live_full} —
    alle drei Werte müssen Kandidaten produzieren."""
    _write(vault, f"{APPROVED}/eval_p.md", _approved_note(backtest_slug="2026-06-01_grid_a", stage="paper"))
    _write(vault, f"{APPROVED}/eval_ls.md", _approved_note(backtest_slug="2026-06-02_grid_b", stage="live_small"))
    _write(vault, f"{APPROVED}/eval_lf.md", _approved_note(backtest_slug="2026-06-01_grid_a", stage="live_full"))

    reg = load_registry(vault)

    assert {c.go_live_stage for c in reg.candidates} == {"paper", "live_small", "live_full"}
    assert reg.n_deployable == 3


def test_weighting_basis_marks_neutral_placeholder(vault: Path):
    """Solange kein Signal-Overlay existiert, muss der Ziel-Vektor maschinell
    als Platzhalter erkennbar sein — nicht nur über Prosa-Warnings."""
    from quantrace.paper.registry import WEIGHTING_BASIS_NEUTRAL

    _write(vault, f"{APPROVED}/eval_a.md", _approved_note(backtest_slug="2026-06-01_grid_a"))
    reg = load_registry(vault)
    assert reg.weighting_basis == WEIGHTING_BASIS_NEUTRAL


def test_stage_filter_and_readme_and_foreign_types_skipped(vault: Path):
    _write(vault, f"{APPROVED}/README.md", "# Ordner-Doku, kein Kandidat\n")
    _write(
        vault, f"{APPROVED}/eval_paused.md",
        _approved_note(backtest_slug="2026-06-01_grid_a", stage="paused"),
    )
    _write(vault, f"{APPROVED}/notiz.md", "---\ntype: note\n---\nfreie Notiz\n")

    reg = load_registry(vault)

    assert reg.candidates == []
    assert reg.target_weights == {}
    assert any("go_live_stage" in w for w in reg.warnings)


def test_unresolvable_universe_carries_no_weight_but_is_listed(vault: Path):
    _write(
        vault, f"{APPROVED}/eval_ghost.md",
        _approved_note(backtest_slug="9999-99-99_missing"),
    )
    _write(vault, f"{APPROVED}/eval_a.md", _approved_note(backtest_slug="2026-06-01_grid_a"))

    reg = load_registry(vault)

    assert len(reg.candidates) == 2
    assert reg.n_deployable == 1
    ghost = next(c for c in reg.candidates if c.slug == "eval_ghost")
    assert "universe_unresolved" in ghost.flags
    assert ghost.sleeve_weight == 0.0
    # Der deploybare Kandidat bekommt das volle NAV.
    assert reg.gross_weight == pytest.approx(1.0)


def test_human_override_flags_surface(vault: Path):
    _write(
        vault, f"{APPROVED}/eval_a.md",
        _approved_note(
            backtest_slug="2026-06-01_grid_a",
            verdict="reject",
            guardrails_passed=False,
        ),
    )

    reg = load_registry(vault)

    cand = reg.candidates[0]
    assert "guardrails_failed (human override)" in cand.flags
    assert "agent_verdict=reject (human override)" in cand.flags
    # Human decision ist die Autorität: trägt trotzdem Gewichte.
    assert cand.deployable and cand.sleeve_weight == pytest.approx(1.0)


def test_universe_from_own_frontmatter_wins(vault: Path):
    note = _approved_note(backtest_slug="2026-06-02_grid_b").replace(
        "type: approved_strategy", "type: approved_strategy\nuniverse: uni_a"
    )
    _write(vault, f"{APPROVED}/eval_a.md", note)

    reg = load_registry(vault)

    assert reg.candidates[0].universe == "uni_a"
    assert set(reg.target_weights) == {"SPY", "QQQ"}


def test_missing_vault_dir_yields_empty_registry(tmp_path: Path):
    reg = load_registry(tmp_path)
    assert reg.candidates == [] and reg.target_weights == {}
    assert any("existiert nicht" in w for w in reg.warnings)


def test_invalid_weighting_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="weighting"):
        load_registry(tmp_path, weighting="vol_target")


# ---------------------------------------------------------------------------
# Risikomodell-Pfad (sizing != equal, braucht Sleeve-Return-Pfade)
# ---------------------------------------------------------------------------


def _sleeve_returns(vol_a: float, vol_b: float, n: int = 400, seed: int = 3) -> dict[str, list[float]]:
    """Zwei unkorrelierte Sleeves mit vorgegebener periodischer Vol."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return {
        "eval_a": list(rng.standard_normal(n) * vol_a),
        "eval_b": list(rng.standard_normal(n) * vol_b),
    }


def _two_candidates(vault: Path, score_a: float = 0.8, score_b: float = 0.4) -> None:
    _write(
        vault, f"{APPROVED}/eval_a.md",
        _approved_note(backtest_slug="2026-06-01_grid_a", score=score_a),
    )
    _write(
        vault, f"{APPROVED}/eval_b.md",
        _approved_note(backtest_slug="2026-06-02_grid_b", score=score_b),
    )


def test_risk_parity_sizing_downweights_the_volatile_sleeve(vault: Path):
    _two_candidates(vault)

    reg = load_registry(
        vault, sizing="risk_parity", sleeve_returns=_sleeve_returns(0.005, 0.020)
    )

    assert reg.sizing == "risk_parity"
    assert reg.construction is not None
    sleeves = {c.slug: c.sleeve_weight for c in reg.candidates}
    # Der ruhige Sleeve bekommt deutlich mehr Kapital ...
    assert sleeves["eval_a"] > 0.7
    # ... und beide tragen dasselbe Risiko.
    shares = {c.name: c.risk_share for c in reg.construction.contributions}
    assert shares["eval_a"] == pytest.approx(0.5, abs=1e-5)
    assert reg.gross_weight == pytest.approx(1.0)


def test_sizing_accepts_backtest_slug_keys(vault: Path):
    """Return-Pfade dürfen per Backtest-Slug kommen (so liest sie der Vault)."""
    _two_candidates(vault)
    by_eval = _sleeve_returns(0.005, 0.020)
    by_backtest = {
        "2026-06-01_grid_a": by_eval["eval_a"],
        "2026-06-02_grid_b": by_eval["eval_b"],
    }

    reg = load_registry(vault, sizing="risk_parity", sleeve_returns=by_backtest)

    assert reg.sizing == "risk_parity"
    assert reg.candidates[0].sleeve_weight > 0.7


def test_sizing_falls_back_when_a_path_is_missing(vault: Path):
    """Ein Risikomodell über nur einen Teil des Buchs wäre irreführend."""
    _two_candidates(vault)
    partial = {"eval_a": _sleeve_returns(0.005, 0.02)["eval_a"]}

    reg = load_registry(vault, sizing="risk_parity", sleeve_returns=partial)

    assert reg.sizing == "equal"
    assert reg.construction is None
    assert all(c.sleeve_weight == pytest.approx(0.5) for c in reg.candidates)
    assert any("kein Return-Pfad" in w for w in reg.warnings)


def test_sizing_falls_back_without_any_paths(vault: Path):
    _two_candidates(vault)
    reg = load_registry(vault, sizing="min_variance")
    assert reg.sizing == "equal"
    assert any("keine Sleeve-Return-Pfade" in w for w in reg.warnings)


def test_sizing_falls_back_on_too_short_history(vault: Path):
    _two_candidates(vault)
    short = {"eval_a": [0.001] * 10, "eval_b": [0.002] * 10}

    reg = load_registry(vault, sizing="risk_parity", sleeve_returns=short)

    assert reg.sizing == "equal"
    assert any("nicht möglich" in w for w in reg.warnings)


def test_score_becomes_risk_budget_under_risk_model(vault: Path):
    _two_candidates(vault, score_a=0.6, score_b=0.4)

    reg = load_registry(
        vault,
        weighting="score",
        sizing="risk_parity",
        sleeve_returns=_sleeve_returns(0.010, 0.010),
    )

    shares = {c.name: c.risk_share for c in reg.construction.contributions}
    assert shares["eval_a"] == pytest.approx(0.6, abs=1e-5)
    assert shares["eval_b"] == pytest.approx(0.4, abs=1e-5)
    assert any("Risikobudget" in w for w in reg.warnings)


def test_max_weight_constraint_binds_on_sleeves(vault: Path):
    from quantrace.portfolio import Constraints

    _two_candidates(vault)
    reg = load_registry(
        vault,
        sizing="risk_parity",
        sleeve_returns=_sleeve_returns(0.002, 0.030),
        constraints=Constraints(max_weight=0.6),
    )

    assert max(c.sleeve_weight for c in reg.candidates) == pytest.approx(0.6)
    assert reg.gross_weight == pytest.approx(1.0)


def test_vol_targeting_leaves_cash_in_the_symbol_vector(vault: Path):
    from quantrace.portfolio import Constraints

    _two_candidates(vault)
    # Sleeve-Vol ~0.02/Tag → ~32 % p.a.; Ziel 10 % ⇒ deutlich unter voll investiert.
    reg = load_registry(
        vault,
        sizing="risk_parity",
        sleeve_returns=_sleeve_returns(0.020, 0.020),
        constraints=Constraints(target_volatility=0.10),
    )

    assert reg.gross_weight < 0.6
    assert reg.construction.cash_weight == pytest.approx(1.0 - reg.gross_weight, abs=1e-6)
    assert reg.construction.annualised_volatility == pytest.approx(0.10, rel=1e-6)


def test_infeasible_governance_cap_is_relaxed_not_fatal(vault: Path):
    """max_weight=0.4 ist bei zwei Sleeves unerfüllbar — lockern, nicht scheitern."""
    from quantrace.portfolio import Constraints

    _two_candidates(vault)
    reg = load_registry(
        vault,
        sizing="risk_parity",
        sleeve_returns=_sleeve_returns(0.005, 0.020),
        constraints=Constraints(max_weight=0.4),
    )

    assert reg.sizing == "risk_parity"
    assert all(c.sleeve_weight == pytest.approx(0.5) for c in reg.candidates)
    assert any("nicht erfüllbar" in w for w in reg.warnings)
