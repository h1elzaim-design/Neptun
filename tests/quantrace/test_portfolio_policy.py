"""Portfolio-Policy — Governance-YAML → typisierte Constraints.

Wichtigste Eigenschaft: **kein Crash bei Unsinn im YAML**. Eine kaputte
Governance-Datei darf die Registry nicht lahmlegen, sondern muss auf das
konservative Verhalten (equal, kein Hebel) zurückfallen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantrace.portfolio.policy import PortfolioPolicy


def test_defaults_are_the_pre_risk_model_behaviour():
    policy = PortfolioPolicy.from_config({})
    assert policy.sizing == "equal"
    assert policy.risk_budget_basis == "equal"
    assert policy.constraints.max_weight == 1.0
    assert policy.constraints.target_volatility is None
    assert policy.constraints.max_leverage == 1.0


def test_reads_the_portfolio_block():
    policy = PortfolioPolicy.from_config(
        {
            "portfolio": {
                "sizing": "risk_parity",
                "risk_budget_basis": "score",
                "max_weight": 0.35,
                "target_volatility": 0.12,
                "max_turnover": 0.25,
                "min_obs": 120,
            }
        }
    )
    assert policy.sizing == "risk_parity"
    assert policy.risk_budget_basis == "score"
    assert policy.constraints.max_weight == pytest.approx(0.35)
    assert policy.constraints.target_volatility == pytest.approx(0.12)
    assert policy.constraints.max_turnover == pytest.approx(0.25)
    assert policy.min_obs == 120


def test_unknown_values_fall_back_to_conservative_defaults():
    policy = PortfolioPolicy.from_config(
        {"portfolio": {"sizing": "black_litterman", "risk_budget_basis": "vibes"}}
    )
    assert policy.sizing == "equal"
    assert policy.risk_budget_basis == "equal"


def test_garbage_numbers_do_not_crash():
    policy = PortfolioPolicy.from_config(
        {"portfolio": {"max_weight": "viel", "target_volatility": "hoch", "min_obs": None}}
    )
    assert policy.constraints.max_weight == 1.0
    assert policy.constraints.target_volatility is None
    assert policy.min_obs == 60


def test_load_missing_file_yields_defaults(tmp_path: Path):
    assert PortfolioPolicy.load(tmp_path / "nope.yaml").sizing == "equal"


def test_load_broken_yaml_yields_defaults(tmp_path: Path):
    path = tmp_path / "governance.yaml"
    path.write_text("portfolio: [unbalanced\n", encoding="utf-8")
    assert PortfolioPolicy.load(path).sizing == "equal"


def test_repository_governance_file_parses():
    """Die eingecheckte governance.yaml muss immer ladbar sein."""
    repo_root = Path(__file__).resolve().parents[2]
    policy = PortfolioPolicy.load(repo_root / "config" / "research_rules" / "governance.yaml")
    assert policy.sizing in ("equal", "inverse_vol", "risk_parity", "min_variance", "mean_variance")
    assert 0.0 < policy.constraints.max_weight <= 1.0
    assert policy.to_dict()["sizing"] == policy.sizing
