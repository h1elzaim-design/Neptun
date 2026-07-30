"""Tests für die Portfolio-Konstruktion.

Prüfprinzip: wo eine Closed Form existiert, wird gegen sie getestet
(Min-Variance, Inverse-Vol, ERC bei Unkorreliertheit), sonst gegen die
definierende Invariante (Euler-Zerlegung, Trace-Erhaltung des
Ledoit-Wolf-Shrinkage, gleiche Risikoanteile bei ERC).
"""

from __future__ import annotations

import numpy as np
import pytest

from quantrace.portfolio import (
    Constraints,
    CovarianceEstimate,
    construct_portfolio,
    correlation_from_covariance,
    ledoit_wolf_shrinkage,
    risk_contributions,
    sample_covariance,
)

PERIODS = 252.0


def _cov(matrix: list[list[float]], names: list[str], n_obs: int = 500) -> CovarianceEstimate:
    return CovarianceEstimate(
        names=names,
        matrix=np.array(matrix, dtype=float),
        shrinkage=0.0,
        method="fixture",
        n_obs=n_obs,
    )


def _returns(n_obs: int, cov: np.ndarray, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(cov)
    return rng.standard_normal((n_obs, cov.shape[0])) @ chol.T


# ---------------------------------------------------------------------------
# Risikomodell
# ---------------------------------------------------------------------------


def test_ledoit_wolf_preserves_trace():
    """Shrinkage gegen μ·I verschiebt die Gesamtvarianz nicht."""
    true_cov = np.diag([0.0004, 0.0009, 0.0016])
    data = _returns(400, true_cov)
    sample = sample_covariance(data, names=["a", "b", "c"])
    shrunk = ledoit_wolf_shrinkage(data, names=["a", "b", "c"])
    assert np.trace(shrunk.matrix) == pytest.approx(np.trace(sample.matrix), rel=1e-12)


def test_ledoit_wolf_shrinkage_in_unit_interval_and_shrinks_more_with_less_data():
    true_cov = np.array([[0.0004, 0.0002, 0.0], [0.0002, 0.0009, 0.0001], [0.0, 0.0001, 0.0016]])
    long_run = ledoit_wolf_shrinkage(_returns(2000, true_cov), names=list("abc"))
    short_run = ledoit_wolf_shrinkage(_returns(80, true_cov), names=list("abc"))
    for est in (long_run, short_run):
        assert 0.0 <= est.shrinkage <= 1.0
    assert short_run.shrinkage > long_run.shrinkage


def test_ledoit_wolf_result_lies_between_sample_and_target():
    true_cov = np.diag([0.0004, 0.0025])
    data = _returns(300, true_cov)
    sample = sample_covariance(data, names=["a", "b"]).matrix
    shrunk = ledoit_wolf_shrinkage(data, names=["a", "b"])
    target = (np.trace(sample) / 2.0) * np.eye(2)
    expected = shrunk.shrinkage * target + (1 - shrunk.shrinkage) * sample
    assert np.allclose(shrunk.matrix, expected)
    # Shrinkage ist immer besser konditioniert als das Sample.
    assert np.linalg.cond(shrunk.matrix) <= np.linalg.cond(sample) + 1e-9


def test_covariance_rejects_misaligned_and_short_series():
    with pytest.raises(ValueError, match="unterschiedliche Längen"):
        ledoit_wolf_shrinkage({"a": [0.1] * 100, "b": [0.1] * 99})
    with pytest.raises(ValueError, match="Minimum"):
        ledoit_wolf_shrinkage({"a": [0.01] * 10, "b": [0.02] * 10})
    with pytest.raises(ValueError, match="non-finite"):
        ledoit_wolf_shrinkage({"a": [0.01] * 99 + [float("nan")], "b": [0.02] * 100})


def test_correlation_from_covariance_handles_degenerate_column():
    corr = correlation_from_covariance(np.array([[0.04, 0.0], [0.0, 0.0]]))
    assert corr[0, 0] == pytest.approx(1.0)
    assert corr[1, 1] == pytest.approx(0.0)
    assert corr[0, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Sizing — Closed-Form-Vergleiche
# ---------------------------------------------------------------------------


def test_equal_weights():
    cov = _cov([[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.16]], ["a", "b", "c"])
    result = construct_portfolio(covariance=cov, method="equal")
    assert list(result.weights.values()) == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert result.gross == pytest.approx(1.0)
    assert result.effective_n == pytest.approx(3.0)


def test_inverse_vol_matches_closed_form():
    cov = _cov([[0.04, 0.01], [0.01, 0.16]], ["a", "b"])
    result = construct_portfolio(covariance=cov, method="inverse_vol")
    inv = np.array([1 / 0.2, 1 / 0.4])
    assert list(result.weights.values()) == pytest.approx(list(inv / inv.sum()))


def test_risk_parity_equals_inverse_vol_when_uncorrelated():
    """Bei Diagonal-Kovarianz ist ERC genau die Inverse-Vol-Lösung."""
    cov = _cov([[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.16]], ["a", "b", "c"])
    erc = construct_portfolio(covariance=cov, method="risk_parity")
    inv = np.array([1 / 0.2, 1 / 0.3, 1 / 0.4])
    assert list(erc.weights.values()) == pytest.approx(list(inv / inv.sum()), rel=1e-6)
    assert erc.converged


def test_risk_parity_equalises_risk_shares_with_correlations():
    cov = _cov(
        [
            [0.0400, 0.0180, 0.0048],
            [0.0180, 0.0900, 0.0180],
            [0.0048, 0.0180, 0.1600],
        ],
        ["a", "b", "c"],
    )
    result = construct_portfolio(covariance=cov, method="risk_parity")
    shares = [c.risk_share for c in result.contributions]
    assert shares == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-6)


def test_risk_budgets_are_honoured():
    cov = _cov([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]], ["a", "b", "c"])
    budgets = {"a": 0.5, "b": 0.3, "c": 0.2}
    result = construct_portfolio(covariance=cov, method="risk_parity", risk_budgets=budgets)
    shares = {c.name: c.risk_share for c in result.contributions}
    assert shares["a"] == pytest.approx(0.5, abs=1e-6)
    assert shares["b"] == pytest.approx(0.3, abs=1e-6)
    assert shares["c"] == pytest.approx(0.2, abs=1e-6)


def test_min_variance_matches_analytical_solution_when_interior():
    matrix = np.array([[0.04, 0.005, 0.0], [0.005, 0.09, 0.01], [0.0, 0.01, 0.16]])
    cov = _cov(matrix.tolist(), ["a", "b", "c"])
    inv = np.linalg.inv(matrix)
    ones = np.ones(3)
    closed_form = inv @ ones / (ones @ inv @ ones)
    assert (closed_form > 0).all(), "Fixture muss eine innere Lösung haben"

    result = construct_portfolio(covariance=cov, method="min_variance")
    assert list(result.weights.values()) == pytest.approx(list(closed_form), abs=1e-5)
    # ... und ist tatsächlich das Varianz-Minimum.
    w = np.array(list(result.weights.values()))
    for other in (np.full(3, 1 / 3), np.array([0.5, 0.3, 0.2])):
        assert w @ matrix @ w <= other @ matrix @ other + 1e-12


def test_min_variance_respects_max_weight():
    matrix = np.array([[0.01, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.16]])
    cov = _cov(matrix.tolist(), ["a", "b", "c"])
    unconstrained = construct_portfolio(covariance=cov, method="min_variance")
    assert max(unconstrained.weights.values()) > 0.5  # ohne Cap konzentriert

    capped = construct_portfolio(
        covariance=cov, method="min_variance", constraints=Constraints(max_weight=0.5)
    )
    assert max(capped.weights.values()) <= 0.5 + 1e-9
    assert sum(capped.weights.values()) == pytest.approx(1.0)


def test_mean_variance_tilts_towards_higher_expected_return():
    matrix = np.array([[0.04, 0.0], [0.0, 0.04]])
    cov = _cov(matrix.tolist(), ["a", "b"])
    result = construct_portfolio(
        covariance=cov,
        method="mean_variance",
        expected_returns={"a": 0.002, "b": 0.0005},
        risk_aversion=5.0,
    )
    assert result.weights["a"] > result.weights["b"]
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_mean_variance_without_expected_returns_degenerates_to_min_variance():
    matrix = np.array([[0.04, 0.005], [0.005, 0.16]])
    cov = _cov(matrix.tolist(), ["a", "b"])
    mv = construct_portfolio(covariance=cov, method="mean_variance")
    minvar = construct_portfolio(covariance=cov, method="min_variance")
    assert list(mv.weights.values()) == pytest.approx(list(minvar.weights.values()), abs=1e-6)
    assert any("min_variance" in w for w in mv.warnings)


# ---------------------------------------------------------------------------
# Constraints, Turnover, Vol-Targeting
# ---------------------------------------------------------------------------


def test_constraints_projection_binds_for_heuristic_methods():
    cov = _cov([[0.0004, 0.0], [0.0, 0.16]], ["a", "b"])
    result = construct_portfolio(
        covariance=cov, method="inverse_vol", constraints=Constraints(max_weight=0.6)
    )
    assert max(result.weights.values()) == pytest.approx(0.6)
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert any("Constraints binden" in w for w in result.warnings)


def test_infeasible_constraints_raise():
    cov = _cov([[0.04, 0.0], [0.0, 0.04]], ["a", "b"])
    with pytest.raises(ValueError, match="infeasible"):
        construct_portfolio(covariance=cov, method="equal", constraints=Constraints(max_weight=0.4))
    with pytest.raises(ValueError, match="infeasible"):
        construct_portfolio(
            covariance=cov, method="equal", constraints=Constraints(min_weight=0.6, max_weight=0.9)
        )


def test_turnover_is_reported_and_capped():
    cov = _cov([[0.04, 0.0], [0.0, 0.16]], ["a", "b"])
    current = {"a": 0.0, "b": 1.0}

    uncapped = construct_portfolio(covariance=cov, method="equal", current_weights=current)
    assert uncapped.turnover == pytest.approx(1.0)  # 0.5 raus, 0.5 rein

    capped = construct_portfolio(
        covariance=cov,
        method="equal",
        current_weights=current,
        constraints=Constraints(max_turnover=0.4),
    )
    assert capped.turnover == pytest.approx(0.4)
    assert capped.weights["a"] == pytest.approx(0.2)
    assert capped.weights["b"] == pytest.approx(0.8)
    assert sum(capped.weights.values()) == pytest.approx(1.0)
    assert any("Turnover-Deckel" in w for w in capped.warnings)


def test_vol_targeting_scales_gross_and_leaves_cash():
    # Zwei unkorrelierte Sleeves mit je 20 % Jahresvol → Portfolio ~14.1 %.
    daily_var = (0.20**2) / PERIODS
    cov = _cov([[daily_var, 0.0], [0.0, daily_var]], ["a", "b"])
    result = construct_portfolio(
        covariance=cov, method="equal", constraints=Constraints(target_volatility=0.07)
    )
    assert result.annualised_volatility == pytest.approx(0.07, rel=1e-6)
    assert result.gross < 1.0
    assert result.cash_weight == pytest.approx(1.0 - result.gross)
    assert result.leverage_scalar < 1.0


def test_vol_targeting_never_levers_beyond_max_leverage():
    daily_var = (0.05**2) / PERIODS
    cov = _cov([[daily_var, 0.0], [0.0, daily_var]], ["a", "b"])
    result = construct_portfolio(
        covariance=cov, method="equal", constraints=Constraints(target_volatility=0.20)
    )
    assert result.leverage_scalar == pytest.approx(1.0)
    assert result.gross == pytest.approx(1.0)
    assert any("gedeckelt" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Reporting-Invarianten
# ---------------------------------------------------------------------------


def test_risk_contributions_satisfy_euler_decomposition():
    matrix = np.array([[0.04, 0.01], [0.01, 0.09]])
    weights = np.array([0.6, 0.4])
    contrib, shares, sigma = risk_contributions(weights, matrix)
    assert contrib.sum() == pytest.approx(sigma)
    assert shares.sum() == pytest.approx(1.0)


def test_reported_contributions_sum_to_portfolio_volatility():
    cov = _cov([[0.0004, 0.0001], [0.0001, 0.0009]], ["a", "b"])
    result = construct_portfolio(covariance=cov, method="risk_parity")
    total = sum(c.risk_contribution for c in result.contributions)
    assert total == pytest.approx(result.annualised_volatility, rel=1e-9)
    assert result.diversification_ratio >= 1.0


def test_equal_weight_reveals_risk_concentration():
    """Der Grund für das ganze Modul: gleiches Kapital ≠ gleiches Risiko."""
    cov = _cov([[0.0004, 0.0], [0.0, 0.0064]], ["ruhig", "wild"])
    equal = construct_portfolio(covariance=cov, method="equal")
    shares = {c.name: c.risk_share for c in equal.contributions}
    assert shares["wild"] > 0.9
    erc = construct_portfolio(covariance=cov, method="risk_parity")
    assert {c.name: c.risk_share for c in erc.contributions}["wild"] == pytest.approx(0.5, abs=1e-6)


def test_single_sleeve_book_is_fully_allocated():
    cov = _cov([[0.04]], ["solo"])
    result = construct_portfolio(covariance=cov, method="risk_parity")
    assert result.weights == {"solo": pytest.approx(1.0)}
    assert any("Nur ein Sleeve" in w for w in result.warnings)


def test_thin_history_is_flagged():
    true_cov = np.diag([0.0004, 0.0009, 0.0016, 0.0025])
    est = ledoit_wolf_shrinkage(_returns(60, true_cov), names=list("abcd"), min_obs=60)
    ample = construct_portfolio(covariance=est, method="risk_parity")
    assert not any("dünn geschätzt" in w for w in ample.warnings)
    est_thin = CovarianceEstimate(
        names=list("abcd"), matrix=est.matrix, shrinkage=est.shrinkage, method="lw", n_obs=6
    )
    thin = construct_portfolio(covariance=est_thin, method="risk_parity")
    assert any("dünn geschätzt" in w for w in thin.warnings)


def test_unknown_method_rejected():
    cov = _cov([[0.04]], ["a"])
    with pytest.raises(ValueError, match="method muss"):
        construct_portfolio(covariance=cov, method="black_litterman")


def test_construct_from_return_series_end_to_end():
    true_cov = np.array([[0.0004, 0.0002], [0.0002, 0.0016]])
    data = _returns(500, true_cov)
    result = construct_portfolio(
        {"sleeve_a": data[:, 0], "sleeve_b": data[:, 1]},
        method="risk_parity",
        periods_per_year=PERIODS,
    )
    assert set(result.weights) == {"sleeve_a", "sleeve_b"}
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert result.weights["sleeve_a"] > result.weights["sleeve_b"]  # ruhiger Sleeve größer
    payload = result.to_dict()
    assert payload["method"] == "risk_parity"
    assert len(payload["contributions"]) == 2
    assert 0.0 <= payload["shrinkage"] <= 1.0
