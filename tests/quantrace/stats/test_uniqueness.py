"""Portfolio-Uniqueness — Korrelationen + marginaler Sleeve-Sharpe."""

from __future__ import annotations

import numpy as np
import pytest

from quantrace.stats.uniqueness import uniqueness


def _rets(mu: float, sigma: float, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(mu, sigma, n)


def test_clone_candidate_flagged_and_adds_nothing():
    # Kandidat = Buch-Strategie + Mini-Rauschen: ρ ≈ 1, ΔSharpe ≈ 0.
    base = _rets(0.0006, 0.01, 1000, seed=1)
    clone = base + _rets(0.0, 0.0005, 1000, seed=2)

    res = uniqueness(clone, {"existing": base})
    assert res.max_correlation > 0.95
    assert res.correlations[0].name == "existing"
    assert abs(res.delta_sharpe) < 0.15
    assert res.n_book == 1 and res.n_obs == 1000


def test_orthogonal_candidate_improves_book():
    # Zwei unkorrelierte Strategien mit gleichem Sharpe: Diversifikation
    # hebt den Sleeve-Sharpe um ~√2 gegenüber jeder einzelnen.
    a = _rets(0.0006, 0.01, 3000, seed=3)
    b = _rets(0.0006, 0.01, 3000, seed=4)

    res = uniqueness(b, {"a": a})
    assert abs(res.correlations[0].correlation) < 0.1
    assert res.delta_sharpe > 0.2
    assert res.sharpe_with_candidate > res.sharpe_book


def test_correlations_sorted_by_abs_value():
    base = _rets(0.0004, 0.01, 800, seed=5)
    noise = _rets(0.0004, 0.01, 800, seed=6)
    anti = -base + _rets(0.0, 0.002, 800, seed=7)  # stark negativ korreliert

    res = uniqueness(base, {"noise": noise, "anti": anti})
    assert res.correlations[0].name == "anti"  # |ρ| dominiert das Ranking
    assert res.correlations[0].correlation < -0.9
    assert res.max_correlation > 0.9  # max über |ρ|


def test_rejects_empty_book_and_short_overlap():
    r = _rets(0.0005, 0.01, 100, seed=8)
    with pytest.raises(ValueError, match="leer"):
        uniqueness(r, {})
    with pytest.raises(ValueError, match="Minimum"):
        uniqueness(r[:30], {"a": r[:30]})


def test_rejects_misaligned_lengths_and_nan():
    r = _rets(0.0005, 0.01, 200, seed=9)
    with pytest.raises(ValueError, match="aligned"):
        uniqueness(r, {"a": r[:150]})
    bad = r.copy()
    bad[7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        uniqueness(bad, {"a": r})


def test_constant_series_yields_zero_correlation():
    r = _rets(0.0005, 0.01, 200, seed=10)
    res = uniqueness(r, {"flat": np.zeros(200)})
    assert res.correlations[0].correlation == 0.0


def test_to_dict_json_friendly():
    import json

    r = _rets(0.0005, 0.01, 200, seed=11)
    d = uniqueness(r, {"a": _rets(0.0004, 0.012, 200, seed=12)}).to_dict()
    json.dumps(d)
    assert d["method"] == "equal_weight_marginal"
    assert {"name", "correlation"} == set(d["correlations"][0])
