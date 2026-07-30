"""Tests für Capacity / Turnover / Break-even-bps.

Wo eine geschlossene Formel existiert (Break-even, Square-root-Law), wird
gegen sie gerechnet — nicht gegen einen eingefrorenen Zahlenwert.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantrace.stats.capacity import (
    TURNOVER_ESTIMATED,
    TURNOVER_FROM_ORDERS,
    capacity_estimate,
    cost_sensitivity,
    estimate_turnover_from_trades,
    turnover_from_orders,
)

PERIODS = 252.0


def _returns(n: int, mu: float, sigma: float, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n)


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------


def test_turnover_from_orders_matches_hand_calculation():
    # 4 Orders à 25k auf einem 100k-Konto über ein halbes Jahr (126 Tage):
    # 100k gehandelt / 100k NAV = 1.0 einseitiger Umschlag in 0.5 Jahren → 2.0 p.a.
    profile = turnover_from_orders([25_000] * 4, [100_000.0] * 126)
    assert profile.annual_turnover == pytest.approx(2.0)
    assert profile.basis == TURNOVER_FROM_ORDERS
    assert profile.is_estimate is False
    assert profile.n_orders == 4


def test_turnover_uses_absolute_notionals():
    """Verkäufe zählen wie Käufe — Turnover ist ein Betrag, keine Bilanz."""
    buys_only = turnover_from_orders([10_000, 10_000], [100_000.0] * 252)
    mixed = turnover_from_orders([10_000, -10_000], [100_000.0] * 252)
    assert mixed.annual_turnover == pytest.approx(buys_only.annual_turnover)


def test_turnover_uses_average_account_value():
    """Wächst das Konto, ist der Startwert der falsche Nenner."""
    growing = list(np.linspace(100_000, 300_000, 252))
    profile = turnover_from_orders([200_000], growing)
    assert profile.average_account_value == pytest.approx(200_000, rel=1e-3)
    assert profile.annual_turnover == pytest.approx(1.0, rel=1e-3)


def test_turnover_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="leer"):
        turnover_from_orders([1000.0], [])
    with pytest.raises(ValueError, match="≤ 0"):
        turnover_from_orders([1000.0], [0.0] * 10)
    with pytest.raises(ValueError, match="non-finite"):
        turnover_from_orders([float("nan")], [100_000.0] * 10)


def test_trade_count_estimate_counts_round_trips():
    """20 Round-Trips à 25 % Gewicht in einem Jahr = 2× 20× 0.25 = 10.0 p.a."""
    profile = estimate_turnover_from_trades(20, 252, average_position_weight=0.25)
    assert profile.annual_turnover == pytest.approx(10.0)
    assert profile.basis == TURNOVER_ESTIMATED
    assert profile.is_estimate is True


def test_trade_count_estimate_validates_weight():
    with pytest.raises(ValueError, match="average_position_weight"):
        estimate_turnover_from_trades(10, 252, average_position_weight=0.0)
    with pytest.raises(ValueError, match="average_position_weight"):
        estimate_turnover_from_trades(10, 252, average_position_weight=1.5)


# ---------------------------------------------------------------------------
# Kosten-Sensitivität + Break-even
# ---------------------------------------------------------------------------


def test_break_even_matches_closed_form():
    r = _returns(2520, mu=0.0004, sigma=0.01)
    turnover, cost_bps = 4.0, 3.0

    result = cost_sensitivity(r, turnover_annual=turnover, baseline_cost_bps=cost_bps)

    net_annual = float(r.mean()) * PERIODS
    gross_annual = net_annual + turnover * cost_bps / 1e4
    assert result.gross_annual_return == pytest.approx(gross_annual)
    assert result.break_even_bps == pytest.approx(gross_annual * 1e4 / turnover)
    assert result.cost_buffer == pytest.approx(result.break_even_bps / cost_bps)


def test_break_even_drops_out_when_strategy_loses_gross():
    r = _returns(1000, mu=-0.0005, sigma=0.01)
    result = cost_sensitivity(r, turnover_annual=2.0, baseline_cost_bps=3.0)
    assert result.break_even_bps is None
    assert result.cost_buffer is None
    assert result.survives_double_costs is False


def test_multiplier_one_reproduces_the_baseline():
    r = _returns(1000, mu=0.0004, sigma=0.01)
    result = cost_sensitivity(r, turnover_annual=3.0, baseline_cost_bps=4.0)
    at_one = next(p for p in result.points if p.multiplier == 1.0)
    assert at_one.sharpe == pytest.approx(result.baseline_sharpe)
    assert at_one.cagr == pytest.approx(result.baseline_cagr)


def test_higher_costs_monotonically_hurt():
    r = _returns(1500, mu=0.0005, sigma=0.008)
    result = cost_sensitivity(r, turnover_annual=6.0, baseline_cost_bps=3.0)
    sharpes = [p.sharpe for p in sorted(result.points, key=lambda p: p.multiplier)]
    assert sharpes == sorted(sharpes, reverse=True)


def test_high_turnover_strategy_is_more_cost_fragile():
    """Der eigentliche Punkt des Moduls: gleicher Sharpe, andere Kostenrobustheit."""
    r = _returns(2520, mu=0.0004, sigma=0.01)
    slow = cost_sensitivity(r, turnover_annual=0.5, baseline_cost_bps=3.0)
    fast = cost_sensitivity(r, turnover_annual=12.0, baseline_cost_bps=3.0)

    assert slow.baseline_sharpe == pytest.approx(fast.baseline_sharpe)
    assert slow.break_even_bps > fast.break_even_bps
    assert slow.cost_buffer > fast.cost_buffer


def test_zero_turnover_has_no_break_even():
    r = _returns(500, mu=0.0004, sigma=0.01)
    result = cost_sensitivity(r, turnover_annual=0.0, baseline_cost_bps=3.0)
    assert result.break_even_bps is None
    for p in result.points:
        assert p.sharpe == pytest.approx(result.baseline_sharpe)


def test_cost_sensitivity_rejects_bad_inputs():
    with pytest.raises(ValueError, match="zu kurz"):
        cost_sensitivity([0.01], turnover_annual=1.0, baseline_cost_bps=3.0)
    with pytest.raises(ValueError, match="non-finite"):
        cost_sensitivity([0.01, float("inf")], turnover_annual=1.0, baseline_cost_bps=3.0)
    with pytest.raises(ValueError, match="turnover_annual"):
        cost_sensitivity([0.01, 0.02], turnover_annual=-1.0, baseline_cost_bps=3.0)


def test_estimate_flag_is_carried_through():
    r = _returns(300, mu=0.0004, sigma=0.01)
    result = cost_sensitivity(
        r, turnover_annual=2.0, baseline_cost_bps=3.0, turnover_is_estimate=True
    )
    assert result.to_dict()["turnover_is_estimate"] is True


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


def test_capacity_matches_square_root_law_closed_form():
    turnover, sigma, k, limit_bps = 4.0, 0.012, 1.0, 10.0
    adv = {"SPY": 20e9}
    est = capacity_estimate(
        turnover_annual=turnover,
        weights={"SPY": 1.0},
        adv_notional=adv,
        daily_volatility=sigma,
        max_impact_bps=limit_bps,
        impact_coefficient=k,
    )
    participation = (limit_bps / (1e4 * k * sigma)) ** 2
    expected = participation * adv["SPY"] * PERIODS / turnover
    assert est.capacity_usd == pytest.approx(expected)
    assert est.binding_symbol == "SPY"


def test_impact_at_capacity_equals_the_limit():
    """Selbstkonsistenz: am Kapazitäts-AUM ist der Impact genau die Schranke."""
    limit_bps, sigma, k = 10.0, 0.012, 1.0
    est = capacity_estimate(
        turnover_annual=3.0,
        weights={"XYZ": 1.0},
        adv_notional={"XYZ": 5e8},
        daily_volatility=sigma,
        max_impact_bps=limit_bps,
        impact_coefficient=k,
        reference_aum=1_000_000.0,
    )
    at_capacity = capacity_estimate(
        turnover_annual=3.0,
        weights={"XYZ": 1.0},
        adv_notional={"XYZ": 5e8},
        daily_volatility=sigma,
        max_impact_bps=limit_bps,
        impact_coefficient=k,
        reference_aum=est.capacity_usd,
    )
    assert at_capacity.per_symbol[0].impact_bps_at_reference == pytest.approx(limit_bps)


def test_thinnest_symbol_binds_the_portfolio():
    est = capacity_estimate(
        turnover_annual=2.0,
        weights={"SPY": 0.5, "THIN": 0.5},
        adv_notional={"SPY": 20e9, "THIN": 2e6},
        daily_volatility=0.012,
    )
    assert est.binding_symbol == "THIN"
    per = {s.symbol: s for s in est.per_symbol}
    assert per["THIN"].capacity_usd < per["SPY"].capacity_usd
    assert per["THIN"].pct_adv_at_reference > per["SPY"].pct_adv_at_reference


def test_capacity_scales_linearly_with_adv_and_inversely_with_turnover():
    base = dict(weights={"A": 1.0}, daily_volatility=0.01)
    small = capacity_estimate(turnover_annual=4.0, adv_notional={"A": 1e8}, **base)
    deep = capacity_estimate(turnover_annual=4.0, adv_notional={"A": 2e8}, **base)
    churny = capacity_estimate(turnover_annual=8.0, adv_notional={"A": 1e8}, **base)

    assert deep.capacity_usd == pytest.approx(2 * small.capacity_usd)
    assert churny.capacity_usd == pytest.approx(small.capacity_usd / 2)


def test_missing_adv_is_reported_not_silently_dropped():
    est = capacity_estimate(
        turnover_annual=2.0,
        weights={"SPY": 0.5, "UNKNOWN": 0.5},
        adv_notional={"SPY": 20e9},
        daily_volatility=0.012,
    )
    assert [s.symbol for s in est.per_symbol] == ["SPY"]
    assert any("UNKNOWN" in w for w in est.warnings)


def test_zero_turnover_means_no_capacity_limit():
    est = capacity_estimate(
        turnover_annual=0.0,
        weights={"SPY": 1.0},
        adv_notional={"SPY": 20e9},
        daily_volatility=0.012,
    )
    assert est.capacity_usd is None
    assert any("kein Handel" in w for w in est.warnings)


def test_participation_limit_is_capped_at_full_adv():
    """Eine absurd hohe Schranke darf nicht > 100 % ADV implizieren."""
    est = capacity_estimate(
        turnover_annual=1.0,
        weights={"SPY": 1.0},
        adv_notional={"SPY": 1e9},
        daily_volatility=0.012,
        max_impact_bps=5_000.0,
    )
    assert any("gedeckelt" in w for w in est.warnings)
    assert est.capacity_usd == pytest.approx(1e9 * PERIODS / 1.0)


def test_weights_are_normalised():
    unnormalised = capacity_estimate(
        turnover_annual=2.0,
        weights={"A": 2.0, "B": 2.0},
        adv_notional={"A": 1e9, "B": 1e9},
        daily_volatility=0.01,
    )
    normalised = capacity_estimate(
        turnover_annual=2.0,
        weights={"A": 0.5, "B": 0.5},
        adv_notional={"A": 1e9, "B": 1e9},
        daily_volatility=0.01,
    )
    assert unnormalised.capacity_usd == pytest.approx(normalised.capacity_usd)


def test_zero_volatility_is_flagged_not_crashed():
    est = capacity_estimate(
        turnover_annual=2.0,
        weights={"A": 1.0},
        adv_notional={"A": 1e9},
        daily_volatility=0.0,
    )
    assert est.capacity_usd is None
    assert any("nicht schätzbar" in w for w in est.warnings)


def test_capacity_rejects_bad_inputs():
    with pytest.raises(ValueError, match="weights ist leer"):
        capacity_estimate(
            turnover_annual=1.0, weights={}, adv_notional={}, daily_volatility=0.01
        )
    with pytest.raises(ValueError, match="turnover_annual"):
        capacity_estimate(
            turnover_annual=-1.0,
            weights={"A": 1.0},
            adv_notional={"A": 1e9},
            daily_volatility=0.01,
        )
    with pytest.raises(ValueError, match="Summe der Gewichte"):
        capacity_estimate(
            turnover_annual=1.0,
            weights={"A": 0.0},
            adv_notional={"A": 1e9},
            daily_volatility=0.01,
        )


def test_payload_shape_is_serialisable():
    est = capacity_estimate(
        turnover_annual=2.0,
        weights={"SPY": 1.0},
        adv_notional={"SPY": 20e9},
        daily_volatility=0.012,
    )
    payload = est.to_dict()
    assert payload["method"] == "square_root_impact_law"
    assert isinstance(payload["per_symbol"], list)
    assert math.isfinite(float(payload["worst_pct_adv"]))


def test_per_symbol_volatility_is_used_where_given():
    """Ein volatiler dünner Name muss härter bestraft werden als ein ruhiger."""
    common = dict(
        turnover_annual=2.0,
        weights={"CALM": 0.5, "WILD": 0.5},
        adv_notional={"CALM": 1e8, "WILD": 1e8},
    )
    per_symbol = capacity_estimate(daily_volatility={"CALM": 0.005, "WILD": 0.030}, **common)
    per = {s.symbol: s for s in per_symbol.per_symbol}

    assert per["WILD"].daily_volatility == pytest.approx(0.030)
    assert per["WILD"].capacity_usd < per["CALM"].capacity_usd
    assert per_symbol.binding_symbol == "WILD"
    # Repräsentative Vol fürs Reporting = Median der gehandelten Namen.
    assert per_symbol.daily_volatility == pytest.approx(np.median([0.005, 0.030]))


def test_missing_symbol_volatility_falls_back_to_the_median():
    est = capacity_estimate(
        turnover_annual=2.0,
        weights={"A": 0.4, "B": 0.3, "C": 0.3},
        adv_notional={"A": 1e9, "B": 1e9, "C": 1e9},
        daily_volatility={"A": 0.01, "B": 0.02},
    )
    per = {s.symbol: s for s in est.per_symbol}
    assert per["C"].daily_volatility == pytest.approx(np.median([0.01, 0.02]))


def test_scalar_volatility_still_supported():
    est = capacity_estimate(
        turnover_annual=2.0,
        weights={"A": 1.0},
        adv_notional={"A": 1e9},
        daily_volatility=0.012,
    )
    assert est.per_symbol[0].daily_volatility == pytest.approx(0.012)


def test_empty_volatility_mapping_is_flagged():
    est = capacity_estimate(
        turnover_annual=2.0,
        weights={"A": 1.0},
        adv_notional={"A": 1e9},
        daily_volatility={},
    )
    assert est.capacity_usd is None
    assert any("nicht schätzbar" in w for w in est.warnings)
