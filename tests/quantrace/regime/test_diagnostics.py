"""Regime-Robustheits-Diagnostik: ADF (from scratch), Persistenz, Refit-Stabilität.

Der ADF-Test wird gegen Serien mit bekannter Wahrheit geprüft (AR(1) mit
starkem Mean-Reversion → stationär; Random Walk → Einheitswurzel). Die
Persistenz-Metriken gegen handkonstruierte Transition-Matrizen und Label-Pfade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantrace.regime import RegimeDetector, adf_test, refit_stability, regime_diagnostics
from quantrace.regime.diagnostics import (
    _stationary_distribution,
    transition_diagnostics,
)


def _two_regime_prices(seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    bull1 = rng.normal(0.0006, 0.006, 400)
    crash = rng.normal(-0.004, 0.030, 120)
    bull2 = rng.normal(0.0007, 0.007, 400)
    rets = np.concatenate([bull1, crash, bull2])
    idx = pd.bdate_range("2020-01-01", periods=len(rets))
    return pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx, name="px")


# --- ADF ---------------------------------------------------------------------------


def test_adf_rejects_for_strongly_stationary_ar1():
    rng = np.random.default_rng(1)
    x = np.zeros(1500)
    for t in range(1, len(x)):
        x[t] = 0.5 * x[t - 1] + rng.normal()  # klarer Mean-Reverter
    res = adf_test(x, feature="ar1")
    assert res.reject_1pct and res.reject_5pct
    assert res.statistic < -5.0


def test_adf_does_not_reject_for_random_walk():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(size=1500))  # Einheitswurzel
    res = adf_test(x, feature="rw")
    assert not res.reject_5pct


def test_adf_needs_enough_observations():
    with pytest.raises(ValueError, match="25"):
        adf_test(np.arange(10.0))


def test_adf_lag_rule_and_metadata():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    res = adf_test(x, feature="noise")
    assert res.feature == "noise"
    assert res.n_lags >= 1
    assert res.critical_values["5%"] == pytest.approx(-2.86)


# --- Transition-Persistenz ----------------------------------------------------------


def test_stationary_distribution_of_symmetric_chain():
    transmat = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi = _stationary_distribution(transmat)
    np.testing.assert_allclose(pi, [0.5, 0.5], atol=1e-9)


def test_transition_diagnostics_hand_computed():
    transmat = np.array([[0.95, 0.05], [0.20, 0.80]])
    labels = {0: "risk_on", 1: "risk_off"}
    # Pfad: 5 Tage on, 2 Tage off, 5 Tage on → 3 Runs, 2 Switches, 1 Flicker-Run.
    idx = pd.bdate_range("2024-01-01", periods=12)
    path = pd.Series(["risk_on"] * 5 + ["risk_off"] * 2 + ["risk_on"] * 5, index=idx)

    diag = transition_diagnostics(transmat, labels, path)

    on = next(s for s in diag.states if s.label == "risk_on")
    off = next(s for s in diag.states if s.label == "risk_off")
    assert on.expected_dwell_days == pytest.approx(20.0)   # 1 / 0.05
    assert off.expected_dwell_days == pytest.approx(5.0)   # 1 / 0.20
    # π ∝ (0.20, 0.05) normiert → (0.8, 0.2)
    assert on.stationary_prob == pytest.approx(0.8)
    assert off.stationary_prob == pytest.approx(0.2)
    assert on.n_runs == 2 and on.mean_run_days == pytest.approx(5.0)
    assert off.occupancy == pytest.approx(2 / 12)
    assert diag.n_switches == 2
    assert diag.flicker_share == pytest.approx(1 / 3)
    # |λ₂| einer 2-State-Kette = |p11 + p22 − 1| = 0.75.
    assert diag.second_eigenvalue_modulus == pytest.approx(0.75)
    assert diag.diag_min == pytest.approx(0.80)


# --- Refit-Stabilität + Orchestrierung ----------------------------------------------


def test_refit_stability_high_agreement_on_clear_regimes():
    px = _two_regime_prices()
    res = refit_stability(px, n_states=2, feature_window=21, split=0.7)
    assert res.n_overlap > 500
    # Zwei klar getrennte Regime: die Definition darf am Trainingsfenster
    # nicht kippen.
    assert res.agreement > 0.8


def test_regime_diagnostics_end_to_end():
    px = _two_regime_prices()
    det = RegimeDetector(n_states=2, feature_window=21).fit(px)
    diag = regime_diagnostics(det, px, with_refit=True)

    assert {a.feature for a in diag.stationarity} == {"trend", "vol"}
    assert len(diag.transitions.states) == 2
    assert all(s.expected_dwell_days > 0 for s in diag.transitions.states)
    assert 0.0 <= diag.transitions.flicker_share <= 1.0
    assert diag.refit is not None
    # Persistente, klar getrennte Regime: kein Flicker-Verdikt.
    assert not any("Flicker" in w for w in diag.warnings)


def test_regime_diagnostics_without_refit():
    px = _two_regime_prices()
    det = RegimeDetector(n_states=2, feature_window=21).fit(px)
    diag = regime_diagnostics(det, px, with_refit=False)
    assert diag.refit is None
