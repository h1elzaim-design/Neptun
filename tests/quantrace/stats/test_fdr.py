"""Benjamini-Hochberg-FDR + Sharpe-p-Werte (Mertens-SE).

Die q-Werte sind die monotone Step-up-Adjustierung — `q ≤ α` ist exakt die
BH-Rejektionsregel. p-Werte für H0 "true SR ≤ 0" kommen aus derselben
Mertens-Varianz wie die PSR, d.h. Skew/Fat-Tails weiten die Fehlerbalken.
"""

from __future__ import annotations

import math

import pytest

from quantrace.stats import benjamini_hochberg, sharpe_p_value

# --- benjamini_hochberg ---------------------------------------------------------


def test_bh_hand_computed_small_case():
    # m=2: q = [0.01*2/1, 0.4*2/2] = [0.02, 0.4] — schon monoton.
    res = benjamini_hochberg([0.01, 0.4], alpha=0.05)
    assert res.q_values == pytest.approx([0.02, 0.4])
    assert res.significant == [True, False]
    assert res.n_significant == 1
    assert res.n_tests == 2


def test_bh_equal_spaced_all_reject_at_boundary():
    # p_(i) = i/m * 0.04 → jedes raw q = 0.04; alle vier verworfen bei α=0.05.
    res = benjamini_hochberg([0.01, 0.02, 0.03, 0.04], alpha=0.05)
    assert res.q_values == pytest.approx([0.04, 0.04, 0.04, 0.04])
    assert res.n_significant == 4


def test_bh_step_up_rescues_smaller_p():
    # Klassische Step-up-Eigenschaft: p=[0.04, 0.045], m=2.
    # raw: [0.08, 0.045] → monoton von hinten: [0.045, 0.045] → beide ≤ 0.05.
    res = benjamini_hochberg([0.04, 0.045], alpha=0.05)
    assert res.q_values == pytest.approx([0.045, 0.045])
    assert res.significant == [True, True]


def test_bh_preserves_input_order():
    shuffled = [0.4, 0.01]
    res = benjamini_hochberg(shuffled, alpha=0.05)
    assert res.q_values == pytest.approx([0.4, 0.02])
    assert res.significant == [False, True]


def test_bh_q_monotone_in_p_and_never_below_p():
    p = [0.001, 0.20, 0.03, 0.9, 0.049, 0.02]
    res = benjamini_hochberg(p)
    pairs = sorted(zip(p, res.q_values, strict=True))
    qs = [q for _, q in pairs]
    assert qs == sorted(qs)  # p_i ≤ p_j ⇒ q_i ≤ q_j
    for pi, qi in zip(p, res.q_values, strict=True):
        assert qi >= pi - 1e-12  # Adjustierung verkleinert nie


def test_bh_nan_counts_toward_family_but_never_rejects():
    res = benjamini_hochberg([0.001, float("nan"), 0.002], alpha=0.05)
    assert res.n_tests == 3
    assert res.significant[1] is False
    assert res.q_values[1] == 1.0
    # Die NaN-Hypothese vergrößert m und macht die echten Tests konservativer.
    solo = benjamini_hochberg([0.001, 0.002], alpha=0.05)
    assert res.q_values[0] >= solo.q_values[0]


def test_bh_empty_and_invalid_alpha():
    res = benjamini_hochberg([])
    assert res.n_tests == 0 and res.q_values == []
    with pytest.raises(ValueError, match="alpha"):
        benjamini_hochberg([0.01], alpha=1.5)


# --- sharpe_p_value ---------------------------------------------------------------


def test_p_value_half_at_zero_sharpe():
    assert sharpe_p_value(sharpe_period=0.0, n_obs=1000) == pytest.approx(0.5)


def test_p_value_small_for_strong_long_sample():
    # SR 2.0 p.a. über 8 Jahre Daily → hochsignifikant.
    p = sharpe_p_value(sharpe_period=2.0 / math.sqrt(252), n_obs=2000)
    assert p < 1e-6


def test_p_value_degenerate_sample_is_one():
    assert sharpe_p_value(sharpe_period=1.0, n_obs=2) == 1.0


def test_fat_tails_widen_p_value():
    # Gleicher Sharpe, gleiches T — höhere Kurtosis ⇒ größere SE ⇒ größeres p.
    sr = 1.0 / math.sqrt(252)
    p_gauss = sharpe_p_value(sharpe_period=sr, n_obs=500, skew=0.0, kurt=3.0)
    p_fat = sharpe_p_value(sharpe_period=sr, n_obs=500, skew=-1.0, kurt=9.0)
    assert p_fat > p_gauss


def test_more_observations_shrink_p_value():
    sr = 0.8 / math.sqrt(252)
    assert sharpe_p_value(sharpe_period=sr, n_obs=2000) < sharpe_p_value(
        sharpe_period=sr, n_obs=200
    )
