"""Unit tests for quantrace.stats.sharpe.

These are mathematical assertions, not regressions. Each test encodes a
property the implementation must satisfy by definition (idempotence,
known-value cases, monotonicity).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantrace.stats.sharpe import (
    _expected_max_sharpe_period,
    _inv_phi,
    _phi,
    _sample_skew_kurt,
    annualised_sharpe,
    deflated_sharpe_from_summary,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_from_summary,
    probabilistic_sharpe_ratio,
)


class TestNormalHelpers:
    def test_phi_at_zero_is_half(self):
        assert _phi(0.0) == pytest.approx(0.5, abs=1e-12)

    def test_phi_known_quantiles(self):
        assert _phi(1.96) == pytest.approx(0.975, abs=1e-4)
        assert _phi(-1.96) == pytest.approx(0.025, abs=1e-4)
        assert _phi(2.33) == pytest.approx(0.99, abs=1e-3)

    def test_inv_phi_is_inverse_of_phi(self):
        for z in (-2.5, -1.0, -0.5, 0.5, 1.0, 2.5):
            assert _inv_phi(_phi(z)) == pytest.approx(z, abs=1e-6)

    def test_inv_phi_known_quantiles(self):
        assert _inv_phi(0.975) == pytest.approx(1.96, abs=1e-3)
        assert _inv_phi(0.5) == pytest.approx(0.0, abs=1e-9)
        assert _inv_phi(0.025) == pytest.approx(-1.96, abs=1e-3)

    def test_inv_phi_rejects_invalid(self):
        with pytest.raises(ValueError):
            _inv_phi(0.0)
        with pytest.raises(ValueError):
            _inv_phi(1.0)


class TestAnnualisedSharpe:
    def test_zero_volatility_returns_zero(self):
        assert annualised_sharpe([0.001] * 100) == 0.0

    def test_too_few_observations_returns_zero(self):
        assert annualised_sharpe([0.01]) == 0.0
        assert annualised_sharpe([]) == 0.0

    def test_matches_manual_formula(self):
        rng = np.random.default_rng(7)
        r = rng.normal(0.0005, 0.012, size=1000)
        expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
        assert annualised_sharpe(r) == pytest.approx(expected)


class TestProbabilisticSharpe:
    def test_observed_equals_benchmark_gives_psr_half(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.0010, 0.012, size=500)
        sr_annual = annualised_sharpe(r)
        out = probabilistic_sharpe_ratio(r, benchmark_sharpe_annual=sr_annual)
        # σ_SR > 0 but numerator = 0 → Φ(0) = 0.5
        assert out.psr == pytest.approx(0.5, abs=1e-3)

    def test_observed_above_benchmark_above_half(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.0015, 0.010, size=1000)  # high SR
        out = probabilistic_sharpe_ratio(r, benchmark_sharpe_annual=0.0)
        assert out.psr > 0.5
        assert out.sr_period_observed > 0.0

    def test_longer_history_gives_higher_psr_at_same_sr(self):
        # Same per-period drift, longer T → smaller σ_SR → higher PSR
        rng_short = np.random.default_rng(3)
        rng_long = np.random.default_rng(3)
        short = rng_short.normal(0.0008, 0.012, size=252)
        long_ = rng_long.normal(0.0008, 0.012, size=2520)
        psr_short = probabilistic_sharpe_ratio(short).psr
        psr_long = probabilistic_sharpe_ratio(long_).psr
        assert psr_long > psr_short

    def test_degenerate_inputs_return_zero(self):
        out = probabilistic_sharpe_ratio([0.01, 0.01], benchmark_sharpe_annual=0.0)
        assert out.psr == 0.0
        out = probabilistic_sharpe_ratio([])
        assert out.psr == 0.0


class TestExpectedMaxSharpePeriod:
    def test_no_trial_variance_yields_zero(self):
        assert _expected_max_sharpe_period(n_trials=10, cross_trial_sharpe_variance=0.0) == 0.0

    def test_more_trials_means_higher_expected_max(self):
        v = 0.01  # per-period sharpe variance
        n10 = _expected_max_sharpe_period(10, v)
        n100 = _expected_max_sharpe_period(100, v)
        n1000 = _expected_max_sharpe_period(1000, v)
        assert n10 < n100 < n1000

    def test_single_trial_yields_zero(self):
        # Selection has no effect with a single trial
        assert _expected_max_sharpe_period(n_trials=1, cross_trial_sharpe_variance=0.5) == 0.0


class TestDeflatedSharpe:
    def test_more_trials_strictly_reduces_dsr(self):
        rng = np.random.default_rng(11)
        r = rng.normal(0.0010, 0.010, size=1500)
        dsr_few = deflated_sharpe_ratio(
            r, n_trials=10, cross_trial_sharpe_variance=0.005,
        ).dsr
        dsr_many = deflated_sharpe_ratio(
            r, n_trials=500, cross_trial_sharpe_variance=0.005,
        ).dsr
        assert dsr_few > dsr_many

    def test_single_trial_raises(self):
        rng = np.random.default_rng(12)
        r = rng.normal(0.001, 0.01, size=100)
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(r, n_trials=1, cross_trial_sharpe_variance=0.01)

    def test_zero_trial_variance_recovers_psr_vs_zero_benchmark(self):
        # No selection effect → DSR ≈ PSR(0)
        rng = np.random.default_rng(13)
        r = rng.normal(0.0010, 0.010, size=2000)
        psr = probabilistic_sharpe_ratio(r, benchmark_sharpe_annual=0.0).psr
        dsr = deflated_sharpe_ratio(
            r, n_trials=100, cross_trial_sharpe_variance=0.0,
        ).dsr
        assert dsr == pytest.approx(psr, abs=1e-9)


class TestProbabilisticSharpeFromSummary:
    def test_matches_returns_based_psr_on_same_moments(self):
        # The summary path must reproduce the returns path exactly when fed the
        # moments the returns path would compute itself.
        rng = np.random.default_rng(21)
        r = rng.normal(0.0012, 0.011, size=900)
        ref = probabilistic_sharpe_ratio(r, benchmark_sharpe_annual=0.0)
        skew, kurt = _sample_skew_kurt(r)
        out = probabilistic_sharpe_from_summary(
            observed_sharpe_period=ref.sr_period_observed,
            n_obs=ref.n_obs,
            benchmark_sharpe_period=0.0,
            skew=skew,
            kurt=kurt,
        )
        assert out.psr == pytest.approx(ref.psr, abs=1e-12)
        assert out.sigma_sr == pytest.approx(ref.sigma_sr, abs=1e-12)

    def test_gaussian_variance_collapses_to_closed_form(self):
        # γ₃=0, γ₄=3 → σ_SR² = (1 + ½·SR̂²)/(T−1)
        sr, T = 0.08, 500  # noqa: N806 — T matches the formula in the comment above
        out = probabilistic_sharpe_from_summary(observed_sharpe_period=sr, n_obs=T)
        expected = math.sqrt((1.0 + 0.5 * sr**2) / (T - 1))
        assert out.sigma_sr == pytest.approx(expected, abs=1e-12)

    def test_too_few_observations_is_safe(self):
        out = probabilistic_sharpe_from_summary(observed_sharpe_period=0.1, n_obs=2)
        assert out.psr == 0.0
        assert out.sigma_sr == 0.0


class TestDeflatedSharpeFromSummary:
    def test_more_trials_strictly_reduces_dsr(self):
        trials = [0.4, 0.55, 0.62, 0.71, 0.83]  # annualised-ish, per-period units here
        few = deflated_sharpe_from_summary(
            observed_sharpe_period=0.083, trial_sharpes_period=trials, n_obs=1500, n_trials=10,
        ).dsr
        many = deflated_sharpe_from_summary(
            observed_sharpe_period=0.083, trial_sharpes_period=trials, n_obs=1500, n_trials=500,
        ).dsr
        assert few > many

    def test_variance_and_expected_max_are_consistent(self):
        trials = [0.02, 0.04, 0.06, 0.08, 0.10]
        out = deflated_sharpe_from_summary(
            observed_sharpe_period=0.10, trial_sharpes_period=trials, n_obs=1000,
        )
        v = float(np.var(trials, ddof=1))
        assert out.cross_trial_sharpe_variance == pytest.approx(v, abs=1e-15)
        assert out.n_trials == len(trials)
        assert out.expected_max_sharpe_period == pytest.approx(
            _expected_max_sharpe_period(len(trials), v), abs=1e-12
        )

    def test_dsr_is_psr_against_expected_max(self):
        trials = [0.03, 0.05, 0.07, 0.09]
        out = deflated_sharpe_from_summary(
            observed_sharpe_period=0.09, trial_sharpes_period=trials, n_obs=800,
        )
        psr_vs_max = probabilistic_sharpe_from_summary(
            observed_sharpe_period=0.09,
            n_obs=800,
            benchmark_sharpe_period=out.expected_max_sharpe_period,
        ).psr
        assert out.dsr == pytest.approx(psr_vs_max, abs=1e-12)

    def test_fewer_than_two_trials_raises(self):
        with pytest.raises(ValueError):
            deflated_sharpe_from_summary(
                observed_sharpe_period=0.1, trial_sharpes_period=[0.1], n_obs=500,
            )
        with pytest.raises(ValueError):
            deflated_sharpe_from_summary(
                observed_sharpe_period=0.1, trial_sharpes_period=[0.1, 0.2], n_obs=500, n_trials=1,
            )


class TestExpectedMaxSharpe:
    def test_under_two_trials_is_zero(self):
        assert expected_max_sharpe([1.5]) == 0.0
        assert expected_max_sharpe([]) == 0.0

    def test_zero_dispersion_is_zero(self):
        # Identical trial Sharpes ⇒ no selection effect.
        assert expected_max_sharpe([1.2, 1.2, 1.2, 1.2]) == 0.0

    def test_more_trials_raise_the_bar(self):
        few = [1.0 + 0.02 * i for i in range(10)]
        many = [1.0 + 0.02 * i for i in range(200)]
        assert expected_max_sharpe(many) > expected_max_sharpe(few) > 0.0

    def test_clustered_trials_deflate_less_than_dispersed(self):
        # Same count, same max — tighter dispersion ⇒ lower expected maximum.
        tight = [1.8 - 0.005 * i for i in range(20)]
        wide = [1.8 - 0.060 * i for i in range(20)]
        assert expected_max_sharpe(tight) < expected_max_sharpe(wide)

    def test_matches_internal_period_engine(self):
        trials = [0.5, 0.8, 1.1, 1.4, 1.7]
        ppy = 252.0
        v_period = float(np.var([s / np.sqrt(ppy) for s in trials], ddof=1))
        expected = _expected_max_sharpe_period(len(trials), v_period) * np.sqrt(ppy)
        assert expected_max_sharpe(trials) == pytest.approx(expected, abs=1e-12)
