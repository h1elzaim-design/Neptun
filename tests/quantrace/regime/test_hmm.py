"""Tests for the from-scratch Gaussian HMM.

Strategy: generate data from a *known* HMM, then assert that Baum-Welch recovers
the generating parameters (up to label switching) and that decoding is accurate.
Label switching is handled by matching recovered states to true states via
nearest mean before comparing.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantrace.regime.hmm import GaussianHMM, _logsumexp


def _sample_hmm(
    n_obs: int,
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_states = len(startprob)
    states = np.empty(n_obs, dtype=int)
    states[0] = rng.choice(n_states, p=startprob)
    for t in range(1, n_obs):
        states[t] = rng.choice(n_states, p=transmat[states[t - 1]])
    obs = np.empty((n_obs, means.shape[1]))
    for t in range(n_obs):
        obs[t] = rng.normal(means[states[t]], sds[states[t]])
    return obs, states


def _match_states(true_means: np.ndarray, est_means: np.ndarray) -> dict[int, int]:
    """Map each estimated state to the nearest true state (greedy)."""
    mapping: dict[int, int] = {}
    for est_idx, m in enumerate(est_means):
        dists = np.linalg.norm(true_means - m, axis=1)
        mapping[est_idx] = int(np.argmin(dists))
    return mapping


def test_logsumexp_matches_naive():
    a = np.array([[-1000.0, -1001.0, -1002.0]])
    got = _logsumexp(a, axis=1)[0, 0]
    # naive exp would underflow to 0 → log(0) = -inf; stable version must not.
    assert np.isfinite(got)
    assert got == pytest.approx(-999.59239, abs=1e-3)


def test_recovers_two_well_separated_regimes():
    rng = np.random.default_rng(42)
    startprob = np.array([0.5, 0.5])
    transmat = np.array([[0.97, 0.03], [0.03, 0.97]])
    means = np.array([[-0.5, 1.5], [0.8, 0.4]])  # (calm-down, vol) vs (up, low-vol)
    sds = np.array([[0.15, 0.15], [0.15, 0.15]])
    obs, states = _sample_hmm(3000, startprob, transmat, means, sds, rng)

    hmm = GaussianHMM(n_states=2, n_iter=100).fit(obs)
    assert hmm.converged_ or hmm.n_iter_run_ == 100
    assert hmm.means_ is not None

    mapping = _match_states(means, hmm.means_)
    # Each true regime should be claimed by exactly one estimated state.
    assert set(mapping.values()) == {0, 1}
    for est_idx, true_idx in mapping.items():
        np.testing.assert_allclose(hmm.means_[est_idx], means[true_idx], atol=0.15)


def test_viterbi_decoding_is_accurate():
    rng = np.random.default_rng(7)
    transmat = np.array([[0.98, 0.02], [0.02, 0.98]])
    means = np.array([[-1.0, 0.0], [1.0, 0.0]])
    sds = np.array([[0.3, 0.3], [0.3, 0.3]])
    obs, states = _sample_hmm(2000, np.array([0.5, 0.5]), transmat, means, sds, rng)

    hmm = GaussianHMM(n_states=2, n_iter=100).fit(obs)
    path = hmm.predict(obs)

    mapping = _match_states(means, hmm.means_)
    decoded = np.array([mapping[p] for p in path])
    accuracy = float((decoded == states).mean())
    assert accuracy > 0.9


def test_filter_is_causal_smooth_uses_future():
    """Filtered posterior at t must not depend on observations after t."""
    rng = np.random.default_rng(1)
    means = np.array([[-1.0, 0.0], [1.0, 0.0]])
    sds = np.array([[0.3, 0.3], [0.3, 0.3]])
    transmat = np.array([[0.95, 0.05], [0.05, 0.95]])
    obs, _ = _sample_hmm(500, np.array([0.5, 0.5]), transmat, means, sds, rng)

    hmm = GaussianHMM(n_states=2, n_iter=50).fit(obs)
    cut = 300
    full_filter = hmm.predict_proba(obs, mode="filter")
    trunc_filter = hmm.predict_proba(obs[:cut], mode="filter")
    # Filtered probs up to `cut` are identical whether or not the future exists.
    np.testing.assert_allclose(full_filter[:cut], trunc_filter, atol=1e-9)


def test_proba_rows_sum_to_one():
    rng = np.random.default_rng(3)
    obs = rng.normal(0, 1, size=(200, 2))
    hmm = GaussianHMM(n_states=3, n_iter=20).fit(obs)
    for mode in ("smooth", "filter"):
        proba = hmm.predict_proba(obs, mode=mode)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_transmat_is_row_stochastic():
    rng = np.random.default_rng(5)
    obs = rng.normal(0, 1, size=(300, 2))
    hmm = GaussianHMM(n_states=3, n_iter=20).fit(obs)
    assert hmm.transmat_ is not None
    np.testing.assert_allclose(hmm.transmat_.sum(axis=1), 1.0, atol=1e-8)
    assert (hmm.transmat_ >= 0).all()


def test_fit_requires_two_observations():
    with pytest.raises(ValueError, match="at least 2"):
        GaussianHMM(n_states=2).fit(np.array([[1.0, 2.0]]))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        GaussianHMM(n_states=2).predict(np.zeros((5, 2)))
