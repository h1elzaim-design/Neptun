"""Gaussian Hidden Markov Model — from-scratch, log-space, diagonal covariance.

Why hand-rolled instead of ``hmmlearn``? The API container had OOM trouble with
heavy scientific stacks, and a regime detector should not drag scikit-learn into
the image. A diagonal-covariance Gaussian HMM is ~200 lines of NumPy and gives us
full control over initialisation, numerical stability and the causal/filtered
posterior we need for honest (non-look-ahead) regime gating.

Algorithms
----------
* **Baum-Welch (EM)** for parameter estimation, fully in log-space via
  log-sum-exp so it never underflows on long daily series.
* **Forward-backward** for smoothed posteriors γ (uses the whole sample).
* **Forward filtering** for the *causal* posterior P(sₜ | x₁..ₜ) — this is what
  a trading strategy is allowed to condition on at time *t*.
* **Viterbi** for the most-likely state path.

Conventions
-----------
``transmat_[i, j] = P(state j at t+1 | state i at t)`` (row-stochastic).
States are anonymous integer indices; semantic labelling lives in
:mod:`quantrace.regime.detector`.

Memory notes
------------
The M-step accumulates xi_sum directly in an (S, S) buffer instead of
materialising the full (T, S, S) log_xi tensor. This keeps peak memory
O(T * S) rather than O(T * S²), which matters on Render Free (512 MB).
"""

from __future__ import annotations

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log(sum(exp(a))) along ``axis``, keeping that dim."""
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    return np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max


class GaussianHMM:
    """Diagonal-covariance Gaussian HMM fit by Baum-Welch.

    Parameters
    ----------
    n_states:
        Number of hidden regimes.
    n_iter:
        Maximum EM iterations (default 30 — regime HMMs converge fast;
        lower than the old 50 to save memory on constrained runtimes).
    tol:
        Convergence threshold on the log-likelihood improvement.
    covariance_floor:
        Per-feature variances are floored at ``covariance_floor`` times the
        feature's overall variance to avoid the classic EM singularity where a
        state collapses onto a single observation.
    random_state:
        Reserved for reproducibility. Initialisation is deterministic
        (quantile split on the first feature) so results are stable without it.
    """

    def __init__(
        self,
        n_states: int = 3,
        *,
        n_iter: int = 30,
        tol: float = 1e-4,
        covariance_floor: float = 1e-3,
        random_state: int | None = 0,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        self.n_states = int(n_states)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.covariance_floor = float(covariance_floor)
        self.random_state = random_state

        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None
        self.log_likelihood_: float = -np.inf
        self.n_iter_run_: int = 0
        self.converged_: bool = False

    # -- initialisation -------------------------------------------------------

    def _init_params(self, x: np.ndarray) -> None:
        n_obs, n_features = x.shape
        k = min(self.n_states, n_obs)

        order = np.argsort(x[:, 0])
        groups = np.array_split(order, k)

        means = np.zeros((self.n_states, n_features))
        covars = np.zeros((self.n_states, n_features))
        global_var = x.var(axis=0, ddof=0)
        global_var = np.where(global_var > 0, global_var, 1.0)
        for s in range(self.n_states):
            idx = groups[s % k]
            block = x[idx]
            means[s] = block.mean(axis=0)
            covars[s] = block.var(axis=0, ddof=0)
        covars = self._floor_covars(covars, global_var)

        self.means_ = means
        self.covars_ = covars
        self.startprob_ = np.full(self.n_states, 1.0 / self.n_states)
        stay = 0.90
        off = (1.0 - stay) / max(self.n_states - 1, 1)
        transmat = np.full((self.n_states, self.n_states), off)
        np.fill_diagonal(transmat, stay if self.n_states > 1 else 1.0)
        self.transmat_ = transmat
        self._global_var = global_var

    def _floor_covars(self, covars: np.ndarray, global_var: np.ndarray) -> np.ndarray:
        floor = self.covariance_floor * global_var
        return np.maximum(covars, np.maximum(floor, 1e-12))

    # -- emission -------------------------------------------------------------

    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        """Log N(xₜ; μₛ, diag σ²ₛ) → array (n_obs, n_states)."""
        assert self.means_ is not None and self.covars_ is not None
        n_features = x.shape[1]
        diff = x[:, None, :] - self.means_[None, :, :]  # (T, S, D)
        log_det = np.sum(np.log(self.covars_), axis=1)   # (S,)
        quad = np.sum(diff**2 / self.covars_[None, :, :], axis=2)  # (T, S)
        return -0.5 * (n_features * _LOG_2PI + log_det[None, :] + quad)

    # -- E-step pieces --------------------------------------------------------

    def _forward(self, log_b: np.ndarray) -> np.ndarray:
        assert self.startprob_ is not None and self.transmat_ is not None
        n_obs = log_b.shape[0]
        # Pre-compute once — avoids a log() call inside the hot loop.
        log_t = np.log(self.transmat_ + 1e-300)  # (S, S)
        log_alpha = np.empty_like(log_b)
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + log_b[0]
        for t in range(1, n_obs):
            tmp = log_alpha[t - 1][:, None] + log_t          # (S, S)
            log_alpha[t] = log_b[t] + _logsumexp(tmp, axis=0)[0]
        return log_alpha

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        assert self.transmat_ is not None
        n_obs = log_b.shape[0]
        log_t = np.log(self.transmat_ + 1e-300)  # (S, S)
        log_beta = np.zeros_like(log_b)
        for t in range(n_obs - 2, -1, -1):
            tmp = log_t + (log_b[t + 1] + log_beta[t + 1])[None, :]  # (S, S)
            log_beta[t] = _logsumexp(tmp, axis=1)[:, 0]
        return log_beta

    # -- fit ------------------------------------------------------------------

    def fit(self, x: np.ndarray) -> GaussianHMM:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be 2-D (n_obs, n_features)")
        if x.shape[0] < 2:
            raise ValueError("need at least 2 observations to fit an HMM")

        self._init_params(x)
        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            log_b = self._log_emission(x)
            log_alpha = self._forward(log_b)
            log_beta = self._backward(log_b)

            log_likelihood = float(_logsumexp(log_alpha[-1], axis=0)[0])
            self.log_likelihood_ = log_likelihood
            self.n_iter_run_ = iteration + 1

            log_gamma = log_alpha + log_beta - log_likelihood
            gamma = np.exp(log_gamma)
            gamma_sum = gamma.sum(axis=0)  # (S,)

            self._m_step(x, log_b, log_alpha, log_beta, log_likelihood, gamma, gamma_sum)

            if log_likelihood - prev_ll < self.tol and iteration > 0:
                self.converged_ = True
                break
            prev_ll = log_likelihood
        return self

    def _m_step(
        self,
        x: np.ndarray,
        log_b: np.ndarray,
        log_alpha: np.ndarray,
        log_beta: np.ndarray,
        log_likelihood: float,
        gamma: np.ndarray,
        gamma_sum: np.ndarray,
    ) -> None:
        """M-step: update parameters from expected sufficient statistics.

        Key memory optimisation: instead of building the full (T, S, S)
        log_xi tensor and then summing over T, we accumulate xi_sum in an
        (S, S) buffer one timestep at a time.  Peak extra allocation is
        O(S²) rather than O(T * S²), which on T=2000, S=5 cuts ~160 MB of
        temporary arrays per EM iteration.
        """
        assert self.transmat_ is not None
        log_t = np.log(self.transmat_ + 1e-300)  # (S, S)
        n_obs = x.shape[0]

        # Accumulate xi_sum (S, S) one slice at a time — never build (T, S, S).
        xi_sum = np.full((self.n_states, self.n_states), -np.inf)  # log-space
        for t in range(n_obs - 1):
            # log_xi_t[i, j] = alpha[t,i] + logT[i,j] + b[t+1,j] + beta[t+1,j] - ll
            log_xi_t = (
                log_alpha[t, :, None]                     # (S, 1)
                + log_t                                   # (S, S)
                + log_b[t + 1, None, :]                   # (1, S)
                + log_beta[t + 1, None, :]                # (1, S)
                - log_likelihood
            )  # (S, S) — one timestep, no T dimension
            # log-sum-exp accumulation: log(exp(xi_sum) + exp(log_xi_t))
            mx = np.maximum(xi_sum, log_xi_t)
            xi_sum = mx + np.log(
                np.exp(xi_sum - mx) + np.exp(log_xi_t - mx)
            )
        xi_sum_lin = np.exp(xi_sum)  # (S, S)

        # Start probabilities.
        self.startprob_ = gamma[0] / max(gamma[0].sum(), 1e-300)

        # Transition matrix.
        row = xi_sum_lin.sum(axis=1, keepdims=True)
        self.transmat_ = xi_sum_lin / np.where(row > 0, row, 1e-300)

        # Emission parameters.
        denom = np.where(gamma_sum > 1e-300, gamma_sum, 1e-300)[:, None]
        means = (gamma.T @ x) / denom
        covars = np.empty_like(means)
        for s in range(self.n_states):
            diff = x - means[s]
            covars[s] = (gamma[:, s][:, None] * diff**2).sum(axis=0) / denom[s]
        self.means_ = means
        self.covars_ = self._floor_covars(covars, self._global_var)
        _ = n_obs

    # -- inference ------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.means_ is None:
            raise RuntimeError("GaussianHMM is not fitted — call fit() first")

    def predict_proba(self, x: np.ndarray, *, mode: str = "smooth") -> np.ndarray:
        """Posterior state probabilities, shape (n_obs, n_states).

        ``mode='smooth'`` returns γ (forward-backward, uses the whole sample —
        appropriate for *analysis*). ``mode='filter'`` returns the causal
        posterior P(sₜ | x₁..ₜ) which a strategy may condition on at time *t*
        without look-ahead.
        """
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        log_b = self._log_emission(x)
        log_alpha = self._forward(log_b)
        if mode == "filter":
            log_post = log_alpha - _logsumexp(log_alpha, axis=1)
            return np.exp(log_post)
        if mode == "smooth":
            log_beta = self._backward(log_b)
            log_likelihood = _logsumexp(log_alpha[-1], axis=0)[0]
            return np.exp(log_alpha + log_beta - log_likelihood)
        raise ValueError("mode must be 'smooth' or 'filter'")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Most-likely state path via Viterbi, shape (n_obs,)."""
        self._check_fitted()
        assert self.startprob_ is not None and self.transmat_ is not None
        x = np.asarray(x, dtype=float)
        log_b = self._log_emission(x)
        n_obs = log_b.shape[0]
        log_t = np.log(self.transmat_ + 1e-300)

        delta = np.empty((n_obs, self.n_states))
        psi = np.zeros((n_obs, self.n_states), dtype=int)
        delta[0] = np.log(self.startprob_ + 1e-300) + log_b[0]
        for t in range(1, n_obs):
            scores = delta[t - 1][:, None] + log_t
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = log_b[t] + np.max(scores, axis=0)

        path = np.empty(n_obs, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(n_obs - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def score(self, x: np.ndarray) -> float:
        """Log-likelihood of ``x`` under the fitted model."""
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        log_alpha = self._forward(self._log_emission(x))
        return float(_logsumexp(log_alpha[-1], axis=0)[0])
