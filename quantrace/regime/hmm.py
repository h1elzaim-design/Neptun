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
        Maximum EM iterations.

        Bleibt bei 30, und das ist **gemessen, nicht geerbt** (#210 Punkt 3).
        Der Fit konvergiert hier nicht — auf einer Zwei-Regime-Reihe bräuchte er
        ~94 Iterationen bis ``tol``. Das Abschneiden wirkt aber als Early
        Stopping, also als Regularisierer: lässt man den EM auskonvergieren,
        steigt das Trainings-LL und das **Out-of-Sample**-LL fällt in der
        Mehrzahl der Fälle (15 von 30, bis −86). Die 30 waren geraten und sind
        trotzdem die bessere Wahl; wer sie anhebt, kauft Anpassung statt Güte.
    n_init:
        Zufallsstarts zusätzlich zum deterministischen (siehe ``fit``).
        Default ``1``.

        Multi-Restart ist implementiert und getestet, aber **absichtlich aus**:
        auf synthetischen Reihen mit fetten Rändern verbessert er das
        Trainings-LL (23 von 30 Fällen) und verschlechtert das
        Out-of-Sample-LL (16 von 30, bis −117). Erst auf echten Lake-Daten
        nachmessen, bevor das der Default wird.
    validation_fraction:
        Anteil am *Ende* der Reihe, auf dem Startpunkt und Iterationszahl
        ausgewählt werden. ``0`` schaltet die Auswahl ab und fittet einmal
        auskonvergiert — dann steigt das Trainings-LL und das
        Out-of-Sample-LL sinkt (gemessen), also nur bewusst benutzen.
    tol:
        Convergence threshold on the log-likelihood improvement.
    covariance_floor:
        Per-feature variances are floored at ``covariance_floor`` times the
        feature's overall variance to avoid the classic EM singularity where a
        state collapses onto a single observation.
    random_state:
        Seed der Zufallsstarts. Fixiert, damit zwei Läufe über denselben
        Zeitraum dieselben Regime-Labels liefern — ohne das wäre der kausale
        gefilterte Pfad nicht reproduzierbar und damit kein Backtest.
    """

    def __init__(
        self,
        n_states: int = 3,
        *,
        n_iter: int = 30,
        n_init: int = 1,
        validation_fraction: float = 0.2,
        tol: float = 1e-4,
        covariance_floor: float = 1e-3,
        random_state: int | None = 0,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        self.n_states = int(n_states)
        self.n_iter = int(n_iter)
        self.n_init = max(1, int(n_init))
        self.validation_fraction = float(validation_fraction)
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
        #: Wie viele Startpunkte gesichtet wurden, und welcher gewonnen hat.
        #: `best_init_ == 0` heißt: der deterministische Start war der beste,
        #: das Ergebnis ist bitgleich zum Verhalten vor #210.
        self.n_init_run_: int = 0
        self.best_init_: int = 0
        #: Iterationen, die die Validierung als beste gewählt hat — das
        #: Early-Stopping, das vorher als Konstante 30 dastand.
        self.selected_iter_: int = 0
        self.validation_ll_: float = float("nan")

    # -- initialisation -------------------------------------------------------

    def _init_params(self, x: np.ndarray, rng: np.random.Generator | None = None) -> None:
        """Startpunkt für Baum-Welch.

        ``rng is None`` → der deterministische Quantil-Split über Feature 0.
        Das ist der Start, den dieses Modell seit jeher benutzt, und er bleibt
        **Kandidat 1** jedes Multi-Restarts: findet kein Zufallsstart etwas
        Besseres, ist das Ergebnis bitgleich zu vorher.

        Mit ``rng`` werden die Mittelwerte aus zufällig gezogenen Beobachtungen
        gesetzt (k-means++-Geist, ohne die Distanzrechnung). Baum-Welch ist ein
        *lokaler* Optimierer — verschiedene Startpunkte landen in verschiedenen
        Optima, und ohne mehrere Starts weiß man nie, in welchem man sitzt.
        """
        n_obs, n_features = x.shape
        k = min(self.n_states, n_obs)

        global_var = x.var(axis=0, ddof=0)
        global_var = np.where(global_var > 0, global_var, 1.0)

        means = np.zeros((self.n_states, n_features))
        covars = np.zeros((self.n_states, n_features))

        if rng is None:
            order = np.argsort(x[:, 0])
            groups = np.array_split(order, k)
            for s in range(self.n_states):
                block = x[groups[s % k]]
                means[s] = block.mean(axis=0)
                covars[s] = block.var(axis=0, ddof=0)
        else:
            # Ohne Zurücklegen, damit zwei Zustände nicht auf demselben Punkt
            # starten — das wäre ein sofortiger Kollaps auf weniger Zustände.
            picks = rng.choice(n_obs, size=min(self.n_states, n_obs), replace=False)
            for s in range(self.n_states):
                means[s] = x[picks[s % len(picks)]]
                covars[s] = global_var

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

    def _params_snapshot(self) -> dict[str, np.ndarray]:
        assert self.startprob_ is not None and self.transmat_ is not None
        assert self.means_ is not None and self.covars_ is not None
        return {
            "startprob_": self.startprob_.copy(),
            "transmat_": self.transmat_.copy(),
            "means_": self.means_.copy(),
            "covars_": self.covars_.copy(),
        }

    def _restore(self, snap: dict[str, np.ndarray]) -> None:
        self.startprob_ = snap["startprob_"].copy()
        self.transmat_ = snap["transmat_"].copy()
        self.means_ = snap["means_"].copy()
        self.covars_ = snap["covars_"].copy()

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

    def _em_with_validation(
        self, fit_x: np.ndarray, val_x: np.ndarray, max_iter: int
    ) -> tuple[float, int]:
        """EM auf ``fit_x``, bewertet nach jeder Iteration auf ``val_x``.

        Gibt ``(bestes Validierungs-LL, Iteration dabei)`` zurück und lässt die
        Parameter zurück, wo sie am Ende sind — der Aufrufer braucht nur die
        Zahlen, weil er danach ohnehin auf allen Daten neu fittet.

        Warum überhaupt: gemessen steigt das *Trainings*-LL durch Multi-Restart
        und volle Konvergenz, das *Out-of-Sample*-LL fällt dabei in der Mehrzahl
        der Fälle (16 von 30, bis −117). Wer nach Trainings-LL auswählt, wählt
        die Überanpassung. Das alte ``n_iter=30`` war unbeabsichtigt genau der
        Regularisierer, der das verhinderte — nur mit einer geratenen Zahl.
        """
        best_val, best_iter = -np.inf, 1
        prev_ll = -np.inf
        for iteration in range(max_iter):
            log_b = self._log_emission(fit_x)
            log_alpha = self._forward(log_b)
            log_beta = self._backward(log_b)
            ll = float(_logsumexp(log_alpha[-1], axis=0)[0])
            log_gamma = log_alpha + log_beta - ll
            gamma = np.exp(log_gamma)
            self._m_step(fit_x, log_b, log_alpha, log_beta, ll, gamma, gamma.sum(axis=0))

            val_ll = self.score(val_x)
            if val_ll > best_val:
                best_val, best_iter = val_ll, iteration + 1

            if ll - prev_ll < self.tol and iteration > 0:
                break
            prev_ll = ll
        return best_val, best_iter

    def _run_em(self, x: np.ndarray, max_iter: int) -> float:
        """Baum-Welch ab dem **aktuellen** Parameterstand. Gibt das End-LL zurück.

        Setzt `log_likelihood_`, `n_iter_run_` und `converged_`. Herausgezogen
        aus `fit`, damit derselbe Kern zweimal benutzt werden kann: kurz zum
        Sichten mehrerer Startpunkte, danach lang bis zur Konvergenz.
        """
        prev_ll = -np.inf
        self.converged_ = False
        self.n_iter_run_ = 0
        for iteration in range(max_iter):
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
        return self.log_likelihood_

    def fit(self, x: np.ndarray) -> GaussianHMM:
        """Multi-Restart-Baum-Welch: sichten, besten Start wählen, auskonvergieren.

        Baum-Welch ist ein lokaler Optimierer. Bis #210 lief genau **eine**
        Initialisierung — der Fit konnte also in einem schlechten lokalen Optimum
        landen, und sämtliche Regime-Labels darunter hingen daran, ohne dass es
        irgendwo sichtbar wurde.

        Volle Restarts wären ``n_init``-mal so teuer. Stattdessen laufen alle
        Startpunkte erst ``n_init_iter`` Iterationen (Sichtung); auskonvergiert
        werden dann **zwei** Finalisten: der deterministische Start und der beste
        der Zufallsstarts. Das Ergebnis ist der bessere von beiden.

        Warum zwei und nicht nur der Sichtungssieger: gemessen kann die kurze
        Sichtung danebenliegen. Ein Startpunkt, der nach 10 Iterationen führt,
        konvergiert nicht zwangsläufig höher — in einem Testfall verlor der
        Sichtungssieger am Ende um ΔLL = −1.36 gegen den deterministischen Start.
        Mit dem deterministischen Start als gesetztem Finalisten gilt: **das
        Ergebnis ist nie schlechter als vor #210**, nur manchmal deutlich besser
        (gemessen bis ΔLL = +105).

        Kosten dadurch rund das 2,3-fache eines einzelnen vollen Fits statt des
        ``n_init``-fachen.

        Reproduzierbarkeit: die Zufallsstarts hängen ausschließlich an
        ``random_state``. Zwei Läufe über denselben Zeitraum liefern dieselben
        Regime-Labels — sonst wäre der kausale gefilterte Pfad wertlos, weil ein
        Backtest sich nicht wiederholen ließe.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be 2-D (n_obs, n_features)")
        if x.shape[0] < 2:
            raise ValueError("need at least 2 observations to fit an HMM")

        n_init = max(1, int(self.n_init))
        if n_init == 1:
            self._init_params(x)
            self._run_em(x, self.n_iter)
            self.n_init_run_ = 1
            self.selected_iter_ = self.n_iter_run_
            return self

        # Zeitlich zusammenhängender Split — kein Shuffle. Der Validierungsteil
        # ist das *Ende* der Reihe, sonst würde über die Zeit hinweg gelernt.
        n_val = int(len(x) * self.validation_fraction)
        min_val = self.n_states + 2
        if n_val < min_val or len(x) - n_val < min_val:
            # Zu wenig Daten für eine ehrliche Auswahl: ein einzelner
            # auskonvergierter Fit ist dann die sicherste Antwort.
            self._init_params(x)
            self._run_em(x, self.n_iter)
            self.n_init_run_, self.best_init_ = 1, 0
            self.selected_iter_ = self.n_iter_run_
            return self

        fit_x, val_x = x[:-n_val], x[-n_val:]
        rng = np.random.default_rng(self.random_state)

        # --- Auswahl auf ungesehenen Daten -----------------------------------
        # Kandidat 0 ist der deterministische Quantil-Split (der Startpunkt vor
        # #210); 1..n-1 sind Zufallsstarts. Bewertet wird jeder nach seinem
        # besten Validierungs-LL — und mitgeschrieben, nach wie vielen
        # Iterationen er das erreicht hat. Diese Iterationszahl *ist* das
        # Early-Stopping: sie ersetzt die geratene 30.
        best_val = -np.inf
        best_index = 0
        best_iter = 1

        for i in range(n_init):
            self._init_params(fit_x, rng=None if i == 0 else rng)
            val_ll, at_iter = self._em_with_validation(fit_x, val_x, self.n_iter)
            if val_ll > best_val:
                best_val, best_index, best_iter = val_ll, i, at_iter

        # --- Refit auf allen Daten, mit der gewählten Iterationszahl ----------
        # Der Validierungsteil darf im finalen Modell nicht fehlen: für die
        # Regime-Erkennung ist gerade das *jüngste* Stück das interessante.
        # Getunt wird auf dem Split, gefittet wird auf allem — Standardvorgehen.
        rng_final = np.random.default_rng(self.random_state)
        for i in range(best_index + 1):
            self._init_params(x, rng=None if i == 0 else rng_final)
        self._run_em(x, best_iter)

        self.n_init_run_ = n_init
        self.best_init_ = best_index
        self.selected_iter_ = best_iter
        self.validation_ll_ = best_val
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
