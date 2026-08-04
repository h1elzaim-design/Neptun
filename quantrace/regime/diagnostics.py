"""Regime-Robustheits-Diagnostik — Stationarität, Persistenz, Fit-Stabilität.

Drei Fragen, die ein HMM-Regime beantworten muss, bevor man Strategien darauf
konditioniert:

1. **Sind die Features stationär?** Die Gauss-Emissionen des HMM setzen
   voraus, dass Trend/Vol *innerhalb* eines Regimes aus einer stabilen
   Verteilung kommen. Ein Random-Walk-Feature macht die Regime-Definition
   zeitabhängig — der Augmented-Dickey-Fuller-Test (from scratch, kein
   statsmodels) prüft H0 "Einheitswurzel" pro Feature.
2. **Sind die Regime persistent — oder Flicker-Noise?** Erwartete Verweildauer
   aus der Transition-Matrix (geometrisch: 1/(1−p_ii)) vs. empirische
   Run-Längen des kausalen Label-Pfads; Flicker-Anteil (Runs ≤ 2 Tage);
   |λ₂| der Transition-Matrix als Persistenz-/Mixing-Maß.
3. **Ist die Regime-Definition stabil gegenüber dem Trainingsfenster?**
   Refit auf einem Präfix, kausales Decoding mit eingefrorenen Parametern,
   Label-Agreement gegen den Full-Sample-Fit.

Alle Funktionen sind rein (NumPy/pandas, kein IO); die API-Schicht
orchestriert nur.

References
----------
Dickey & Fuller (1979); Said & Dickey (1984) — augmented regression.
MacKinnon (2010) — asymptotische kritische Werte (constant, no trend).
Schwert (1989) — Lag-Faustregel p = ⌊12·(T/100)^0.25⌋.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# MacKinnon (2010), asymptotisch, Regression mit Konstante ohne Trend.
# Für Daily-Features mit T ≫ 250 ist die Finite-Sample-Korrektur vernachlässigbar.
_ADF_CRIT = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


# -----------------------------------------------------------------------------
# 1. Stationarität — Augmented Dickey-Fuller
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AdfResult:
    """ADF-Test (Konstante, kein Trend). H0: Einheitswurzel (nicht stationär).

    Statt eines interpolierten p-Werts werden die Ablehnungen gegen die
    asymptotischen MacKinnon-Schranken berichtet — ehrlicher als eine
    Pseudo-Präzision aus Tabellen-Interpolation.
    """

    feature: str
    statistic: float
    n_obs: int
    n_lags: int
    critical_values: dict[str, float]
    reject_1pct: bool
    reject_5pct: bool
    reject_10pct: bool

    @property
    def stationary_5pct(self) -> bool:
        return self.reject_5pct


def adf_test(series: pd.Series | np.ndarray, *, feature: str = "x", max_lags: int | None = None) -> AdfResult:
    """Augmented Dickey-Fuller: Δy_t = α + β·y_{t−1} + Σ γ_i·Δy_{t−i} + ε_t.

    Teststatistik ist die t-Statistik von β̂; stark negativ ⇒ H0 (Einheits-
    wurzel) abgelehnt ⇒ Serie stationär. Lags nach Schwert-Regel, gekappt so,
    dass ≥ 20 Freiheitsgrade bleiben.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    T = int(x.size)  # noqa: N806
    if T < 25:
        raise ValueError(f"ADF braucht ≥ 25 Beobachtungen, hat {T}")

    p = int(np.floor(12.0 * (T / 100.0) ** 0.25)) if max_lags is None else int(max_lags)
    p = max(0, min(p, (T - 20) // 2))

    dy = np.diff(x)  # Länge T-1
    # Zeilen t = p .. T-2 (Index in dy): Ziel dy[t], Regressoren y[t], dy[t-1..t-p], 1
    y_lag = x[p:-1]
    target = dy[p:]
    n = target.size
    cols = [np.ones(n), y_lag]
    cols += [dy[p - i : -i] for i in range(1, p + 1)]
    X = np.column_stack(cols)  # noqa: N806

    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    dof = n - X.shape[1]
    if dof < 5:
        raise ValueError("ADF: zu wenige Freiheitsgrade nach Lag-Wahl")
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_beta = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    stat = float(beta[1] / se_beta) if se_beta > 0 else 0.0

    return AdfResult(
        feature=feature,
        statistic=stat,
        n_obs=n,
        n_lags=p,
        critical_values=dict(_ADF_CRIT),
        reject_1pct=stat < _ADF_CRIT["1%"],
        reject_5pct=stat < _ADF_CRIT["5%"],
        reject_10pct=stat < _ADF_CRIT["10%"],
    )


# -----------------------------------------------------------------------------
# 2. Transition-Persistenz
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StatePersistence:
    """Persistenz eines Regimes: Modell (Transition-Matrix) vs. Empirie (Pfad)."""

    label: str
    self_transition: float          # p_ii
    expected_dwell_days: float      # 1 / (1 − p_ii), geometrische Verweildauer
    stationary_prob: float          # π_i der Kette
    occupancy: float                # empirischer Zeitanteil im Fenster
    n_runs: int
    mean_run_days: float
    max_run_days: int


@dataclass(frozen=True, slots=True)
class TransitionDiagnostics:
    states: list[StatePersistence]
    n_switches: int
    flicker_share: float            # Anteil der Runs ≤ 2 Tage
    second_eigenvalue_modulus: float  # |λ₂|: nahe 1 = träge Kette, nahe 0 = Noise
    diag_min: float                 # schwächste Selbst-Transition
    total_variation_occupancy: float  # TV(π, empirische Occupancy)


def _stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """π mit π·P = π: Links-Eigenvektor zum Eigenwert 1, normiert."""
    vals, vecs = np.linalg.eig(transmat.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, idx])
    pi = np.abs(pi)
    s = pi.sum()
    return pi / s if s > 0 else np.full(len(pi), 1.0 / len(pi))


def _runs(labels: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for lab in labels:
        if out and out[-1][0] == lab:
            out[-1] = (lab, out[-1][1] + 1)
        else:
            out.append((lab, 1))
    return out


def transition_diagnostics(
    transmat: np.ndarray,
    state_to_label: dict[int, str],
    regime_series: pd.Series,
) -> TransitionDiagnostics:
    """Modell- und Pfad-Persistenz eines gefitteten Detectors zusammenführen."""
    P = np.asarray(transmat, dtype=float)  # noqa: N806
    n = P.shape[0]
    pi = _stationary_distribution(P)

    labels_path = [str(v) for v in regime_series.tolist()]
    runs = _runs(labels_path)
    total = max(len(labels_path), 1)

    per_label_runs: dict[str, list[int]] = {}
    for lab, length in runs:
        per_label_runs.setdefault(lab, []).append(length)

    states: list[StatePersistence] = []
    occ_vec = np.zeros(n)
    for state in range(n):
        label = state_to_label.get(state, str(state))
        p_ii = float(P[state, state])
        lengths = per_label_runs.get(label, [])
        occupancy = sum(lengths) / total
        occ_vec[state] = occupancy
        states.append(
            StatePersistence(
                label=label,
                self_transition=p_ii,
                expected_dwell_days=float(1.0 / max(1.0 - p_ii, 1e-9)),
                stationary_prob=float(pi[state]),
                occupancy=float(occupancy),
                n_runs=len(lengths),
                mean_run_days=float(np.mean(lengths)) if lengths else 0.0,
                max_run_days=int(max(lengths)) if lengths else 0,
            )
        )

    eigs = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    lambda2 = float(eigs[1]) if len(eigs) > 1 else 0.0
    flicker = sum(1 for _, length in runs if length <= 2) / max(len(runs), 1)

    return TransitionDiagnostics(
        states=states,
        n_switches=max(len(runs) - 1, 0),
        flicker_share=float(flicker),
        second_eigenvalue_modulus=lambda2,
        diag_min=float(np.min(np.diag(P))),
        total_variation_occupancy=float(0.5 * np.abs(pi - occ_vec).sum()),
    )


# -----------------------------------------------------------------------------
# 3. Fit-Stabilität — Refit auf Präfix, Agreement auf dem Overlap
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RefitStability:
    """Label-Agreement zwischen Full-Sample-Fit und Präfix-Fit (frozen decode).

    Ein Detector, dessen Regime-Definition kippt, sobald man die letzten 30 %
    der Daten weglässt, ist kein belastbares Konditionierungs-Signal.
    """

    split: float
    train_end: pd.Timestamp
    n_overlap: int
    agreement: float  # Anteil identischer kausaler Labels auf dem Overlap


def refit_stability(
    prices: pd.Series | pd.DataFrame,
    *,
    n_states: int = 3,
    feature_window: int = 21,
    split: float = 0.7,
    full_detector=None,
) -> RefitStability:
    """``full_detector``: bereits auf der vollen Serie gefitteter
    :class:`RegimeDetector` — spart den redundanten dritten HMM-Fit, wenn der
    Aufrufer (z.B. :func:`regime_diagnostics`) schon einen hat. Ohne ihn wird
    hier frisch gefittet."""
    from quantrace.regime.detector import RegimeDetector
    from quantrace.regime.features import benchmark_series

    if not (0.3 <= split <= 0.9):
        raise ValueError("split must be in [0.3, 0.9]")

    bench = benchmark_series(prices)
    cut = int(len(bench) * split)
    if cut < feature_window + n_states + 2:
        raise ValueError("Zu wenig Historie für einen Präfix-Fit")

    full = full_detector or RegimeDetector(
        n_states=n_states, feature_window=feature_window
    ).fit(bench)
    prefix = RegimeDetector(n_states=n_states, feature_window=feature_window).fit(bench.iloc[:cut])

    # Beide decodieren kausal über die volle Serie; der Präfix-Detector nutzt
    # dabei eingefrorene Parameter — genau das Signal-Pattern aus
    # strategies/templates/regime_filter.py.
    full_labels = full.regime_series(bench)
    prefix_labels = prefix.regime_series(bench)
    overlap = full_labels.index.intersection(prefix_labels.index)
    agree = float((full_labels.loc[overlap] == prefix_labels.loc[overlap]).mean())

    return RefitStability(
        split=split,
        train_end=bench.index[cut - 1],
        n_overlap=int(len(overlap)),
        agreement=agree,
    )


# -----------------------------------------------------------------------------
# Orchestrierung + Verdikte
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RegimeDiagnostics:
    stationarity: list[AdfResult]
    transitions: TransitionDiagnostics
    refit: RefitStability | None
    warnings: list[str] = field(default_factory=list)


def regime_diagnostics(
    detector,
    prices: pd.Series | pd.DataFrame,
    *,
    with_refit: bool = True,
    refit_split: float = 0.7,
) -> RegimeDiagnostics:
    """Volle Robustheits-Diagnostik eines gefitteten :class:`RegimeDetector`.

    Verdikt-Heuristiken (bewusst konservativ, jede Zahl steht daneben):
    - Feature nicht stationär (ADF 5 %) → Regime-Definition zeitabhängig.
    - Flicker-Anteil > 25 % oder mittlere Run-Länge < 5 Tage → Noise statt Regime.
    - |λ₂| < 0.5 → kaum Persistenz; TV(π, Occupancy) > 0.25 → Fenster nicht
      repräsentativ für die gefittete Kette.
    - Refit-Agreement < 80 % → Definition instabil gegenüber dem Trainingsfenster.
    """
    from quantrace.regime.features import regime_features

    # fit() hat die Feature-Matrix bereits berechnet und gecacht — die
    # Rolling-Window-Konstruktion nicht pro Diagnostics-Call wiederholen.
    feats = getattr(detector, "_features", None)
    if feats is None:
        feats = regime_features(prices, window=detector.feature_window)
    stationarity = [
        adf_test(feats[col], feature=str(col)) for col in feats.columns
    ]

    series = detector.regime_series(prices)
    trans = transition_diagnostics(detector.hmm.transmat_, detector.state_to_label_, series)

    refit = None
    if with_refit:
        refit = refit_stability(
            prices,
            n_states=detector.n_states,
            feature_window=detector.feature_window,
            split=refit_split,
            full_detector=detector,
        )

    warnings: list[str] = []
    for adf in stationarity:
        if not adf.reject_5pct:
            warnings.append(
                f"Feature '{adf.feature}' nicht stationär (ADF {adf.statistic:.2f} > "
                f"{_ADF_CRIT['5%']:.2f}) — Regime-Definition kann über die Zeit driften."
            )
    if trans.flicker_share > 0.25:
        warnings.append(
            f"{trans.flicker_share:.0%} der Regime-Runs dauern ≤ 2 Tage — "
            "Flicker-Noise statt persistenter Regime."
        )
    mean_runs = [s.mean_run_days for s in trans.states if s.n_runs > 0]
    if mean_runs and min(mean_runs) < 5.0:
        worst = min(trans.states, key=lambda s: s.mean_run_days if s.n_runs else np.inf)
        warnings.append(
            f"Regime '{worst.label}' hält im Mittel nur {worst.mean_run_days:.1f} Tage — "
            "zu kurz, um Strategien darauf zu konditionieren."
        )
    if trans.second_eigenvalue_modulus < 0.5:
        warnings.append(
            f"|λ₂| = {trans.second_eigenvalue_modulus:.2f} — die Kette mischt fast "
            "instantan, Regime tragen kaum Information über morgen."
        )
    if trans.total_variation_occupancy > 0.25:
        warnings.append(
            f"Occupancy weicht stark von der stationären Verteilung ab "
            f"(TV = {trans.total_variation_occupancy:.2f}) — Fenster evtl. nicht repräsentativ."
        )
    if refit is not None and refit.agreement < 0.8:
        warnings.append(
            f"Refit-Stabilität nur {refit.agreement:.0%} Label-Agreement "
            f"(Train-Split {refit.split:.0%}) — Regime-Definition hängt am Trainingsfenster."
        )

    return RegimeDiagnostics(
        stationarity=stationarity,
        transitions=trans,
        refit=refit,
        warnings=warnings,
    )


__all__ = [
    "AdfResult",
    "RefitStability",
    "RegimeDiagnostics",
    "StatePersistence",
    "TransitionDiagnostics",
    "adf_test",
    "refit_stability",
    "regime_diagnostics",
    "transition_diagnostics",
]
