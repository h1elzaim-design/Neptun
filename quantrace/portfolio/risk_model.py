"""Risikomodell: Kovarianz mit Ledoit-Wolf-Shrinkage — from scratch.

Warum überhaupt Shrinkage
-------------------------
Das Approved-Buch hat wenige Sleeves (N ≈ 2…10) und vergleichsweise wenige
gemeinsame Beobachtungen (T = überlappendes Fenster der persistierten
Return-Pfade). Die Sample-Kovarianz ist in diesem Regime laut — ihre
kleinsten Eigenwerte sind systematisch zu klein, und genau die invertiert
ein Min-Variance-/Mean-Variance-Optimierer am liebsten. Das Ergebnis sind
extreme, instabile Gewichte („error maximisation", Michaud 1989).

Ledoit & Wolf (2004) ziehen die Sample-Kovarianz *S* daher optimal (im
Frobenius-Sinne) auf ein strukturiertes Ziel *F* zu:

    Σ̂ = δ · F + (1 − δ) · S ,   F = μ · I ,   μ = ⟨S, I⟩ = trace(S)/N

Der Shrinkage-Faktor δ ∈ [0, 1] wird **aus den Daten** geschätzt, nicht
getunt:

    δ = b² / d²  mit  d² = ‖S − F‖²,  b² = min( b̄², d² ),
    b̄² = (1/T²) Σ_t ‖x_t x_tᵀ − S‖²

Normkonvention wie im Paper: ⟨A, B⟩ = trace(A Bᵀ) / N. Wenig Daten (großes
b̄²) ⇒ δ → 1 ⇒ fast diagonales, gut konditioniertes Ziel. Viele Daten ⇒
δ → 0 ⇒ Sample-Kovarianz.

Quelle
------
Ledoit, O. & Wolf, M. (2004), "A Well-Conditioned Estimator for
Large-Dimensional Covariance Matrices", *Journal of Multivariate Analysis*
88(2), 365–411.

Konventionen
------------
- Input ist eine (T, N)-Matrix **periodischer Returns** (nicht annualisiert).
- Ausgegeben wird die *MLE*-Kovarianz (Division durch T, nicht T−1) —
  das ist die Größe, auf der die Ledoit-Wolf-Herleitung beruht. Für
  Gewichtungszwecke ist der Faktor T/(T−1) irrelevant (Skalierung
  verschiebt keine Min-Variance-/ERC-Lösung), fürs Vol-Targeting wird er
  über ``periods_per_year`` ohnehin relativiert.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

#: Unter dieser Zahl gemeinsamer Beobachtungen ist eine Kovarianz Rauschen.
MIN_OBS = 60


@dataclass(frozen=True, slots=True)
class CovarianceEstimate:
    """Geschätzte Kovarianzmatrix samt Herkunft.

    Attributes
    ----------
    names:
        Spaltennamen in Matrix-Reihenfolge.
    matrix:
        (N, N) Kovarianz der *periodischen* Returns.
    shrinkage:
        δ ∈ [0, 1] — 0.0 bei ``method="sample"``.
    method:
        ``ledoit_wolf`` | ``sample``.
    n_obs:
        T des gemeinsamen Fensters.
    """

    names: list[str]
    matrix: np.ndarray
    shrinkage: float
    method: str
    n_obs: int

    @property
    def volatilities(self) -> np.ndarray:
        """(N,) periodische Standardabweichungen."""
        return np.sqrt(np.clip(np.diag(self.matrix), 0.0, None))

    def annualised_volatilities(self, periods_per_year: float = 252.0) -> np.ndarray:
        return self.volatilities * np.sqrt(periods_per_year)

    def to_dict(self, *, periods_per_year: float = 252.0) -> dict[str, object]:
        vols = self.annualised_volatilities(periods_per_year)
        return {
            "names": list(self.names),
            "method": self.method,
            "shrinkage": float(self.shrinkage),
            "n_obs": int(self.n_obs),
            "annualised_volatility": {
                n: float(v) for n, v in zip(self.names, vols, strict=True)
            },
            "correlation": [
                [float(x) for x in row] for row in correlation_from_covariance(self.matrix)
            ],
        }


def _as_matrix(
    returns: Mapping[str, Sequence[float]] | np.ndarray,
    names: Sequence[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Normalisiert Mapping-/Array-Input auf (Namen, (T, N)-Matrix)."""
    if isinstance(returns, Mapping):
        if not returns:
            raise ValueError("returns ist leer — Kovarianz braucht ≥ 1 Reihe")
        cols = list(returns.keys())
        arrays = [np.asarray(list(returns[c]), dtype=float) for c in cols]
        lengths = {a.size for a in arrays}
        if len(lengths) != 1:
            raise ValueError(
                f"Return-Reihen haben unterschiedliche Längen {sorted(lengths)} — "
                "upstream auf das gemeinsame Datumsfenster alignen"
            )
        matrix = np.column_stack(arrays) if arrays else np.empty((0, 0))
    else:
        matrix = np.asarray(returns, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if matrix.ndim != 2:
            raise ValueError(f"returns muss 2-dimensional sein, hat ndim={matrix.ndim}")
        cols = list(names) if names is not None else [f"asset_{i}" for i in range(matrix.shape[1])]
        if len(cols) != matrix.shape[1]:
            raise ValueError(
                f"names hat {len(cols)} Einträge, Matrix hat {matrix.shape[1]} Spalten"
            )

    if matrix.size == 0 or matrix.shape[1] == 0:
        raise ValueError("returns ist leer — Kovarianz braucht ≥ 1 Reihe")
    if not np.isfinite(matrix).all():
        raise ValueError("returns enthält non-finite Werte — upstream bereinigen")
    return cols, matrix


def sample_covariance(
    returns: Mapping[str, Sequence[float]] | np.ndarray,
    *,
    names: Sequence[str] | None = None,
    min_obs: int = MIN_OBS,
) -> CovarianceEstimate:
    """Sample-Kovarianz (MLE, Division durch T) der periodischen Returns."""
    cols, matrix = _as_matrix(returns, names)
    t_obs = matrix.shape[0]
    if t_obs < min_obs:
        raise ValueError(f"nur {t_obs} Beobachtungen — Minimum {min_obs}")
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    cov = centred.T @ centred / t_obs
    return CovarianceEstimate(
        names=cols,
        matrix=_symmetrise(cov),
        shrinkage=0.0,
        method="sample",
        n_obs=int(t_obs),
    )


def ledoit_wolf_shrinkage(
    returns: Mapping[str, Sequence[float]] | np.ndarray,
    *,
    names: Sequence[str] | None = None,
    min_obs: int = MIN_OBS,
) -> CovarianceEstimate:
    """Ledoit-Wolf-geshrunkene Kovarianz gegen das skalierte Identitäts-Ziel.

    Parameters
    ----------
    returns:
        ``name → (T,) periodische Returns`` (zeitaligned!) oder (T, N)-Matrix.
    min_obs:
        Mindest-T. Darunter ist jede Kovarianz Rauschen → ``ValueError``.

    Returns
    -------
    CovarianceEstimate
        ``shrinkage`` trägt das geschätzte δ. Bei N = 1 ist δ = 0
        (nichts zu shrinken — das Ziel wäre die Matrix selbst).
    """
    cols, matrix = _as_matrix(returns, names)
    t_obs, n_assets = matrix.shape
    if t_obs < min_obs:
        raise ValueError(f"nur {t_obs} Beobachtungen — Minimum {min_obs}")

    centred = matrix - matrix.mean(axis=0, keepdims=True)
    sample = centred.T @ centred / t_obs

    if n_assets == 1:
        return CovarianceEstimate(cols, _symmetrise(sample), 0.0, "ledoit_wolf", int(t_obs))

    # ⟨A, B⟩ = trace(A Bᵀ) / N — Normkonvention aus dem Paper.
    mu = float(np.trace(sample)) / n_assets
    target = mu * np.eye(n_assets)

    d2 = float(np.sum((sample - target) ** 2)) / n_assets
    if d2 <= 1e-30:
        # S ist bereits exakt μ·I — Shrinkage ändert nichts.
        return CovarianceEstimate(cols, _symmetrise(sample), 0.0, "ledoit_wolf", int(t_obs))

    # b̄² = (1/T²) Σ_t ‖x_t x_tᵀ − S‖²  mit der Identität
    #      Σ_t ‖x_t x_tᵀ − S‖²_F = Σ_t (x_tᵀx_t)² − T·‖S‖²_F
    sq_norms = np.einsum("ti,ti->t", centred, centred)
    b_bar2 = float(np.sum(sq_norms**2)) - t_obs * float(np.sum(sample**2))
    b_bar2 = b_bar2 / (n_assets * t_obs**2)
    b2 = min(max(b_bar2, 0.0), d2)

    delta = b2 / d2
    shrunk = delta * target + (1.0 - delta) * sample
    return CovarianceEstimate(
        names=cols,
        matrix=_symmetrise(shrunk),
        shrinkage=float(delta),
        method="ledoit_wolf",
        n_obs=int(t_obs),
    )


def correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    """Korrelationsmatrix aus einer Kovarianz; Null-Varianz-Zeilen → 0."""
    cov = np.asarray(cov, dtype=float)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    safe = np.where(std < 1e-15, 1.0, std)
    corr = cov / np.outer(safe, safe)
    degenerate = std < 1e-15
    if degenerate.any():
        corr[degenerate, :] = 0.0
        corr[:, degenerate] = 0.0
    np.fill_diagonal(corr, np.where(degenerate, 0.0, 1.0))
    return np.clip(_symmetrise(corr), -1.0, 1.0)


def _symmetrise(matrix: np.ndarray) -> np.ndarray:
    """Numerische Asymmetrie (1e-18-Größenordnung) glattbügeln."""
    return (matrix + matrix.T) / 2.0


__all__ = [
    "MIN_OBS",
    "CovarianceEstimate",
    "correlation_from_covariance",
    "ledoit_wolf_shrinkage",
    "sample_covariance",
]
