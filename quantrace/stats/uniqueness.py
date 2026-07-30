"""Portfolio-Uniqueness — was trägt ein Kandidat zum bestehenden Buch bei?

Motivation
----------
Alle übrigen Statistiken bewerten eine Strategie *isoliert*. Auf Desk-Ebene
zählt aber der marginale Beitrag: Ein Kandidat mit Sharpe 1.2 und Korrelation
0.95 zum bestehenden Sleeve ist wertlos (dasselbe Risiko noch einmal gekauft);
einer mit Sharpe 0.7 und Korrelation 0.1 verbessert das Buch. Dieses Modul
liefert genau diese Sicht:

- **Paarweise Korrelationen** des Kandidaten zu jeder Buch-Strategie.
- **Sleeve-Vergleich**: annualisierter Sharpe des gleichgewichteten Buchs
  vs. des Buchs *mit* Kandidat (gleichgewichtet über N+1) — der marginale
  Sharpe-Beitrag, den eine Aufnahme tatsächlich hätte.

Alle Inputs sind zeitalignierte Return-Reihen gleicher Länge (das Alignment —
Datums-Schnittmenge über die persistierten Equity-Pfade — besorgt der
Aufrufer). Pure numpy, kein I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from quantrace.stats.sharpe import annualised_sharpe

# Unter dieser Überlappung ist eine Korrelation Rauschen, kein Signal.
MIN_OVERLAP = 60


@dataclass(frozen=True, slots=True)
class CandidateCorrelation:
    """Korrelation des Kandidaten zu einer Buch-Strategie."""

    name: str
    correlation: float

    def to_dict(self) -> dict[str, float | str]:
        return {"name": self.name, "correlation": self.correlation}


@dataclass(frozen=True, slots=True)
class UniquenessResult:
    """Output von :func:`uniqueness`.

    Attributes
    ----------
    correlations:
        Kandidat vs. jede Buch-Strategie, absteigend nach |ρ| sortiert.
    max_correlation, mean_correlation:
        Spitzen- und Durchschnitts-|ρ| — die Redundanz-Kennzahlen.
    sharpe_candidate, sharpe_book, sharpe_with_candidate:
        Annualisierte Sharpes: Kandidat allein, gleichgewichtetes Buch,
        Buch + Kandidat (gleichgewichtet über N+1).
    delta_sharpe:
        ``sharpe_with_candidate − sharpe_book`` — der marginale Beitrag.
    n_book, n_obs:
        Buchgröße und Länge des gemeinsamen Fensters.
    """

    correlations: list[CandidateCorrelation]
    max_correlation: float
    mean_correlation: float
    sharpe_candidate: float
    sharpe_book: float
    sharpe_with_candidate: float
    delta_sharpe: float
    n_book: int
    n_obs: int

    def to_dict(self) -> dict[str, object]:
        return {
            "correlations": [c.to_dict() for c in self.correlations],
            "max_correlation": self.max_correlation,
            "mean_correlation": self.mean_correlation,
            "sharpe_candidate": self.sharpe_candidate,
            "sharpe_book": self.sharpe_book,
            "sharpe_with_candidate": self.sharpe_with_candidate,
            "delta_sharpe": self.delta_sharpe,
            "n_book": self.n_book,
            "n_obs": self.n_obs,
            "method": "equal_weight_marginal",
        }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson-ρ mit Degeneriert-Guard (konstante Reihe → 0.0)."""
    sa, sb = a.std(), b.std()
    if sa < 1e-15 or sb < 1e-15:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def uniqueness(
    candidate: Sequence[float],
    book: Mapping[str, Sequence[float]],
    *,
    periods_per_year: float = 252.0,
    min_overlap: int = MIN_OVERLAP,
) -> UniquenessResult:
    """Uniqueness-Report eines Kandidaten gegen ein Buch aligned Return-Reihen.

    Parameters
    ----------
    candidate:
        (T,) per-period Returns des Kandidaten.
    book:
        name → (T,) Returns jeder bereits approbierten Strategie —
        **zeitaligned** mit dem Kandidaten (gemeinsames Datumsfenster).
    min_overlap:
        Mindest-T; darunter ist die Statistik Rauschen → ValueError.

    Raises
    ------
    ValueError
        Leeres Buch, Längen-Mismatch, non-finite Werte oder T < min_overlap.
    """
    if not book:
        raise ValueError("book ist leer — Uniqueness braucht ≥ 1 approbierte Strategie")

    cand = np.asarray(list(candidate), dtype=float)
    t_obs = cand.size
    if t_obs < min_overlap:
        raise ValueError(f"nur {t_obs} gemeinsame Beobachtungen — Minimum {min_overlap}")
    if not np.isfinite(cand).all():
        raise ValueError("candidate enthält non-finite Werte — upstream alignen")

    cols: dict[str, np.ndarray] = {}
    for name, series in book.items():
        arr = np.asarray(list(series), dtype=float)
        if arr.size != t_obs:
            raise ValueError(
                f"Buch-Reihe '{name}' hat T={arr.size}, Kandidat T={t_obs} — "
                "Reihen müssen aligned sein"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"Buch-Reihe '{name}' enthält non-finite Werte")
        cols[name] = arr

    correlations = sorted(
        (CandidateCorrelation(name=n, correlation=_corr(cand, a)) for n, a in cols.items()),
        key=lambda c: abs(c.correlation),
        reverse=True,
    )
    abs_corrs = [abs(c.correlation) for c in correlations]

    matrix = np.column_stack(list(cols.values()))
    book_returns = matrix.mean(axis=1)
    with_candidate = np.column_stack([matrix, cand]).mean(axis=1)

    sharpe_book = annualised_sharpe(book_returns, periods_per_year=periods_per_year)
    sharpe_with = annualised_sharpe(with_candidate, periods_per_year=periods_per_year)

    return UniquenessResult(
        correlations=correlations,
        max_correlation=float(max(abs_corrs)),
        mean_correlation=float(np.mean(abs_corrs)),
        sharpe_candidate=annualised_sharpe(cand, periods_per_year=periods_per_year),
        sharpe_book=sharpe_book,
        sharpe_with_candidate=sharpe_with,
        delta_sharpe=sharpe_with - sharpe_book,
        n_book=len(cols),
        n_obs=int(t_obs),
    )
