"""Portfolio-Konstruktion: Ziel-Gewichte aus einem Risikomodell — from scratch.

Was hier passiert (und in welcher Reihenfolge)
----------------------------------------------
1. **Risikomodell** — Kovarianz der Sleeve-Returns, per Default
   Ledoit-Wolf-geshrunken (:mod:`quantrace.portfolio.risk_model`).
2. **Sizing** — eine von fünf Methoden (:data:`SIZING_METHODS`):

   ``equal``
       1/n. Der bisherige Stand; bleibt der ehrliche Default, wenn zu wenig
       gemeinsame Historie da ist (keine Kovarianz = keine Meinung).
   ``inverse_vol``
       wᵢ ∝ 1/σᵢ. Naive Risikoparität — ignoriert Korrelationen, ist aber
       robust und braucht nur die Diagonale.
   ``risk_parity``
       Equal Risk Contribution (ERC): jeder Sleeve trägt denselben Anteil am
       Portfoliorisiko, Korrelationen inklusive. Gelöst per **cyclical
       coordinate descent** über das konvexe Hilfsproblem
       min ½·wᵀΣw − Σ bᵢ·ln wᵢ (Spinu 2013; Griveau-Billion et al. 2013).
       Mit ``risk_budgets`` wird daraus allgemeines **Risk-Budgeting**
       (z.B. Governance-Score als Risikobudget statt als Kapitalgewicht).
   ``min_variance``
       min wᵀΣw unter den Constraints. Projizierter Gradientenabstieg;
       ohne bindende Constraints reproduziert das die Closed Form
       Σ⁻¹1 / (1ᵀΣ⁻¹1) (so getestet).
   ``mean_variance``
       max wᵀμ − (λ/2)·wᵀΣw. Braucht ``expected_returns``; ohne sie ist es
       min_variance. λ = ``risk_aversion``.

3. **Constraints** (:class:`Constraints`) — Long-only-Box [min_weight,
   max_weight] und Ziel-Brutto-Exposure, durchgesetzt als exakte Projektion
   auf {w : lo ≤ w ≤ hi, Σw = gross} (Bisektion über den Dual-Parameter).
   Heuristische Methoden (equal/inverse_vol/risk_parity) werden nach dem
   Sizing auf dieselbe Menge projiziert; die Optimierer haben sie im
   Iterationsschritt drin.
4. **Turnover-Deckel** — optional. Statt einer Penalty im Ziel (deren λ
   niemand kalibrieren kann) wird der Weg zum Ziel skaliert:
   w = w_prev + α·(w* − w_prev) mit dem größten α ≤ 1, das
   Σ|w − w_prev| ≤ max_turnover einhält. Partielles Rebalancing, exakt
   und ohne Tuning-Knopf. Turnover-Konvention = Σ|Δw| (one-way, wie
   :mod:`quantrace.paper.rebalance`).
5. **Vol-Targeting** — optional. Skaliert das Brutto-Exposure auf eine
   annualisierte Ziel-Vol; ``max_leverage`` deckelt (Default 1.0 = es darf
   nur *de*-risked werden, nie gehebelt). Der Rest ist Cash.

Grenzen (bewusst)
-----------------
- Kovarianz aus *realisierten* Sleeve-Returns, kein Faktor-Risikomodell im
  Barra-Stil. Bei wenig Historie ist auch die geshrunkene Schätzung eine
  Schätzung — deshalb der Equal-Weight-Fallback.
- Kein Shorting, kein Netto-≠-Brutto-Fall: das Buch ist long-only.
- ``expected_returns`` ist der gefährlichste Input überhaupt (Mean-Variance
  ist ein Error-Maximierer). Deshalb ist mean_variance **nicht** Default und
  verlangt die Schätzung explizit vom Aufrufer.
- Die Ausgabe ist ein **Vorschlag**. Live-Schaltung bleibt menschlich gegated.

Quellen
-------
- Maillard, Roncalli & Teïletche (2010), "The Properties of Equally
  Weighted Risk Contribution Portfolios".
- Spinu (2013), "An Algorithm for Computing Risk Parity Weights";
  Griveau-Billion, Richard & Roncalli (2013), "A Fast Algorithm for
  Computing High-Dimensional Risk Parity Portfolios".
- Michaud (1989), "The Markowitz Optimization Enigma: Is Optimized
  Optimal?" — warum Shrinkage + Constraints kein Luxus sind.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from quantrace.portfolio.risk_model import (
    MIN_OBS,
    CovarianceEstimate,
    ledoit_wolf_shrinkage,
)

#: Unterstützte Sizing-Methoden, in aufsteigender Aggressivität.
SIZING_METHODS = (
    "equal",
    "inverse_vol",
    "risk_parity",
    "min_variance",
    "mean_variance",
)

_TOL = 1e-10
_MAX_ITER = 5_000


@dataclass(frozen=True, slots=True)
class Constraints:
    """Zulässige Menge der Gewichte — long-only, voll investiert.

    Attributes
    ----------
    max_weight:
        Obergrenze je Sleeve. 1.0 = keine Konzentrationsgrenze.
    min_weight:
        Untergrenze je Sleeve (≥ 0 — das Buch ist long-only).
    gross:
        Ziel-Summe der Gewichte vor Vol-Targeting (1.0 = voll investiert).
    max_turnover:
        Deckel für Σ|w − w_prev| beim Rebalancing. ``None`` = kein Deckel.
    target_volatility:
        Annualisierte Ziel-Vol des Portfolios. ``None`` = kein Targeting.
    max_leverage:
        Obergrenze des Brutto-Exposures nach Vol-Targeting. Default 1.0 —
        Vol-Targeting darf nur reduzieren, nie hebeln.
    """

    max_weight: float = 1.0
    min_weight: float = 0.0
    gross: float = 1.0
    max_turnover: float | None = None
    target_volatility: float | None = None
    max_leverage: float = 1.0

    def validate(self, n_assets: int) -> None:
        if n_assets < 1:
            raise ValueError("n_assets muss ≥ 1 sein")
        if self.min_weight < 0.0:
            raise ValueError("min_weight < 0 — das Buch ist long-only")
        if self.max_weight <= 0.0 or self.max_weight > 1.0:
            raise ValueError("max_weight muss in (0, 1] liegen")
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight > max_weight")
        if self.gross <= 0.0:
            raise ValueError("gross muss > 0 sein")
        if self.max_turnover is not None and self.max_turnover < 0.0:
            raise ValueError("max_turnover muss ≥ 0 sein")
        if self.target_volatility is not None and self.target_volatility <= 0.0:
            raise ValueError("target_volatility muss > 0 sein")
        if self.max_leverage <= 0.0:
            raise ValueError("max_leverage muss > 0 sein")
        if n_assets * self.max_weight < self.gross - 1e-12:
            raise ValueError(
                f"infeasible: {n_assets} × max_weight={self.max_weight} < gross={self.gross}"
            )
        if n_assets * self.min_weight > self.gross + 1e-12:
            raise ValueError(
                f"infeasible: {n_assets} × min_weight={self.min_weight} > gross={self.gross}"
            )


@dataclass(frozen=True, slots=True)
class RiskContribution:
    """Wie viel Kapital, wie viel Risiko — pro Sleeve.

    Der springende Punkt: ``weight`` und ``risk_share`` fallen bei
    Equal-Weight regelmäßig weit auseinander. Genau diese Differenz ist
    das Argument für ein Risikomodell.
    """

    name: str
    weight: float
    risk_contribution: float
    risk_share: float
    annualised_volatility: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "weight": self.weight,
            "risk_contribution": self.risk_contribution,
            "risk_share": self.risk_share,
            "annualised_volatility": self.annualised_volatility,
        }


@dataclass(frozen=True, slots=True)
class PortfolioConstruction:
    """Ergebnis von :func:`construct_portfolio` — ein *Vorschlag*, kein Auftrag."""

    weights: dict[str, float]
    method: str
    covariance: CovarianceEstimate
    contributions: list[RiskContribution]
    annualised_volatility: float
    diversification_ratio: float
    gross: float
    cash_weight: float
    leverage_scalar: float
    turnover: float | None
    converged: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def effective_n(self) -> float:
        """Inverse Herfindahl der Gewichte — „wie viele Sleeves wirklich?"."""
        w = np.array(list(self.weights.values()), dtype=float)
        denom = float(np.sum(w**2))
        return float(1.0 / denom) if denom > 1e-15 else 0.0

    @property
    def max_risk_share(self) -> float:
        return max((c.risk_share for c in self.contributions), default=0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "weights": dict(self.weights),
            "method": self.method,
            "contributions": [c.to_dict() for c in self.contributions],
            "annualised_volatility": self.annualised_volatility,
            "diversification_ratio": self.diversification_ratio,
            "gross": self.gross,
            "cash_weight": self.cash_weight,
            "leverage_scalar": self.leverage_scalar,
            "turnover": self.turnover,
            "effective_n": self.effective_n,
            "max_risk_share": self.max_risk_share,
            "converged": self.converged,
            "shrinkage": self.covariance.shrinkage,
            "n_obs": self.covariance.n_obs,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Projektion
# ---------------------------------------------------------------------------


def _project_box_sum(y: np.ndarray, lo: float, hi: float, total: float) -> np.ndarray:
    """Euklidische Projektion auf {w : lo ≤ wᵢ ≤ hi, Σw = total}.

    ``w(θ) = clip(y − θ, lo, hi)`` ist monoton fallend in θ, also findet
    Bisektion das eindeutige θ mit Σw(θ) = total. Das ist die exakte
    Projektion (KKT der Box-restringierten Simplex-Projektion), keine
    Heuristik.
    """
    n = y.size
    if n * hi < total - 1e-12 or n * lo > total + 1e-12:
        raise ValueError("Constraints sind infeasible (Box lässt Σw = total nicht zu)")

    theta_lo = float(np.min(y) - hi - 1.0)
    theta_hi = float(np.max(y) - lo + 1.0)
    for _ in range(200):
        theta = 0.5 * (theta_lo + theta_hi)
        s = float(np.clip(y - theta, lo, hi).sum())
        if abs(s - total) < 1e-14:
            break
        if s > total:
            theta_lo = theta
        else:
            theta_hi = theta
    return np.clip(y - 0.5 * (theta_lo + theta_hi), lo, hi)


# ---------------------------------------------------------------------------
# Sizing-Kerne
# ---------------------------------------------------------------------------


def _equal(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _inverse_vol(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    if np.all(vol < 1e-15):
        return _equal(cov.shape[0])
    vol = np.where(vol < 1e-15, np.median(vol[vol >= 1e-15]), vol)
    inv = 1.0 / vol
    return inv / inv.sum()


def _risk_budget_ccd(
    cov: np.ndarray, budgets: np.ndarray, *, max_iter: int = _MAX_ITER, tol: float = 1e-12
) -> tuple[np.ndarray, bool]:
    """Risk-Budgeting-Gewichte per cyclical coordinate descent.

    Löst min ½·xᵀΣx − Σ bᵢ ln xᵢ koordinatenweise: die Stationaritäts-
    bedingung je i ist die Quadratik σᵢᵢxᵢ² + cᵢxᵢ − bᵢ = 0 mit
    cᵢ = Σ_{j≠i} σᵢⱼxⱼ, deren positive Wurzel geschlossen vorliegt. Die
    Skalierung ist frei (das Log-Barrier-Problem ist bis auf einen Faktor
    homogen), am Ende wird normiert.
    """
    n = cov.shape[0]
    diag = np.clip(np.diag(cov), 1e-18, None)
    x = budgets / np.sqrt(diag)
    x = x / x.sum()
    converged = False
    for _ in range(max_iter):
        x_prev = x.copy()
        for i in range(n):
            c = float(cov[i] @ x - cov[i, i] * x[i])
            sigma_ii = float(diag[i])
            x[i] = (-c + np.sqrt(c * c + 4.0 * sigma_ii * budgets[i])) / (2.0 * sigma_ii)
        if float(np.max(np.abs(x - x_prev))) < tol * max(1.0, float(np.max(np.abs(x_prev)))):
            converged = True
            break
    total = float(x.sum())
    if total <= 0 or not np.isfinite(total):
        return _equal(n), False
    return x / total, converged


def _projected_gradient(
    cov: np.ndarray,
    mu: np.ndarray | None,
    risk_aversion: float,
    constraints: Constraints,
    *,
    max_iter: int = _MAX_ITER,
    tol: float = _TOL,
) -> tuple[np.ndarray, bool]:
    """min ½·λ·wᵀΣw − wᵀμ auf {lo ≤ w ≤ hi, Σw = gross}.

    Projizierter Gradientenabstieg mit Schrittweite 1/L (L = größter
    Eigenwert von λΣ) — für ein konvexes Ziel auf einer konvexen Menge
    konvergiert das monoton. Das Problem ist klein (N ≈ 2…20), deshalb ist
    ein exakter Eigenwert billiger als jede Heuristik.
    """
    n = cov.shape[0]
    scaled = risk_aversion * cov
    eig_max = float(np.max(np.linalg.eigvalsh(scaled)))
    step = 1.0 / eig_max if eig_max > 1e-15 else 1.0

    w = _project_box_sum(
        np.full(n, constraints.gross / n),
        constraints.min_weight,
        constraints.max_weight,
        constraints.gross,
    )
    converged = False
    for _ in range(max_iter):
        grad = scaled @ w - (mu if mu is not None else 0.0)
        nxt = _project_box_sum(
            w - step * grad,
            constraints.min_weight,
            constraints.max_weight,
            constraints.gross,
        )
        if float(np.max(np.abs(nxt - w))) < tol:
            w = nxt
            converged = True
            break
        w = nxt
    return w, converged


# ---------------------------------------------------------------------------
# Risikobeiträge
# ---------------------------------------------------------------------------


def risk_contributions(
    weights: np.ndarray, cov: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """(absolute Beiträge, Anteile, Portfolio-Vol) — Euler-Zerlegung.

    RCᵢ = wᵢ·(Σw)ᵢ / σ_p mit σ_p = √(wᵀΣw); die RCᵢ summieren sich exakt
    auf σ_p (Euler-Theorem für die homogene Funktion σ_p).
    """
    variance = float(weights @ cov @ weights)
    sigma = float(np.sqrt(max(variance, 0.0)))
    if sigma < 1e-15:
        n = weights.size
        return np.zeros(n), np.full(n, 1.0 / n if n else 0.0), 0.0
    marginal = cov @ weights / sigma
    contrib = weights * marginal
    return contrib, contrib / sigma, sigma


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------


def construct_portfolio(
    returns: Mapping[str, Sequence[float]] | np.ndarray | None = None,
    *,
    method: str = "risk_parity",
    names: Sequence[str] | None = None,
    covariance: CovarianceEstimate | None = None,
    constraints: Constraints | None = None,
    risk_budgets: Mapping[str, float] | None = None,
    expected_returns: Mapping[str, float] | None = None,
    risk_aversion: float = 5.0,
    current_weights: Mapping[str, float] | None = None,
    periods_per_year: float = 252.0,
    min_obs: int = MIN_OBS,
) -> PortfolioConstruction:
    """Ziel-Gewichte für ein Buch aus Sleeve-Return-Reihen.

    Parameters
    ----------
    returns:
        ``name → (T,) periodische Returns``, zeitaligned. Optional, wenn
        ``covariance`` direkt übergeben wird.
    method:
        Eine aus :data:`SIZING_METHODS`.
    covariance:
        Vorgerechnetes Risikomodell (überspringt die Schätzung).
    constraints:
        Box + Brutto + optional Turnover-Deckel + Vol-Targeting.
    risk_budgets:
        Nur für ``risk_parity``: ``name → Budget`` (wird normiert). Ohne
        Angabe gleiche Budgets = klassisches ERC.
    expected_returns:
        Nur für ``mean_variance``: ``name → erwarteter *periodischer*
        Return``. Fehlt das, degeneriert die Methode zu min_variance
        (mit Warnung).
    current_weights:
        Aktuelles Buch — Basis für Turnover-Berechnung und -Deckel.
    periods_per_year:
        Annualisierungsfaktor (252 für Tagesdaten).

    Raises
    ------
    ValueError
        Unbekannte Methode, infeasible Constraints, zu wenig
        Beobachtungen, non-finite Inputs.
    """
    if method not in SIZING_METHODS:
        raise ValueError(f"method muss eine aus {SIZING_METHODS} sein, war {method!r}")

    cons = constraints or Constraints()
    warnings: list[str] = []

    if covariance is None:
        if returns is None:
            raise ValueError("returns oder covariance muss gesetzt sein")
        covariance = ledoit_wolf_shrinkage(returns, names=names, min_obs=min_obs)

    asset_names = list(covariance.names)
    cov = np.asarray(covariance.matrix, dtype=float)
    n = len(asset_names)
    cons.validate(n)

    if n == 1:
        warnings.append("Nur ein Sleeve im Buch — jede Sizing-Methode ergibt 100 %.")

    # --- 1) Sizing --------------------------------------------------------
    converged = True
    if method == "equal" or n == 1:
        raw = _equal(n)
    elif method == "inverse_vol":
        raw = _inverse_vol(cov)
    elif method == "risk_parity":
        budgets = _budget_vector(risk_budgets, asset_names, warnings)
        raw, converged = _risk_budget_ccd(cov, budgets)
        if not converged:
            warnings.append(
                "Risk-Parity-CCD hat die Toleranz nicht erreicht — Gewichte sind "
                "die letzte Iteration (in der Regel bereits nahe der Lösung)."
            )
    elif method == "min_variance":
        raw, converged = _projected_gradient(cov, None, 1.0, cons)
    else:  # mean_variance
        mu = _expected_return_vector(expected_returns, asset_names, warnings)
        if risk_aversion <= 0.0:
            raise ValueError("risk_aversion muss > 0 sein")
        raw, converged = _projected_gradient(cov, mu, risk_aversion, cons)
        if not converged:
            warnings.append("Mean-Variance-PGD hat die Toleranz nicht erreicht.")

    # --- 2) Constraints ---------------------------------------------------
    if method in ("equal", "inverse_vol", "risk_parity"):
        scaled = raw * cons.gross
        projected = _project_box_sum(scaled, cons.min_weight, cons.max_weight, cons.gross)
        if float(np.max(np.abs(projected - scaled))) > 1e-9:
            warnings.append(
                f"Constraints binden: {method}-Gewichte auf "
                f"[{cons.min_weight:.2%}, {cons.max_weight:.2%}] projiziert."
            )
        weights = projected
    else:
        weights = raw

    # --- 3) Turnover-Deckel ----------------------------------------------
    prev = _current_vector(current_weights, asset_names)
    turnover: float | None = None
    if prev is not None:
        turnover = float(np.abs(weights - prev).sum())
        if cons.max_turnover is not None and turnover > cons.max_turnover + 1e-12:
            alpha = cons.max_turnover / turnover if turnover > 0 else 1.0
            weights = prev + alpha * (weights - prev)
            new_turnover = float(np.abs(weights - prev).sum())
            warnings.append(
                f"Turnover-Deckel greift: Ziel-Turnover {turnover:.1%} > "
                f"{cons.max_turnover:.1%} — partieller Schritt (α={alpha:.2f}), "
                f"realisiert {new_turnover:.1%}."
            )
            turnover = new_turnover

    # --- 4) Vol-Targeting -------------------------------------------------
    _, _, sigma_period = risk_contributions(weights, cov)
    sigma_annual = sigma_period * float(np.sqrt(periods_per_year))
    leverage = 1.0
    if cons.target_volatility is not None:
        if sigma_annual < 1e-12:
            warnings.append("Portfolio-Vol ≈ 0 — Vol-Targeting übersprungen.")
        else:
            leverage = cons.target_volatility / sigma_annual
            if leverage > cons.max_leverage:
                warnings.append(
                    f"Vol-Targeting will {leverage:.2f}× Exposure für "
                    f"{cons.target_volatility:.1%} Ziel-Vol — auf max_leverage="
                    f"{cons.max_leverage:.2f} gedeckelt (Ziel-Vol nicht erreichbar)."
                )
                leverage = cons.max_leverage
            weights = weights * leverage
            sigma_annual = sigma_annual * leverage

    # --- 5) Report --------------------------------------------------------
    contrib, shares, _ = risk_contributions(weights, cov)
    vols = covariance.annualised_volatilities(periods_per_year)
    contributions = [
        RiskContribution(
            name=name,
            weight=float(w),
            risk_contribution=float(c) * float(np.sqrt(periods_per_year)),
            risk_share=float(s),
            annualised_volatility=float(v),
        )
        for name, w, c, s, v in zip(asset_names, weights, contrib, shares, vols, strict=True)
    ]

    gross = float(weights.sum())
    weighted_vol = float(np.dot(np.abs(weights), covariance.volatilities))
    diversification = (
        weighted_vol / (sigma_period * leverage) if sigma_period * leverage > 1e-15 else 1.0
    )

    if covariance.n_obs < 2 * n:
        warnings.append(
            f"Nur {covariance.n_obs} Beobachtungen für {n} Sleeves — die Kovarianz ist "
            "dünn geschätzt (Shrinkage federt, ersetzt aber keine Historie)."
        )

    return PortfolioConstruction(
        weights={n_: float(w) for n_, w in zip(asset_names, weights, strict=True)},
        method=method,
        covariance=covariance,
        contributions=contributions,
        annualised_volatility=float(sigma_annual),
        diversification_ratio=float(diversification),
        gross=gross,
        cash_weight=float(max(0.0, cons.gross - gross)),
        leverage_scalar=float(leverage),
        turnover=turnover,
        converged=converged,
        warnings=warnings,
    )


def _budget_vector(
    budgets: Mapping[str, float] | None, names: Sequence[str], warnings: list[str]
) -> np.ndarray:
    """Risikobudgets normieren; fehlende/degenerierte Angaben → gleich."""
    n = len(names)
    if not budgets:
        return np.full(n, 1.0 / n)
    raw = np.array([float(budgets.get(name, 0.0)) for name in names], dtype=float)
    if not np.isfinite(raw).all() or np.any(raw < 0) or raw.sum() <= 0:
        warnings.append(
            "risk_budgets unbrauchbar (negativ, nicht-finit oder Summe 0) — "
            "gleiche Budgets (klassisches ERC)."
        )
        return np.full(n, 1.0 / n)
    if np.any(raw <= 0):
        warnings.append(
            "Mindestens ein Risikobudget ist 0 — auf 1 % des Mittels angehoben, "
            "sonst ist das Log-Barrier-Problem unbeschränkt."
        )
        raw = np.where(raw <= 0, 0.01 * raw.mean(), raw)
    return raw / raw.sum()


def _expected_return_vector(
    expected: Mapping[str, float] | None, names: Sequence[str], warnings: list[str]
) -> np.ndarray | None:
    if not expected:
        warnings.append(
            "mean_variance ohne expected_returns — degeneriert zu min_variance. "
            "Erwartete Returns bewusst setzen (Mean-Variance maximiert sonst "
            "Schätzfehler)."
        )
        return None
    missing = [n for n in names if n not in expected]
    if missing:
        warnings.append(
            f"expected_returns fehlen für {missing} — mit 0.0 belegt (konservativ: "
            "keine Meinung = kein Übergewicht)."
        )
    return np.array([float(expected.get(n, 0.0)) for n in names], dtype=float)


def _current_vector(
    current: Mapping[str, float] | None, names: Sequence[str]
) -> np.ndarray | None:
    if current is None:
        return None
    return np.array([float(current.get(n, 0.0)) for n in names], dtype=float)


__all__ = [
    "SIZING_METHODS",
    "Constraints",
    "PortfolioConstruction",
    "RiskContribution",
    "construct_portfolio",
    "risk_contributions",
]
