"""Portfolio-Konstruktion — vom einzelnen Kandidaten zum Buch.

Bis hierher bewertet QuantRace Strategien *isoliert* (Sharpe, DSR, PBO,
Bootstrap, Regime) plus eine Uniqueness-Lens. Die Allokation selbst war
Equal-Weight über Sleeves. Dieses Paket schließt die Lücke: aus den
persistierten Return-Pfaden des Approved-Buchs werden ein Risikomodell
(:mod:`quantrace.portfolio.risk_model`) und daraus Ziel-Gewichte
(:mod:`quantrace.portfolio.construction`) gerechnet.

Alles hier ist **pure** — numpy, kein I/O, kein Netz, keine Optimizer-
Abhängigkeit (kein cvxpy/PyPortfolioOpt). Gleicher Stil wie
``quantrace.stats``: from scratch, gegen Closed-Form-Fälle getestet.

Governance-Invariante bleibt: die Gewichte sind ein **Vorschlag**. Nichts
hier schaltet etwas live.
"""

from quantrace.portfolio.construction import (
    SIZING_METHODS,
    Constraints,
    PortfolioConstruction,
    RiskContribution,
    construct_portfolio,
    risk_contributions,
)
from quantrace.portfolio.risk_model import (
    CovarianceEstimate,
    correlation_from_covariance,
    ledoit_wolf_shrinkage,
    sample_covariance,
)

__all__ = [
    "SIZING_METHODS",
    "Constraints",
    "CovarianceEstimate",
    "PortfolioConstruction",
    "RiskContribution",
    "construct_portfolio",
    "correlation_from_covariance",
    "ledoit_wolf_shrinkage",
    "risk_contributions",
    "sample_covariance",
]
