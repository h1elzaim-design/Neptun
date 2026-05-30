"""Statistical primitives for backtest validation.

Strictly pure functions. No I/O, no Pydantic, no FastAPI. Each module is
independently importable and unit-tested.

Submodules
- sharpe:        annualised / probabilistic / deflated Sharpe variants
- validation:    purged & embargoed K-Fold cross-validation
- cost_stress:   apply slippage / fee multipliers to a return series
- survivorship:  universe audit + risk classification

Sources
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting and Non-Normality"
- López de Prado (2018), "Advances in Financial Machine Learning"
  §7.4 (Purged & Embargoed Cross-Validation), §14 (Backtest Statistics)
"""

from quantrace.stats.cost_stress import CostStressResult, apply_cost_stress
from quantrace.stats.sharpe import (
    DeflatedSharpeResult,
    ProbabilisticSharpeResult,
    annualised_sharpe,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from quantrace.stats.survivorship import SurvivorshipAudit, audit_universe
from quantrace.stats.validation import PurgedFold, purged_kfold

__all__ = [
    "CostStressResult",
    "DeflatedSharpeResult",
    "ProbabilisticSharpeResult",
    "PurgedFold",
    "SurvivorshipAudit",
    "annualised_sharpe",
    "apply_cost_stress",
    "audit_universe",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "purged_kfold",
]
