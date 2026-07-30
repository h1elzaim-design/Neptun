"""Statistical primitives for backtest validation.

Strictly pure functions. No I/O, no Pydantic, no FastAPI. Each module is
independently importable and unit-tested.

Submodules
- sharpe:        annualised / probabilistic / deflated Sharpe variants
- fdr:           Benjamini–Hochberg FDR control + Sharpe p-values
- validation:    purged & embargoed K-Fold cross-validation
- pbo:           Probability of Backtest Overfitting (CSCV)
- bootstrap:     stationary-bootstrap Sharpe CIs + drawdown distributions
- attribution:   OLS factor attribution with Newey–West HAC errors
- uniqueness:    candidate-vs-book correlations + marginal sleeve Sharpe
- capacity:      turnover, break-even costs, square-root-law capacity
- cost_stress:   apply slippage / fee multipliers to a return series
- survivorship:  universe audit + risk classification

Sources
- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting and Non-Normality"
- Bailey, Borwein, López de Prado & Zhu (2017), "The Probability of
  Backtest Overfitting"
- Benjamini & Hochberg (1995), "Controlling the False Discovery Rate"
- Politis & Romano (1994), "The Stationary Bootstrap"
- Newey & West (1987), HAC covariance estimation
- Almgren, Thum, Hauptmann & Li (2005), "Direct Estimation of Equity
  Market Impact" — square-root impact law behind the capacity estimate
- López de Prado (2018), "Advances in Financial Machine Learning"
  §7.4 (Purged & Embargoed Cross-Validation), §14 (Backtest Statistics)
"""

from quantrace.stats.attribution import (
    FactorAttributionResult,
    FactorExposure,
    RollingExposurePoint,
    factor_attribution,
    rolling_factor_attribution,
)
from quantrace.stats.bootstrap import (
    BootstrapDrawdownResult,
    BootstrapSharpeResult,
    bootstrap_drawdown_distribution,
    bootstrap_sharpe_ci,
    stationary_bootstrap_indices,
)
from quantrace.stats.capacity import (
    CapacityEstimate,
    CostSensitivityPoint,
    CostSensitivityResult,
    SymbolCapacity,
    TurnoverProfile,
    capacity_estimate,
    cost_sensitivity,
    estimate_turnover_from_trades,
    turnover_from_orders,
)
from quantrace.stats.cost_stress import CostStressResult, apply_cost_stress
from quantrace.stats.fdr import (
    DEFAULT_FDR_ALPHA,
    FdrResult,
    benjamini_hochberg,
    sharpe_p_value,
)
from quantrace.stats.pbo import PboResult, probability_of_backtest_overfitting
from quantrace.stats.sharpe import (
    DeflatedSharpeResult,
    ProbabilisticSharpeResult,
    annualised_sharpe,
    deflated_sharpe_from_summary,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_from_summary,
    probabilistic_sharpe_ratio,
    sample_skew_kurt,
)
from quantrace.stats.survivorship import SurvivorshipAudit, audit_universe
from quantrace.stats.uniqueness import (
    CandidateCorrelation,
    UniquenessResult,
    uniqueness,
)
from quantrace.stats.validation import PurgedFold, purged_kfold

__all__ = [
    "turnover_from_orders",
    "estimate_turnover_from_trades",
    "cost_sensitivity",
    "capacity_estimate",
    "TurnoverProfile",
    "SymbolCapacity",
    "CostSensitivityResult",
    "CostSensitivityPoint",
    "CapacityEstimate",
    "DEFAULT_FDR_ALPHA",
    "BootstrapDrawdownResult",
    "CandidateCorrelation",
    "BootstrapSharpeResult",
    "CostStressResult",
    "DeflatedSharpeResult",
    "FactorAttributionResult",
    "FactorExposure",
    "FdrResult",
    "PboResult",
    "ProbabilisticSharpeResult",
    "PurgedFold",
    "RollingExposurePoint",
    "SurvivorshipAudit",
    "UniquenessResult",
    "annualised_sharpe",
    "apply_cost_stress",
    "audit_universe",
    "benjamini_hochberg",
    "bootstrap_drawdown_distribution",
    "bootstrap_sharpe_ci",
    "deflated_sharpe_from_summary",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "factor_attribution",
    "probabilistic_sharpe_from_summary",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "purged_kfold",
    "rolling_factor_attribution",
    "sample_skew_kurt",
    "sharpe_p_value",
    "stationary_bootstrap_indices",
    "uniqueness",
]
