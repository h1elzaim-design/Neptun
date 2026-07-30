"""Market-regime detection — Hidden Markov Models on price/volatility features.

This package turns a price series into a *causal* regime label (risk-on /
neutral / risk-off) using a Gaussian Hidden Markov Model fit with the
Baum-Welch EM algorithm. It is implemented from scratch in NumPy (log-space,
numerically stable) so the API Docker image stays lean — no scikit-learn /
hmmlearn dependency.

Public surface:
    GaussianHMM        Low-level fit/decode/score (states are anonymous indices).
    regime_features    Price → trailing trend + realised-vol feature frame.
    RegimeDetector     High-level: fit on prices, get semantic regime labels.
"""

from __future__ import annotations

from quantrace.regime.backtesting import regime_conditioned_metrics
from quantrace.regime.detector import RegimeDetector, RegimeSnapshot
from quantrace.regime.diagnostics import (
    RegimeDiagnostics,
    adf_test,
    refit_stability,
    regime_diagnostics,
)
from quantrace.regime.features import regime_features
from quantrace.regime.hmm import GaussianHMM

__all__ = [
    "GaussianHMM",
    "RegimeDetector",
    "RegimeDiagnostics",
    "RegimeSnapshot",
    "adf_test",
    "refit_stability",
    "regime_conditioned_metrics",
    "regime_diagnostics",
    "regime_features",
]
