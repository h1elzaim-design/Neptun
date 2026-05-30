"""QuantRace — KI-gestützte Trading-Research-Plattform."""

__version__ = "0.1.0"

from quantrace.models import (
    BacktestConfig,
    BacktestResult,
    EvaluationReport,
    FoldResult,
    KnowledgeNote,
    MarketData,
    StrategySpec,
    WalkForwardResult,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "EvaluationReport",
    "FoldResult",
    "KnowledgeNote",
    "MarketData",
    "StrategySpec",
    "WalkForwardResult",
]
