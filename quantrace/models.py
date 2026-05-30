"""Standardisierte Datenobjekte für den gesamten Research-Workflow.

Alle Agenten (Data, Research, Backtest, Evaluation, Knowledge) sprechen
ausschließlich über diese Modelle miteinander. Wenn ein Feld fehlt, ist das
ein Architekturentscheid, kein Implementierungsdetail.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Timeframe(str, Enum):
    DAILY = "1d"
    HOURLY = "1h"
    MINUTE = "1m"


class StrategyStatus(str, Enum):
    DRAFT = "draft"
    TEST = "test"
    REVIEW = "review"
    APPROVED = "approved"
    REJECTED = "rejected"


class MarketData(BaseModel):
    """OHLCV-Bündel für ein Universum, normalisiert und versioniert.

    Die Daten selbst werden NICHT serialisiert — nur die Metadaten und ein
    Hash über den Inhalt. Die DataFrames bleiben on disk (Parquet) oder
    in-memory.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    universe: str = Field(..., description="Logischer Universumsname, z.B. 'sp500_etfs'")
    symbols: list[str]
    timeframe: Timeframe
    start: date
    end: date
    provider: str = Field(..., description="OpenBB-Provider, z.B. 'yfinance', 'fmp'")
    adjusted: bool = True
    frame: pd.DataFrame = Field(..., exclude=True)
    content_hash: str = ""

    @field_validator("frame")
    @classmethod
    def _frame_must_have_ohlcv(cls, v: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        cols_lower = {
            c.lower() if isinstance(c, str) else c for c in v.columns.get_level_values(-1)
        }
        missing = required - cols_lower
        if missing:
            raise ValueError(f"MarketData.frame fehlt OHLCV-Spalten: {missing}")
        if not isinstance(v.index, pd.DatetimeIndex):
            raise ValueError("MarketData.frame muss DatetimeIndex haben")
        return v.sort_index()

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash:
            digest = hashlib.sha256()
            digest.update(str(self.frame.shape).encode())
            digest.update(str(self.frame.index[0]).encode())
            digest.update(str(self.frame.index[-1]).encode())
            digest.update(str(self.frame.iloc[-1].to_dict()).encode())
            object.__setattr__(self, "content_hash", digest.hexdigest()[:16])


class StrategySpec(BaseModel):
    """Deklarative Strategie-Beschreibung — kein Code, sondern Vertrag."""

    strategy_id: str = Field(..., pattern=r"^[a-z0-9_.\-]+$")
    name: str
    class_path: str = Field(
        ..., description="Dotted path: quantrace.strategies.sma_crossover:SmaCrossover"
    )
    strategy_class: Literal[
        "trend_following", "mean_reversion", "momentum", "cross_sectional", "volatility", "custom"
    ]
    universe: str
    timeframe: Timeframe
    params: dict[str, Any] = Field(default_factory=dict)
    param_space: dict[str, list[Any]] = Field(default_factory=dict, description="Für Sweeps")
    description: str = ""
    risks: list[str] = Field(default_factory=list)
    status: StrategyStatus = StrategyStatus.DRAFT


class BacktestConfig(BaseModel):
    """Reproduzierbare Backtest-Annahmen. Wer hier rumdreht, dreht am Ergebnis."""

    cash: float = 100_000.0
    fees_bps: float = Field(2.0, description="Round-trip fee in basis points")
    slippage_bps: float = 5.0
    size_type: Literal["percent", "value", "shares"] = "percent"
    size: float = 0.95
    allow_shorts: bool = False
    freq: str = "1D"
    annualization: int = 252


class TradeMetrics(BaseModel):
    n_trades: int
    win_rate: float
    avg_trade_return: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    expectancy: float


class BacktestResult(BaseModel):
    """Maschinenlesbares Backtest-Ergebnis. Alles, was die Evaluation braucht."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy_id: str
    data_hash: str
    config: BacktestConfig
    start: date
    end: date

    # Performance
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float

    # Risiko
    max_drawdown: float
    avg_drawdown: float
    ulcer_index: float

    # Trades
    trades: TradeMetrics

    # Rohartefakte (optional, nicht serialisiert in JSON)
    equity_curve: pd.Series | None = Field(default=None, exclude=True)
    artifacts_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvaluationReport(BaseModel):
    """Bewertung nach Performance / Risiko / Stabilität / Realismus / Generalisierung / Einfachheit."""

    strategy_id: str
    backtest_ids: list[str]

    # Score-Komponenten (0..1, höher = besser)
    score_performance: float
    score_risk: float
    score_stability: float
    score_realism: float
    score_generalization: float
    score_simplicity: float
    score_total: float

    passed_guardrails: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    next_variation_hint: str | None = None
    notes: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FoldResult(BaseModel):
    """Ergebnis eines Walk-Forward-Folds: Train-Sweep → beste Params → Test-Backtest."""

    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    chosen_params: dict[str, Any]
    train_sharpe: float = Field(description="Sharpe auf Train (In-Sample)")
    train_cagr: float = 0.0
    test_sharpe: float = Field(description="Sharpe auf Test (Out-of-Sample)")
    test_cagr: float = 0.0
    test_max_drawdown: float = 0.0
    test_n_trades: int = 0


class WalkForwardResult(BaseModel):
    """Aggregiertes Ergebnis einer Walk-Forward-Validation über mehrere Folds."""

    strategy_id: str
    n_folds: int
    rank_by: str = "sharpe"
    folds: list[FoldResult] = Field(default_factory=list)

    # Aggregierte Metriken
    is_sharpe_mean: float = Field(0.0, description="Durchschnitt Sharpe In-Sample")
    oos_sharpe_mean: float = Field(0.0, description="Durchschnitt Sharpe Out-of-Sample")
    degradation: float = Field(
        0.0,
        description="OOS/IS-Ratio. 1.0 = perfekt, <0.5 = Overfitting-Warnung",
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class KnowledgeNote(BaseModel):
    """Ein Obsidian-Note-Vertrag. Frontmatter + Body, kein freies Markdown-Chaos."""

    folder: Literal[
        "00 Dashboard",
        "01 Hypothesen",
        "02 Strategien",
        "03 Backtests",
        "04 Evaluations",
        "05 Approved Candidates",
        "06 Rejected Ideas",
        "07 Regime Notes",
        "08 Decision Memos",
    ]
    title: str
    frontmatter: dict[str, Any]
    body: str
    tags: list[str] = Field(default_factory=list)

    @property
    def path(self) -> str:
        safe = self.title.replace("/", "-").strip()
        return f"Trading Research/{self.folder}/{safe}.md"

    def to_markdown(self) -> str:
        import yaml

        fm = dict(self.frontmatter)
        if self.tags:
            fm["tags"] = self.tags
        front = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip()
        return f"---\n{front}\n---\n\n{self.body.strip()}\n"


def dump_json(model: BaseModel) -> str:
    """Konsistente JSON-Serialisierung für Persistenz."""
    return json.dumps(model.model_dump(mode="json"), indent=2, default=str)
