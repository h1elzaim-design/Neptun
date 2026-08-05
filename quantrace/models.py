"""Standardisierte Datenobjekte für den gesamten Research-Workflow.

Alle Agenten (Data, Research, Backtest, Evaluation, Knowledge) sprechen
ausschließlich über diese Modelle miteinander. Wenn ein Feld fehlt, ist das
ein Architekturentscheid, kein Implementierungsdetail.

Pandas-Import-Hinweis
---------------------
`pandas` wird hier NICHT auf Modul-Ebene importiert. Die API lädt models.py
beim Router-Import (FastAPI); ein Top-Level `import pandas` würde ~150 MB RSS
sofort allozieren und Render Free (512 MB) beim Start OOM-killen.

pd.DataFrame bleibt als Laufzeit-Typ vollständig unterstützt — Code der
eklatant mit MarketData.frame arbeitet, hat pandas sowieso installiert und
importiert es selbst. Hier nur TYPE_CHECKING-Guard für Typ-Annotationen.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantrace.calendars import DEFAULT_CALENDAR, get_calendar
from quantrace.calendars import periods_per_year as _ppy

if TYPE_CHECKING:
    pass


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
    #: Handelskalender des Universums (#184). Aus `data/universes/*.yaml`.
    calendar: str = Field(
        default=DEFAULT_CALENDAR,
        description="us_equity | crypto_24_7 — bestimmt die Annualisierung.",
    )
    frame: Any = Field(..., exclude=True)  # pd.DataFrame — lazy import, see module docstring
    content_hash: str = ""

    @field_validator("calendar")
    @classmethod
    def _calendar_must_be_known(cls, v: str) -> str:
        """Tippfehler früh abfangen. Ein unbekannter Kalender darf nicht still
        auf den Default zurückfallen — dann wäre die Annualisierung falsch,
        ohne dass es jemand merkt."""
        return get_calendar(v).name

    @property
    def periods_per_year(self) -> float:
        """Perioden pro Jahr — **abgeleitet**, kein eigenes Feld.

        Als Feld könnte es vom `calendar` abweichen, und genau diese Art von
        stiller Divergenz ist der Fehler, den #184 beseitigt. Es gibt eine
        Wahrheit: den Kalender.
        """
        return _ppy(self.calendar)

    @field_validator("frame")
    @classmethod
    def _frame_must_have_ohlcv(cls, v: Any) -> Any:
        required = {"open", "high", "low", "close", "volume"}
        cols_lower = {
            c.lower() if isinstance(c, str) else c for c in v.columns.get_level_values(-1)
        }
        missing = required - cols_lower
        if missing:
            raise ValueError(f"MarketData.frame fehlt OHLCV-Spalten: {missing}")
        # DatetimeIndex check — import pandas lazily only when validator actually runs
        import pandas as _pd
        if not isinstance(v.index, _pd.DatetimeIndex):
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


class SymbolCosts(BaseModel):
    """Aufgelöste Kosten eines Symbols (pro Order-Seite, in Basispunkten).

    Effektive Slippage pro Seite = ``slippage_bps + spread_bps / 2`` (Market-
    Order kreuzt den halben Spread; slippage_bps ist der Impact obendrauf).
    """

    asset_class: str
    fees_bps: float
    slippage_bps: float
    spread_bps: float

    @property
    def effective_slippage_bps(self) -> float:
        return self.slippage_bps + self.spread_bps / 2.0

    @property
    def total_per_side_bps(self) -> float:
        return self.fees_bps + self.effective_slippage_bps


# Einzige Quelle der erlaubten Kostenmodelle — CLI und Pipeline validieren
# gegen diese Konstante statt eigener String-Listen.
COST_MODELS: tuple[str, ...] = ("flat", "per_asset_class")

# Einzige Quelle der erlaubten Kapitalmodelle (gleiche Konvention wie COST_MODELS).
CAPITAL_MODELS: tuple[str, ...] = ("shared", "independent")


class BacktestConfig(BaseModel):
    """Reproduzierbare Backtest-Annahmen. Wer hier rumdreht, dreht am Ergebnis."""

    cash: float = 100_000.0
    # Pro Order-Seite: vectorbt belastet `fees` bei Entry UND Exit — 2.0 hier
    # heißt 4 bps round-trip. Gleiche Konvention wie SymbolCosts.
    fees_bps: float = Field(2.0, description="Fee in basis points per order side")
    slippage_bps: float = 5.0
    # "flat": fees_bps/slippage_bps für alle Symbole (bisheriges Verhalten).
    # "per_asset_class": der Runner löst pro Symbol Fees/Slippage/Spread aus
    # config/costs.yaml auf (Klassifikation + Profile) und rechnet mit
    # per-Spalte-Kosten; die aufgelöste Tabelle landet in `symbol_costs`,
    # damit das persistierte Ergebnis seine Kosten-Annahmen dokumentiert.
    cost_model: Literal["flat", "per_asset_class"] = "flat"
    # Vorbelegt → gewinnt gegen config/costs.yaml (explizites Override, z.B.
    # für Stress-Szenarien); sonst füllt der Runner das Feld beim Auflösen.
    symbol_costs: dict[str, SymbolCosts] | None = None
    # "shared" (Default): EIN Konto für alle Symbole. Das Kapital konkurriert um
    # einen Pool und wird gleichgewichtet über die gerade aktiven Positionen
    # verteilt (k aktive → je size/k des Kontowerts; k=0 → Cash). Umgeschichtet
    # wird nur, wenn sich die Signal-Mitgliedschaft ändert — dazwischen driften
    # die Gewichte mit dem Markt. `size_type` ist in diesem Modell ohne Wirkung.
    # "independent": Rollback auf die alte Semantik — jede Spalte backtestet mit
    # dem vollen `cash` als eigenes Konto, Equity = Mittelwert der Sleeves
    # (kein geteiltes Kapital, keine Kapazitäts-Restriktion). Siehe ADR-003.
    capital_model: Literal["shared", "independent"] = "shared"
    # Nur im independent-Modell (und für Einzel-Symbol-Läufe) wirksam.
    size_type: Literal["percent", "value", "shares"] = "percent"
    # In beiden Modellen die investierte Obergrenze: shared hält 1−size als
    # Cash-Puffer (Fees/Slippage), independent nutzt size pro Sleeve-Order.
    size: float = 0.95
    allow_shorts: bool = False
    freq: str = "1D"
    annualization: int = 252
    execution_lag: int = Field(
        1,
        ge=0,
        description=(
            "Bars between signal and fill. 1 (default) means a signal derived "
            "from bar t's close executes at t+1, eliminating same-bar "
            "look-ahead. Set 0 only for signals already lagged upstream."
        ),
    )


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
    #: Womit annualisiert wurde (#184). Steht oben statt nur in `config`, weil
    #: die Analytics-Schicht der Webapp es shape-unabhängig lesen muss — Sweep
    #: und Walk-Forward haben andere Ergebnis-Formen, aber denselben Schlüssel.
    periods_per_year: float = 252.0

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

    # Annualisierter Turnover = Σ|gehandeltes Notional| / mittlerer Kontowert
    # pro Jahr (one-way, gleiche Konvention wie quantrace.paper.rebalance).
    # Der Haupttreiber der realisierten Kosten und der Nenner jeder
    # Kapazitätsrechnung. None auf Ergebnissen von vor der Persistierung —
    # die Capacity-Analytik fällt dann auf eine Schätzung aus der Trade-Zahl
    # zurück und weist sie als solche aus.
    turnover_annual: float | None = None

    # Per-regime performance breakdown (written to JSON; computed at backtest time).
    regime_metrics: dict[str, Any] | None = None

    # Rohartefakte (optional, nicht serialisiert in JSON)
    equity_curve: Any | None = Field(default=None, exclude=True)  # pd.Series — lazy
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
    # Annualisierter Turnover des OOS-Fensters (one-way). None auf alten
    # Ergebnissen und wenn die Order-Records nicht auswertbar waren.
    test_turnover_annual: float | None = None
    # Multiple-testing-Statistik des OOS-Fensters (H0: true SR ≤ 0, Mertens-SE
    # mit den echten Return-Momenten des Test-Fensters; q-Wert via
    # Benjamini-Hochberg über alle Folds). None auf alten Ergebnissen.
    test_n_obs: int | None = None
    test_p_value: float | None = None
    test_q_value: float | None = None
    fdr_significant: bool | None = None


class WalkForwardResult(BaseModel):
    """Aggregiertes Ergebnis einer Walk-Forward-Validation über mehrere Folds."""

    strategy_id: str
    #: Siehe BacktestResult.periods_per_year (#184).
    periods_per_year: float = 252.0
    n_folds: int
    rank_by: str = "sharpe"
    folds: list[FoldResult] = Field(default_factory=list)

    #: Die Annahmen, unter denen ALLE Folds gerechnet wurden — allen voran die
    #: Kosten. Sweeps persistieren sie längst (``runs[].result.config``); hier
    #: fehlten sie, und weil der Evaluation-Agent fehlende Kosten als *null*
    #: las, bekam jeder Walk-Forward ``realism 0.00``. Der disziplinierte Pfad
    #: wurde damit gegenüber einem Grid systematisch schlechter bewertet, aus
    #: einem Grund, der nichts mit der Strategie zu tun hat.
    #: ``None`` auf Alt-Ergebnissen — dort ist die Annahme unbekannt, nicht null.
    config: BacktestConfig | None = None

    # Aggregierte Metriken
    is_sharpe_mean: float = Field(0.0, description="Durchschnitt Sharpe In-Sample")
    oos_sharpe_mean: float = Field(0.0, description="Durchschnitt Sharpe Out-of-Sample")
    degradation: float = Field(
        0.0,
        description="OOS/IS-Ratio. 1.0 = perfekt, <0.5 = Overfitting-Warnung",
    )

    # Benjamini-Hochberg-Zusammenfassung über die OOS-Folds (method, alpha,
    # n_tests, n_significant, all_significant). None auf alten Ergebnissen.
    fdr: dict[str, Any] | None = None

    # Inferenz über den *gestitchten* OOS-Pfad (alle Test-Fenster chronologisch
    # konkateniert): annualisierter Sharpe + Stationary-Bootstrap-KI/p-Wert.
    # Die Fold-Mittelung oben gewichtet jeden Fold gleich; der gestitchte Pfad
    # ist die Rendite-Reihe, die ein Live-Deployment tatsächlich erlebt hätte.
    # None auf alten Ergebnissen oder wenn die Test-Fenster zu kurz sind.
    oos_inference: dict[str, Any] | None = None

    # Der gestitchte OOS-Equity-Pfad selbst ([{date, value}], kettennormiert
    # über die Fold-Grenzen). Persistiert, damit Downstream-Analysen (Portfolio-
    # Uniqueness, Factor-Attribution, Bootstrap) auf dem Pfad rechnen können,
    # den ein Deployment erlebt hätte — WF-Ergebnisse hatten bisher gar keinen
    # Return-Pfad im JSON. None auf alten Ergebnissen.
    oos_equity: list[dict[str, Any]] | None = None

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
        "11 Live Monitoring",
        "12 News",
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
