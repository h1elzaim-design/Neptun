"""Backtest-Runner auf vectorbt. Konsistente Kosten- und Slippage-Annahmen.

Der Runner kennt keine Strategielogik — er bekommt MarketData + Strategy +
BacktestConfig und liefert ein BacktestResult. Dadurch sind alle Strategien
unter identischen Bedingungen vergleichbar.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quantrace.data_agent import close_prices
from quantrace.models import (
    BacktestConfig,
    BacktestResult,
    MarketData,
    StrategySpec,
    TradeMetrics,
)
from quantrace.strategy import Strategy, load_strategy

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


def run_backtest(
    spec: StrategySpec,
    data: MarketData,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Führt den Backtest aus und gibt das vollständige Ergebnis zurück."""
    config = config or BacktestConfig()
    strategy = load_strategy(spec)
    return _execute(strategy, spec.strategy_id, data, config)


def run_inline(
    strategy_id: str,
    strategy: Strategy,
    data: MarketData,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Variante ohne StrategySpec — bequem für Notebook-Forschung."""
    return _execute(strategy, strategy_id, data, config or BacktestConfig())


def _execute(
    strategy: Strategy,
    strategy_id: str,
    data: MarketData,
    config: BacktestConfig,
) -> BacktestResult:
    try:
        import vectorbt as vbt
    except ImportError as e:
        raise ImportError("vectorbt nicht installiert. `pip install vectorbt`.") from e

    close = close_prices(data)
    entries, exits = strategy.generate_signals(data)

    fees = config.fees_bps / 10_000.0
    slippage = config.slippage_bps / 10_000.0

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=config.cash,
        fees=fees,
        slippage=slippage,
        size=config.size,
        size_type=config.size_type,
        freq=config.freq,
        cash_sharing=True,
        call_seq="auto",
    )

    equity = _aggregate_equity(pf, config.cash)
    metrics = _compute_metrics(equity, config.annualization)
    trades = _trade_metrics(pf)

    return BacktestResult(
        strategy_id=strategy_id,
        data_hash=data.content_hash,
        config=config,
        start=_to_date(close.index[0]),
        end=_to_date(close.index[-1]),
        total_return=metrics["total_return"],
        cagr=metrics["cagr"],
        sharpe=metrics["sharpe"],
        sortino=metrics["sortino"],
        calmar=metrics["calmar"],
        max_drawdown=metrics["max_drawdown"],
        avg_drawdown=metrics["avg_drawdown"],
        ulcer_index=metrics["ulcer_index"],
        trades=trades,
        equity_curve=equity,
    )


def _aggregate_equity(pf, init_cash: float) -> pd.Series:
    val = pf.value()
    if isinstance(val, pd.DataFrame):
        # Cash sharing: pro Symbol gleicher Cash-Pool, summieren wäre falsch;
        # vectorbt gibt bei cash_sharing=True bereits den geteilten Wert pro Spalte zurück.
        # Wir nehmen den Mittelwert als robuste Aggregation für Multi-Asset.
        val = val.mean(axis=1)
    return val.astype(float)


def _compute_metrics(equity: pd.Series, ann: int) -> dict[str, float]:
    ret = equity.pct_change().dropna()
    if ret.empty or equity.iloc[0] == 0:
        return dict.fromkeys(
            [
                "total_return",
                "cagr",
                "sharpe",
                "sortino",
                "calmar",
                "max_drawdown",
                "avg_drawdown",
                "ulcer_index",
            ],
            0.0,
        )

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)

    std = ret.std(ddof=0)
    sharpe = float(np.sqrt(ann) * ret.mean() / std) if std > 0 else 0.0

    downside = ret[ret < 0].std(ddof=0)
    sortino = float(np.sqrt(ann) * ret.mean() / downside) if downside > 0 else 0.0

    running_max = equity.cummax()
    dd = equity / running_max - 1
    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    ulcer = float(np.sqrt((dd**2).mean()))
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd,
        "ulcer_index": ulcer,
    }


def _trade_metrics(pf) -> TradeMetrics:
    try:
        trades = pf.trades.records_readable
    except Exception:  # pragma: no cover — vectorbt versionsabhängig
        trades = pd.DataFrame()

    if trades.empty or "Return" not in trades.columns:
        return TradeMetrics(
            n_trades=0,
            win_rate=0.0,
            avg_trade_return=0.0,
            avg_winner=0.0,
            avg_loser=0.0,
            profit_factor=0.0,
            expectancy=0.0,
        )

    rets = trades["Return"].astype(float)
    winners = rets[rets > 0]
    losers = rets[rets < 0]
    gross_win = winners.sum()
    gross_loss = -losers.sum()
    profit_factor = (
        float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
    )

    return TradeMetrics(
        n_trades=int(len(rets)),
        win_rate=float(len(winners) / len(rets)),
        avg_trade_return=float(rets.mean()),
        avg_winner=float(winners.mean()) if not winners.empty else 0.0,
        avg_loser=float(losers.mean()) if not losers.empty else 0.0,
        profit_factor=profit_factor,
        expectancy=float(rets.mean()),
    )


def _to_date(ts) -> date:
    return pd.Timestamp(ts).date()
