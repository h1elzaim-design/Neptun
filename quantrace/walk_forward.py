"""Walk-Forward-Validation — robuster Test gegen Overfitting.

Nutzt `split_walk_forward` aus evaluation.py um die Datenreihen zu schneiden.
Pro Fold wird ein In-Sample-Sweep durchgeführt, die besten Parameter werden
gewählt und Out-of-Sample getestet.
"""

from __future__ import annotations

import logging

import pandas as pd

from quantrace.backtest_runner import run_backtest
from quantrace.evaluation import split_walk_forward
from quantrace.models import (
    BacktestConfig,
    FoldResult,
    MarketData,
    StrategySpec,
    WalkForwardResult,
)
from quantrace.sweep import sweep

log = logging.getLogger(__name__)


def _slice_market_data(md: MarketData, start: pd.Timestamp, end: pd.Timestamp) -> MarketData:
    """Schneidet das MarketData-Objekt auf einen Zeitraum zu."""
    sliced_frame = md.frame.loc[start:end]
    if sliced_frame.empty:
        raise ValueError(f"Slice {start.date()}..{end.date()} ist leer.")

    # Update properties based on the actual slice
    actual_start = sliced_frame.index[0].date()
    actual_end = sliced_frame.index[-1].date()

    return MarketData(
        universe=md.universe,
        symbols=md.symbols,
        timeframe=md.timeframe,
        provider=md.provider,
        start=actual_start,
        end=actual_end,
        adjusted=md.adjusted,
        frame=sliced_frame,
    )


def walk_forward(
    spec: StrategySpec,
    data: MarketData,
    config: BacktestConfig | None = None,
    n_folds: int = 4,
    train_ratio: float = 0.6,
    rank_by: str = "sharpe",
) -> WalkForwardResult:
    """Führt eine Walk-Forward-Validation über die übergebenen Daten aus.

    Args:
        spec: StrategySpec mit param_space für In-Sample-Sweep.
        data: Gesamtdatensatz.
        config: Backtest-Config.
        n_folds: Anzahl der Walk-Forward-Epochen.
        train_ratio: Anteil der In-Sample-Daten pro Fold.
        rank_by: Kriterium zur Auswahl der besten In-Sample-Parameter.

    Returns:
        WalkForwardResult mit aggregierten In-Sample- und Out-of-Sample-Metriken.
    """
    config = config or BacktestConfig()
    splits = split_walk_forward(data.frame.index, n_folds=n_folds, train_ratio=train_ratio)

    folds: list[FoldResult] = []

    for i, (train_start, train_end, test_end) in enumerate(splits, 1):
        log.info(
            "Fold %d/%d: Train %s..%s, Test %s..%s",
            i,
            n_folds,
            train_start.date(),
            train_end.date(),
            train_end.date(),
            test_end.date(),
        )

        # 1. Daten schneiden
        try:
            train_data = _slice_market_data(data, train_start, train_end)
            test_data = _slice_market_data(data, train_end, test_end)
        except ValueError as e:
            log.warning("Fold %d übersprungen: %s", i, e)
            continue

        # 2. In-Sample Sweep
        log.info("  IS Sweep Fold %d...", i)
        sweep_res = sweep(spec, train_data, config=config, rank_by=rank_by)

        if not sweep_res.best_run:
            log.warning("Fold %d: Sweep hat keine Ergebnisse geliefert.", i)
            continue

        chosen_params = sweep_res.best_params
        train_result = sweep_res.best_run.result

        # 3. Out-of-Sample Backtest mit gewählten Parametern
        log.info("  OOS Test Fold %d mit %s", i, chosen_params)
        oos_spec = spec.model_copy(update={"params": chosen_params})
        test_result = run_backtest(oos_spec, test_data, config=config)

        # 4. Resultat festhalten
        folds.append(
            FoldResult(
                fold_index=i,
                train_start=train_start.date(),
                train_end=train_end.date(),
                test_start=train_data.end,  # Use actual data end
                test_end=test_data.end,
                chosen_params=chosen_params,
                train_sharpe=train_result.sharpe,
                train_cagr=train_result.cagr,
                test_sharpe=test_result.sharpe,
                test_cagr=test_result.cagr,
                test_max_drawdown=test_result.max_drawdown,
                test_n_trades=test_result.trades.n_trades,
            )
        )

    if not folds:
        raise ValueError("Kein Fold konnte erfolgreich evaluiert werden.")

    # Aggregation
    is_sharpe_mean = sum(f.train_sharpe for f in folds) / len(folds)
    oos_sharpe_mean = sum(f.test_sharpe for f in folds) / len(folds)

    degradation = 0.0
    if is_sharpe_mean > 0:
        degradation = max(0.0, oos_sharpe_mean / is_sharpe_mean)

    return WalkForwardResult(
        strategy_id=spec.strategy_id,
        n_folds=n_folds,
        rank_by=rank_by,
        folds=folds,
        is_sharpe_mean=is_sharpe_mean,
        oos_sharpe_mean=oos_sharpe_mean,
        degradation=degradation,
    )
