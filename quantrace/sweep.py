"""Parameter-Sweep — Grid-Search über den param_space einer StrategySpec.

Erzeugt das kartesische Produkt aller Werte in `StrategySpec.param_space`,
führt pro Kombination einen Backtest durch und liefert die Ergebnisse
sortiert nach einer konfigurierbaren Metrik zurück.

Beispiel:
    spec.param_space = {"fast": [10, 20, 50], "slow": [100, 200]}
    → 6 Backtests (3×2)
"""

from __future__ import annotations

import itertools
import logging
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from quantrace.backtest_runner import run_backtest
from quantrace.models import (
    BacktestConfig,
    BacktestResult,
    MarketData,
    StrategySpec,
)

log = logging.getLogger(__name__)

# Metriken, nach denen sortiert werden kann (höher = besser, außer max_drawdown/ulcer)
RANKABLE_METRICS = Literal[
    "sharpe",
    "sortino",
    "cagr",
    "calmar",
    "max_drawdown",
    "ulcer_index",
    "profit_factor",
]

# Metriken, bei denen niedriger besser ist (Sortierung aufsteigend)
_LOWER_IS_BETTER = {"max_drawdown", "ulcer_index"}


class SweepRun(BaseModel):
    """Ein einzelner Sweep-Lauf: Parameter + Backtest-Ergebnis."""

    params: dict[str, Any]
    result: BacktestResult


class SweepResult(BaseModel):
    """Ergebnis eines kompletten Parameter-Sweeps."""

    strategy_id: str
    param_space: dict[str, list[Any]]
    rank_by: str
    total_combinations: int
    completed: int
    failed: int
    runs: list[SweepRun] = Field(default_factory=list)
    best_params: dict[str, Any] = Field(default_factory=dict)
    best_metric_value: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def best_run(self) -> SweepRun | None:
        return self.runs[0] if self.runs else None


def _extract_metric(result: BacktestResult, metric: str) -> float:
    """Liest eine Metrik aus dem BacktestResult."""
    if hasattr(result, metric):
        return float(getattr(result, metric))
    if hasattr(result.trades, metric):
        return float(getattr(result.trades, metric))
    raise ValueError(f"Unbekannte Metrik: {metric}")


def _generate_param_grid(param_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Erzeugt das kartesische Produkt aller Parameter-Werte."""
    if not param_space:
        return [{}]
    keys = sorted(param_space.keys())
    values = [param_space[k] for k in keys]
    return [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*values)]


def sweep(
    spec: StrategySpec,
    data: MarketData,
    config: BacktestConfig | None = None,
    rank_by: str = "sharpe",
) -> SweepResult:
    """Führt einen vollständigen Parameter-Sweep über spec.param_space durch.

    Args:
        spec: Strategiebeschreibung mit param_space.
        data: MarketData für den Backtest.
        config: BacktestConfig (optional, sonst Default).
        rank_by: Metrik zum Ranken (Default: sharpe).

    Returns:
        SweepResult mit allen Runs, sortiert nach rank_by.

    Raises:
        ValueError: Wenn param_space leer ist oder rank_by ungültig.
    """
    if not spec.param_space:
        raise ValueError(
            f"StrategySpec '{spec.strategy_id}' hat keinen param_space. "
            "Setze spec.param_space = {'param': [wert1, wert2, ...]}."
        )

    config = config or BacktestConfig()
    grid = _generate_param_grid(spec.param_space)
    total = len(grid)

    log.info(
        "Sweep für '%s': %d Kombinationen aus %s",
        spec.strategy_id,
        total,
        {k: len(v) for k, v in spec.param_space.items()},
    )

    runs: list[SweepRun] = []
    failed = 0

    for i, params in enumerate(grid, 1):
        # Neues Spec mit konkreten Params erzeugen
        run_spec = spec.model_copy(update={"params": params})
        run_id = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
        log.info("  [%d/%d] %s", i, total, run_id)

        try:
            result = run_backtest(run_spec, data, config)
            runs.append(SweepRun(params=params, result=result))
        except Exception as exc:
            log.warning("  FAIL [%d/%d] %s: %s", i, total, run_id, exc)
            failed += 1

    # Sortieren
    reverse = rank_by not in _LOWER_IS_BETTER
    runs.sort(key=lambda r: _extract_metric(r.result, rank_by), reverse=reverse)

    best_params: dict[str, Any] = {}
    best_value = 0.0
    if runs:
        best_params = runs[0].params
        best_value = _extract_metric(runs[0].result, rank_by)

    log.info(
        "Sweep fertig: %d/%d erfolgreich. Beste %s = %.4f mit %s",
        len(runs),
        total,
        rank_by,
        best_value,
        best_params,
    )

    return SweepResult(
        strategy_id=spec.strategy_id,
        param_space=spec.param_space,
        rank_by=rank_by,
        total_combinations=total,
        completed=len(runs),
        failed=failed,
        runs=runs,
        best_params=best_params,
        best_metric_value=best_value,
    )
