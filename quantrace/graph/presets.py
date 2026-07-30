"""Referenz-Graphen — bestehende Templates als Graph-IR (#179).

Doppelte Rolle: (1) Paritäts-Fixtures — `tests/test_graph.py` assertet
bit-identische Signale gegen die Python-Klassen; (2) Vorlagen für den
visuellen Editor (#180) — niemand startet auf leerem Canvas.

Cross-Sectional (momentum_12_1, dual_momentum) fehlt bewusst: Ranking über
Symbole ist ein anderes Rechenmodell als per-Symbol-Signale (v2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def sma_crossover(fast: int = 20, slow: int = 100) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "close", "type": "source.close"},
            {"id": "fast", "type": "indicator.sma", "params": {"window": fast},
             "inputs": {"series": "close"}},
            {"id": "slow", "type": "indicator.sma", "params": {"window": slow},
             "inputs": {"series": "close"}},
            {"id": "long", "type": "logic.gt", "inputs": {"a": "fast", "b": "slow"}},
            {"id": "signal", "type": "signal.enter_exit_on_state", "inputs": {"state": "long"}},
        ]
    }


def ema_crossover(fast: int = 12, slow: int = 26) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "close", "type": "source.close"},
            {"id": "fast", "type": "indicator.ema", "params": {"span": fast},
             "inputs": {"series": "close"}},
            {"id": "slow", "type": "indicator.ema", "params": {"span": slow},
             "inputs": {"series": "close"}},
            {"id": "long", "type": "logic.gt", "inputs": {"a": "fast", "b": "slow"}},
            {"id": "signal", "type": "signal.enter_exit_on_state", "inputs": {"state": "long"}},
        ]
    }


def bollinger_bands(lookback: int = 20, k: float = 2.0) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "close", "type": "source.close"},
            {"id": "mean", "type": "indicator.sma", "params": {"window": lookback},
             "inputs": {"series": "close"}},
            {"id": "std", "type": "indicator.rolling_std", "params": {"window": lookback},
             "inputs": {"series": "close"}},
            {"id": "band", "type": "math.scale", "params": {"factor": k},
             "inputs": {"series": "std"}},
            {"id": "lower", "type": "math.sub", "inputs": {"a": "mean", "b": "band"}},
            {"id": "enter", "type": "logic.lt", "inputs": {"a": "close", "b": "lower"}},
            {"id": "exit", "type": "logic.ge", "inputs": {"a": "close", "b": "mean"}},
            {"id": "signal", "type": "signal.enter_exit",
             "inputs": {"enter": "enter", "exit": "exit"}},
        ]
    }


def rsi_2(
    period: int = 2, entry_rsi: float = 10.0, exit_rsi: float = 70.0, trend_sma: int = 200
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [
        {"id": "close", "type": "source.close"},
        {"id": "rsi", "type": "indicator.rsi", "params": {"period": period},
         "inputs": {"series": "close"}},
        {"id": "oversold", "type": "logic.lt_value", "params": {"value": entry_rsi},
         "inputs": {"series": "rsi"}},
        {"id": "exit", "type": "logic.gt_value", "params": {"value": exit_rsi},
         "inputs": {"series": "rsi"}},
    ]
    if trend_sma > 0:
        nodes += [
            {"id": "trend_ma", "type": "indicator.sma", "params": {"window": trend_sma},
             "inputs": {"series": "close"}},
            {"id": "trend_ok", "type": "logic.gt", "inputs": {"a": "close", "b": "trend_ma"}},
            {"id": "enter", "type": "logic.and", "inputs": {"a": "oversold", "b": "trend_ok"}},
        ]
        enter_id = "enter"
    else:
        enter_id = "oversold"
    nodes.append(
        {"id": "signal", "type": "signal.enter_exit",
         "inputs": {"enter": enter_id, "exit": "exit"}}
    )
    return {"nodes": nodes}


def donchian_breakout(entry_period: int = 20, exit_period: int = 10) -> dict[str, Any]:
    return {
        "nodes": [
            {"id": "close", "type": "source.close"},
            {"id": "high", "type": "source.high"},
            {"id": "low", "type": "source.low"},
            {"id": "hh", "type": "indicator.rolling_max", "params": {"window": entry_period},
             "inputs": {"series": "high"}},
            {"id": "upper", "type": "transform.shift", "params": {"periods": 1},
             "inputs": {"series": "hh"}},
            {"id": "ll", "type": "indicator.rolling_min", "params": {"window": exit_period},
             "inputs": {"series": "low"}},
            {"id": "lower", "type": "transform.shift", "params": {"periods": 1},
             "inputs": {"series": "ll"}},
            {"id": "enter", "type": "logic.gt", "inputs": {"a": "close", "b": "upper"}},
            {"id": "exit", "type": "logic.lt", "inputs": {"a": "close", "b": "lower"}},
            {"id": "signal", "type": "signal.enter_exit",
             "inputs": {"enter": "enter", "exit": "exit"}},
        ]
    }


#: name → Builder mit Default-Params (Editor-Vorlagen + Test-Fixtures)
PRESETS: dict[str, Callable[..., dict[str, Any]]] = {
    "sma_crossover": sma_crossover,
    "ema_crossover": ema_crossover,
    "bollinger_bands": bollinger_bands,
    "rsi_2": rsi_2,
    "donchian_breakout": donchian_breakout,
}
