"""Strategy-Registry — Single Source of Truth für CLI, Pipeline und Sweeps.

Jede Strategie im `strategies/templates/`-Baukasten wird hier einmal registriert:
ihr `class_path` (für `load_strategy`), ihre fachliche Klasse (für Vault-Tags) und
ein Default-Grid für Sweeps / Walk-Forward.

Vorher kannte das CLI nur `sma_crossover` und `mean_reversion` hart verdrahtet —
alle anderen 8 Strategien aus dem Katalog crashten beim Start. Diese Registry
schließt die Lücke zwischen `config/strategy_catalog.yaml` (Webapp-Dropdown) und
dem tatsächlichen Ausführungspfad.
"""

from __future__ import annotations

import importlib
from typing import Any

from quantrace.models import StrategySpec, Timeframe

# strategy_id → (class_path "modul:Klasse", strategy_class-Kategorie)
# strategy_class muss zum Literal in models.StrategySpec passen.
REGISTRY: dict[str, tuple[str, str]] = {
    "sma_crossover": ("strategies.templates.sma_crossover:SmaCrossover", "trend_following"),
    "ema_crossover": ("strategies.templates.ema_crossover:EmaCrossover", "trend_following"),
    "macd": ("strategies.templates.macd:Macd", "trend_following"),
    "donchian_breakout": ("strategies.templates.donchian_breakout:DonchianBreakout", "trend_following"),
    "atr_breakout": ("strategies.templates.atr_breakout:AtrBreakout", "volatility"),
    "mean_reversion": ("strategies.templates.mean_reversion:MeanReversion", "mean_reversion"),
    "bollinger_bands": ("strategies.templates.bollinger_bands:BollingerBands", "mean_reversion"),
    "rsi_2": ("strategies.templates.rsi_2:Rsi2", "mean_reversion"),
    "momentum_12_1": ("strategies.templates.momentum_12_1:CrossSectionalMomentum", "cross_sectional"),
    "dual_momentum": ("strategies.templates.dual_momentum:DualMomentum", "cross_sectional"),
    "kalman_trend": ("strategies.templates.kalman_trend:KalmanTrend", "trend_following"),
    "regime_filter": ("strategies.templates.regime_filter:RegimeFilter", "trend_following"),
    "buy_and_hold": ("strategies.templates.buy_and_hold:BuyAndHold", "custom"),
}

# Default-Param-Grids für Sweep / Walk-Forward. Kleine, sinnvolle Gitter
# orientiert an config/strategy_catalog.yaml. Strategien ohne Parameter
# (buy_and_hold) haben kein Grid — Sweep/WF sind dort sinnlos.
DEFAULT_GRIDS: dict[str, dict[str, list[Any]]] = {
    "sma_crossover": {"fast": [5, 10, 20, 50], "slow": [100, 150, 200]},
    "ema_crossover": {"fast": [8, 12, 20], "slow": [26, 50, 100]},
    "macd": {"fast": [8, 12], "slow": [21, 26], "signal": [9]},
    "donchian_breakout": {"entry_period": [20, 40, 55], "exit_period": [10, 20]},
    "atr_breakout": {"lookback": [14, 20, 30], "k": [1.5, 2.0, 3.0]},
    "mean_reversion": {"lookback": [10, 20, 30], "entry_z": [1.5, 2.0, 2.5], "exit_z": [0.0]},
    "bollinger_bands": {"lookback": [10, 20, 30], "k": [1.5, 2.0, 2.5]},
    "rsi_2": {"period": [2, 3], "entry_rsi": [5.0, 10.0, 15.0], "exit_rsi": [65.0, 70.0], "trend_sma": [200]},
    "momentum_12_1": {"lookback": [126, 252], "skip": [21], "top_quantile": [0.2, 0.3]},
    "dual_momentum": {"lookback": [126, 252], "top_quantile": [0.3, 0.5], "abs_threshold": [0.0]},
    "kalman_trend": {"delta": [1e-5, 1e-4, 1e-3], "meas_var": [1e-3]},
    "regime_filter": {"trend_lookback": [50, 100, 200], "feature_window": [21, 63]},
}

# strategy_id → Funktion die einen menschenlesbaren, eindeutigen Slug aus den
# Params baut. Erhält die historischen Slugs für die zwei Alt-Strategien, damit
# bestehende Vault-Notes / Result-Files nicht forken.
_ID_FORMATTERS = {
    "sma_crossover": lambda p: f"sma_{p['fast']}_{p['slow']}",
    "mean_reversion": lambda p: f"mr_{p['lookback']}_{p['entry_z']}_{p['exit_z']}",
}


def known_strategies() -> list[str]:
    return sorted(REGISTRY)


def is_registered(strategy_id: str) -> bool:
    return strategy_id in REGISTRY


def has_param_space(strategy_id: str) -> bool:
    """True, wenn die Strategie tunebare Parameter hat (Sweep/WF sinnvoll)."""
    return bool(DEFAULT_GRIDS.get(strategy_id))


def class_defaults(strategy_id: str) -> dict[str, Any]:
    """Liest die `defaults` der Strategie-Klasse (ohne sie auszuführen)."""
    class_path, _ = _require(strategy_id)
    module_path, _, class_name = class_path.partition(":")
    cls = getattr(importlib.import_module(module_path), class_name)
    return dict(getattr(cls, "defaults", {}) or {})


def default_param_space(strategy_id: str) -> dict[str, list[Any]]:
    """Default-Grid für Sweep / Walk-Forward; leer wenn parameterlos."""
    return {k: list(v) for k, v in DEFAULT_GRIDS.get(strategy_id, {}).items()}


def build_spec(
    strategy_id: str,
    universe: str,
    timeframe: Timeframe,
    params: dict[str, Any] | None = None,
) -> tuple[str, StrategySpec]:
    """Baut eine StrategySpec aus Registry + Klassen-Defaults + Override-Params."""
    class_path, category = _require(strategy_id)
    merged = {**class_defaults(strategy_id), **(params or {})}
    sid = _make_id(strategy_id, merged)
    spec = StrategySpec(
        strategy_id=sid,
        name=f"{strategy_id} {merged}" if merged else strategy_id,
        class_path=class_path,
        strategy_class=category,  # type: ignore[arg-type]
        universe=universe,
        timeframe=timeframe,
        params=merged,
    )
    return sid, spec


def _require(strategy_id: str) -> tuple[str, str]:
    try:
        return REGISTRY[strategy_id]
    except KeyError as e:
        raise KeyError(
            f"Unbekannte Strategie {strategy_id!r}. Bekannt: {', '.join(known_strategies())}"
        ) from e


def _make_id(strategy_id: str, params: dict[str, Any]) -> str:
    fmt = _ID_FORMATTERS.get(strategy_id)
    if fmt is not None:
        try:
            return fmt(params)
        except KeyError:
            pass
    if not params:
        return strategy_id
    suffix = "_".join(str(params[k]) for k in sorted(params))
    return f"{strategy_id}_{suffix}"
