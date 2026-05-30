"""Strategy-Interface.

Jede Strategie ist eine Klasse, die `generate_signals(data, params)` implementiert
und zwei boolesche DataFrames zurückgibt: entries und exits. Damit ist die
Backtest-Engine austauschbar.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from quantrace.data_agent import close_prices
from quantrace.models import MarketData, StrategySpec


class Strategy(ABC):
    """Basisklasse für jede Strategie. Stateless: alle Parameter kommen rein."""

    #: optional default params; werden mit StrategySpec.params gemerged
    defaults: dict[str, Any] = {}

    def __init__(self, **params: Any) -> None:
        self.params = {**self.defaults, **params}

    @abstractmethod
    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returnt (entries, exits) als bool-DataFrames mit denselben columns wie close."""

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def close(data: MarketData) -> pd.DataFrame:
        return close_prices(data)


def load_strategy(spec: StrategySpec) -> Strategy:
    """Importiert eine Strategie über ihren class_path und instanziiert sie."""
    module_path, _, class_name = spec.class_path.partition(":")
    if not class_name:
        raise ValueError(f"class_path muss 'modul:Klasse' sein, war: {spec.class_path}")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not issubclass(cls, Strategy):
        raise TypeError(f"{cls} ist keine Strategy-Subklasse")
    return cls(**spec.params)
