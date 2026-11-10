"""Der Risk-Agent darf nur reduzieren — und schon der Typ lässt nichts anderes zu.

Drei Aktionen, keine vierte. Es gibt hier keinen Konstruktor, der eine Seite
oder eine Zielgrösse annimmt: ``Reduzieren`` trägt einen Anteil in (0, 1],
``Schliessen`` gar keine Zahl. Aus jeder Aktion werden ausschliesslich
Verkaufsaufträge über höchstens den gehaltenen Bestand.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from quantrace.brokers.base import OrderSide
from quantrace.hub.box import Auftrag


@dataclass(frozen=True, slots=True)
class Reduzieren:
    agent_id: str
    symbol: str
    #: Welcher Teil des Bestands verkauft wird. 1,0 = alles.
    anteil: float

    def __post_init__(self) -> None:
        a = self.anteil
        if not isinstance(a, (int, float)) or isinstance(a, bool) or not math.isfinite(a) or not 0 < a <= 1:
            raise ValueError(f"anteil muss in (0, 1] liegen, nicht {a!r}")


@dataclass(frozen=True, slots=True)
class Schliessen:
    agent_id: str
    symbol: str


@dataclass(frozen=True, slots=True)
class Einfrieren:
    """Keine Order — ein Zustand. Umgesetzt wird er in der Verlustleiter."""

    agent_id: str


RiskAktion = Reduzieren | Schliessen | Einfrieren


def auftraege(aktion: RiskAktion, positions: Mapping[str, float]) -> list[Auftrag]:
    """Die Aktion in Aufträge übersetzen. Immer ``SELL``, nie mehr als der Bestand."""
    if isinstance(aktion, Einfrieren):
        return []
    bestand = positions.get(aktion.symbol, 0.0)
    if not (isinstance(bestand, (int, float)) and math.isfinite(bestand) and bestand > 0):
        return []
    menge = bestand * aktion.anteil if isinstance(aktion, Reduzieren) else bestand
    return [Auftrag(aktion.agent_id, aktion.symbol, OrderSide.SELL, min(menge, bestand))]


def glattstellen(agent_id: str, positions: Mapping[str, float]) -> list[Auftrag]:
    """Kill-Switch und Einfrieren: alles verkaufen, was der Agent hält."""
    return [
        Auftrag(agent_id, symbol, OrderSide.SELL, menge)
        for symbol, menge in sorted(positions.items())
        if isinstance(menge, (int, float)) and math.isfinite(menge) and menge > 0
    ]
