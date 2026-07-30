"""Portfolio-Policy aus `config/research_rules/governance.yaml`.

Wie viel Konzentration ist erlaubt, welche Sizing-Methode gilt, wird
Vol-getargetet — das sind **Governance**-Entscheidungen, keine Laufzeit-
Parameter. Sie stehen deshalb im selben YAML wie die Score-Gewichte und
werden hier in ein typisiertes Objekt geladen (unbekannte Keys werden
ignoriert, fehlende fallen auf konservative Defaults zurück).

Konservativ heißt hier: ``sizing=equal``, kein Vol-Targeting, kein Hebel.
Ohne bewusste Freigabe im YAML ändert sich das Verhalten der Registry also
nicht gegenüber dem Stand vor diesem Modul.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantrace.portfolio.construction import SIZING_METHODS, Constraints
from quantrace.portfolio.risk_model import MIN_OBS

#: Zulässige Basis für die Risikobudgets im Risk-Budgeting-Modus.
BUDGET_BASES = ("equal", "score")


@dataclass(frozen=True, slots=True)
class PortfolioPolicy:
    """Governance-Sicht auf die Portfolio-Konstruktion."""

    sizing: str = "equal"
    constraints: Constraints = Constraints()
    risk_budget_basis: str = "equal"
    risk_aversion: float = 5.0
    min_obs: int = MIN_OBS

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> PortfolioPolicy:
        """Liest den ``portfolio:``-Block. Ungültige Werte → Default + kein Crash."""
        block = (cfg or {}).get("portfolio") or {}

        sizing = str(block.get("sizing", "equal"))
        if sizing not in SIZING_METHODS:
            sizing = "equal"

        basis = str(block.get("risk_budget_basis", "equal"))
        if basis not in BUDGET_BASES:
            basis = "equal"

        constraints = Constraints(
            max_weight=_float(block.get("max_weight"), 1.0),
            min_weight=_float(block.get("min_weight"), 0.0),
            gross=_float(block.get("gross"), 1.0),
            max_turnover=_opt_float(block.get("max_turnover")),
            target_volatility=_opt_float(block.get("target_volatility")),
            max_leverage=_float(block.get("max_leverage"), 1.0),
        )
        return cls(
            sizing=sizing,
            constraints=constraints,
            risk_budget_basis=basis,
            risk_aversion=_float(block.get("risk_aversion"), 5.0),
            min_obs=int(_float(block.get("min_obs"), float(MIN_OBS))),
        )

    @classmethod
    def load(cls, path: Path) -> PortfolioPolicy:
        """Policy aus einer governance.yaml. Fehlt die Datei → Defaults."""
        import yaml

        try:
            data = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError):
            return cls()
        return cls.from_config(data if isinstance(data, dict) else {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "sizing": self.sizing,
            "risk_budget_basis": self.risk_budget_basis,
            "risk_aversion": self.risk_aversion,
            "min_obs": self.min_obs,
            "max_weight": self.constraints.max_weight,
            "min_weight": self.constraints.min_weight,
            "gross": self.constraints.gross,
            "max_turnover": self.constraints.max_turnover,
            "target_volatility": self.constraints.target_volatility,
            "max_leverage": self.constraints.max_leverage,
        }


def _float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default  # NaN → default


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


__all__ = ["BUDGET_BASES", "PortfolioPolicy"]
