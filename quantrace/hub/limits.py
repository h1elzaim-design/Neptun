"""Das Mandat eines Agenten — gelesen aus ``config/risk_limits/hub.yaml``.

**Erhöht wird nur per gemergtem PR.** Dieses Modul liest; es gibt keine
Funktion, die schreibt. Und es liest streng: ein Mandat, das sich nicht
vollständig und widerspruchsfrei lesen lässt, ist ein Fehler beim Laden, kein
Mandat mit Vorgabewerten. Die Richtung ist überall dieselbe — was fehlt, ist
nicht erlaubt:

* kein Eintrag für den Agenten        → kein Mandat, jede Order wird abgelehnt
* ``instruments`` fehlt oder ist leer → nichts ist handelbar
* ``capital_usd: 0``                  → kein Mandat
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

HUB_YAML = Path(__file__).resolve().parents[2] / "config" / "risk_limits" / "hub.yaml"

#: Die Aufstiegsleiter (ADR-018, Entscheidung 5). Jede Stufe ist ein Merge.
STUFEN = ("paper", "mini_live", "scaled", "levered")

#: Der harte Deckel aus ADR-018, Entscheidung 6. Steht im Code, nicht in der
#: YAML: eine Zahl, die man per Konfiguration anheben kann, ist kein Deckel.
MAX_HEBEL = 2.0


class MandatFehlerError(ValueError):
    """Die hub.yaml beschreibt ein Mandat, das es nicht geben darf."""


@dataclass(frozen=True, slots=True)
class Verlustleiter:
    """Schwellen als positive Anteile des Agent-Kapitals."""

    daily_loss_pause: float = 0.02
    drawdown_freeze: float = 0.10
    drawdown_retire: float = 0.20

    def __post_init__(self) -> None:
        a, b, c = self.daily_loss_pause, self.drawdown_freeze, self.drawdown_retire
        if not all(math.isfinite(x) for x in (a, b, c)) or not (0 < a < 1 and 0 < b < c < 1):
            raise MandatFehlerError(
                f"Verlustleiter braucht 0 < pause < 1 und 0 < freeze < retire < 1, nicht {a}/{b}/{c}"
            )


@dataclass(frozen=True, slots=True)
class Mandat:
    agent_id: str
    capital_usd: float
    instruments: frozenset[str]
    #: Mindestliquidität — der Schutz gegen Pump-and-Dumps aus Social-Daten.
    min_adv_usd: float
    min_price: float
    min_market_cap_usd: float
    stage: str = "paper"
    #: Anteil des Kapitals in **einem** Papier.
    max_position_pct: float = 0.10
    #: Summe aller Positionen im Verhältnis zum Kapital.
    max_gross_exposure: float = 1.0
    max_leverage: float = 1.0
    ladder: Verlustleiter = field(default_factory=Verlustleiter)

    def __post_init__(self) -> None:
        def _zahl(name: str, wert: float, *, min_: float, max_: float | None = None) -> None:
            if not isinstance(wert, (int, float)) or isinstance(wert, bool) or not math.isfinite(wert):
                raise MandatFehlerError(f"{self.agent_id}: {name} ist keine endliche Zahl ({wert!r})")
            if wert < min_ or (max_ is not None and wert > max_):
                grenze = f"[{min_}, {max_}]" if max_ is not None else f">= {min_}"
                raise MandatFehlerError(f"{self.agent_id}: {name}={wert} liegt nicht in {grenze}")

        if self.stage not in STUFEN:
            raise MandatFehlerError(f"{self.agent_id}: stage {self.stage!r} ist keine von {STUFEN}")
        _zahl("capital_usd", self.capital_usd, min_=0)
        _zahl("max_position_pct", self.max_position_pct, min_=0, max_=1)
        _zahl("max_leverage", self.max_leverage, min_=0, max_=MAX_HEBEL)
        _zahl("max_gross_exposure", self.max_gross_exposure, min_=0, max_=self.max_leverage)
        _zahl("min_adv_usd", self.min_adv_usd, min_=0)
        _zahl("min_price", self.min_price, min_=0)
        _zahl("min_market_cap_usd", self.min_market_cap_usd, min_=0)
        # Hebel ist eine Stufe, kein Parameter: wer über 1× will, muss die
        # ganze Leiter hinauf, und die letzte Sprosse heisst `levered`.
        if self.max_leverage > 1.0 and self.stage != "levered":
            raise MandatFehlerError(
                f"{self.agent_id}: max_leverage={self.max_leverage} verlangt stage 'levered', "
                f"nicht {self.stage!r}"
            )

    @property
    def aktiv(self) -> bool:
        """Hat der Agent überhaupt etwas, womit und worin er handeln darf?"""
        return self.capital_usd > 0 and bool(self.instruments)


@dataclass(frozen=True, slots=True)
class HubLimits:
    kill_switch: bool
    mandate: dict[str, Mandat]

    def mandat(self, agent_id: str) -> Mandat | None:
        return self.mandate.get(agent_id)


_ERLAUBT = {
    "capital_usd", "instruments", "min_adv_usd", "min_price", "min_market_cap_usd",
    "stage", "max_position_pct", "max_gross_exposure", "max_leverage", "loss_ladder",
}  # fmt: skip
_PFLICHT = ("capital_usd", "instruments", "min_adv_usd", "min_price", "min_market_cap_usd")


def _mandat(agent_id: str, roh: Any, vorgaben: dict[str, Any]) -> Mandat:
    if not isinstance(roh, dict):
        raise MandatFehlerError(f"{agent_id}: Mandat ist kein Mapping")
    werte = {**vorgaben, **roh}
    # Ein Tippfehler (`max_postion_pct`) darf nicht still zur Vorgabe werden —
    # der Betreiber glaubte, ein Limit gesetzt zu haben.
    unbekannt = set(werte) - _ERLAUBT
    if unbekannt:
        raise MandatFehlerError(f"{agent_id}: unbekannte Felder {sorted(unbekannt)}")
    fehlend = [k for k in _PFLICHT if werte.get(k) is None]
    if fehlend:
        raise MandatFehlerError(f"{agent_id}: Pflichtfelder fehlen: {fehlend}")
    instrumente = werte.pop("instruments")
    if not isinstance(instrumente, list) or not all(isinstance(s, str) and s for s in instrumente):
        raise MandatFehlerError(f"{agent_id}: instruments muss eine Liste von Symbolen sein")
    leiter = werte.pop("loss_ladder", None)
    try:
        return Mandat(
            agent_id=agent_id,
            instruments=frozenset(instrumente),
            ladder=Verlustleiter(**leiter) if leiter else Verlustleiter(),
            **werte,
        )
    except TypeError as e:
        raise MandatFehlerError(f"{agent_id}: {e}") from e


def lade(path: Path | None = None) -> HubLimits:
    """Die Limits lesen. Wirft bei jeder Unstimmigkeit — fail closed.

    Bewusst **ohne Cache**: die Datei ändert sich durch einen Merge, und ein
    Prozess, der ein altes, grosszügigeres Mandat im Speicher behält, hätte
    eine Absenkung verschlafen.
    """
    p = path or HUB_YAML
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as e:
        raise MandatFehlerError(f"{p} fehlt — ohne Limits kein Handel") from e
    kill = raw.get("kill_switch", False)
    if not isinstance(kill, bool):
        raise MandatFehlerError(f"kill_switch muss true/false sein, nicht {kill!r}")
    vorgaben = raw.get("mandate_defaults") or {}
    mandate = {
        str(aid): _mandat(str(aid), roh, vorgaben) for aid, roh in (raw.get("mandates") or {}).items()
    }
    return HubLimits(kill_switch=kill, mandate=mandate)


def kill_switch_aktiv(limits: HubLimits) -> bool:
    """Datei **oder** Umgebung. Die Umgebung kann nur ein-, nie ausschalten.

    ``QUANTRACE_HUB_KILL=1`` ist der schnelle Weg — ein Merge dauert Minuten,
    eine Umgebungsvariable Sekunden. Zum *Aufheben* braucht es beides aus.
    """
    return limits.kill_switch or os.environ.get("QUANTRACE_HUB_KILL", "").strip().lower() in (
        "1", "true", "yes", "on",
    )  # fmt: skip
