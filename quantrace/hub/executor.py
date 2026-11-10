"""Der einzige Weg von einem Hub-Auftrag zum Broker — und er führt durch die Box.

Dieses Modul ist das **einzige** unter ``quantrace/hub/``, das einen Broker
anfasst (``tests/quantrace/test_hub_box.py`` prüft das auf dem AST). Es
übernimmt die Garantien von ``quantrace/paper/executor.py`` und legt die Box
davor:

* **Paper als Vorgabe.** Live verlangt ``allow_live`` **und** ``live_enabled``
  **und** ein Mandat, dessen Stufe über ``paper`` liegt.
* **Idempotent.** Die ``client_id`` ist deterministisch; ein zweiter Lauf am
  selben Tag wird vom Broker als Duplikat abgewiesen.
* **Die Limits werden bei jeder Order frisch gelesen.** Eine Absenkung per
  Merge gilt ab der nächsten Order, nicht ab dem nächsten Neustart.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from quantrace.brokers.base import Broker, Order
from quantrace.hub import limits as _limits
from quantrace.hub.box import AgentZustand, Allow, Auftrag, Entscheidung, Instrument, Reject, check
from quantrace.paper.executor import LiveExecutionBlocked

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Ergebnis:
    entscheidung: Entscheidung
    client_id: str | None = None
    order_id: str | None = None
    error: str | None = None

    @property
    def eingereicht(self) -> bool:
        return self.order_id is not None


def reiche_ein(
    auftrag: Auftrag,
    *,
    zustand: AgentZustand,
    markt: Mapping[str, Instrument],
    broker: Broker,
    hub_limits: _limits.HubLimits | None = None,
    live: bool = False,
    allow_live: bool = False,
    live_enabled: bool = False,
    client_tag: str | None = None,
) -> Ergebnis:
    """Prüfen, dann — und nur dann — einreichen."""
    grenzen = hub_limits if hub_limits is not None else _limits.lade()
    mandat = grenzen.mandat(auftrag.agent_id)

    if live and not (allow_live and live_enabled and mandat is not None and mandat.stage != "paper"):
        raise LiveExecutionBlocked(
            "Live verlangt allow_live, live_trading.enabled und ein Mandat über der Stufe "
            "'paper'. Abgelehnt — gegen Paper laufen lassen."
        )

    entscheidung = check(
        auftrag, zustand, mandat, markt, kill_switch=_limits.kill_switch_aktiv(grenzen)
    )
    if isinstance(entscheidung, Reject):
        log.info("hub: %s %s %s abgelehnt: %s", auftrag.agent_id, auftrag.side.value,
                 auftrag.symbol, "; ".join(entscheidung.gruende))  # fmt: skip
        return Ergebnis(entscheidung)

    menge = entscheidung.quantity
    if not isinstance(entscheidung, Allow):
        log.info("hub: %s %s %s gekürzt auf %s", auftrag.agent_id, auftrag.side.value,
                 auftrag.symbol, menge)  # fmt: skip
    tag = client_tag or date.today().isoformat()
    client_id = f"hub:{auftrag.agent_id}:{tag}:{auftrag.symbol}:{auftrag.side.value}"
    order = Order(symbol=auftrag.symbol, side=auftrag.side, quantity=menge, client_id=client_id)
    try:
        order_id = broker.submit(order)
    except Exception as exc:  # noqa: BLE001 — ein Broker-Fehler ist ein Ergebnis, kein Absturz
        log.warning("hub: submit für %s gescheitert: %s", auftrag.symbol, exc)
        return Ergebnis(entscheidung, client_id=client_id, error=str(exc))
    return Ergebnis(entscheidung, client_id=client_id, order_id=str(order_id))
