"""Die Prüfung vor jeder Order: ``check(auftrag, …) -> Allow | Reduce | Reject``.

Eine reine Funktion — kein Netz, keine Uhr, kein Zustand. Was sie braucht,
bekommt sie übergeben, und was ihr fehlt, lehnt sie ab:

* kein Mandat, kein Kurs, keine Liquiditätsangabe → ``Reject``
* eine gehaltene Position ohne Kurs → ``Reject`` für jeden Kauf, weil sich das
  Exposure dann nicht rechnen lässt

**Verkäufe eines Bestands gehen immer durch** — auch bei Kill-Switch, auch im
eingefrorenen Zustand. Eine Box, die das Schliessen verhindert, hält das
Risiko fest, das sie begrenzen soll. Leerverkäufe sind dagegen nicht im
Mandat: ein Verkauf wird auf den Bestand gekürzt.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from quantrace.brokers.base import OrderSide
from quantrace.hub.ladder import Stand
from quantrace.hub.limits import Mandat

#: Stückzahlen werden auf sechs Stellen **abgerundet**. Aufrunden könnte ein
#: Limit um einen Bruchteil überschreiten — klein, aber die falsche Richtung.
_STELLEN = 1_000_000


@dataclass(frozen=True, slots=True)
class Auftrag:
    agent_id: str
    symbol: str
    side: OrderSide
    quantity: float


@dataclass(frozen=True, slots=True)
class Instrument:
    """Was die Box über ein Papier wissen muss. ``None`` heisst unbekannt —
    und unbekannt ist nicht „gross genug"."""

    price: float | None
    adv_usd: float | None = None
    market_cap_usd: float | None = None


@dataclass(frozen=True, slots=True)
class AgentZustand:
    equity_usd: float
    positions: Mapping[str, float] = field(default_factory=dict)
    leiter: Stand = field(default_factory=Stand)


@dataclass(frozen=True, slots=True)
class Allow:
    quantity: float


@dataclass(frozen=True, slots=True)
class Reduce:
    quantity: float
    gruende: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Reject:
    gruende: tuple[str, ...]


Entscheidung = Allow | Reduce | Reject


def _positiv(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _ab(x: float) -> float:
    return math.floor(x * _STELLEN) / _STELLEN


def check(
    auftrag: Auftrag,
    zustand: AgentZustand,
    mandat: Mandat | None,
    markt: Mapping[str, Instrument],
    *,
    kill_switch: bool,
) -> Entscheidung:
    if not _positiv(auftrag.quantity):
        return Reject((f"Stückzahl {auftrag.quantity!r} ist keine positive Zahl",))
    if mandat is None or mandat.agent_id != auftrag.agent_id:
        return Reject((f"kein Mandat für Agent {auftrag.agent_id!r}",))

    bestand = zustand.positions.get(auftrag.symbol, 0.0)

    if auftrag.side is OrderSide.SELL:
        if not _positiv(bestand):
            return Reject((f"kein Bestand in {auftrag.symbol} — Leerverkauf ist nicht im Mandat",))
        if auftrag.quantity > bestand:
            return Reduce(bestand, (f"Verkauf auf den Bestand von {bestand} gekürzt",))
        return Allow(auftrag.quantity)

    if auftrag.side is not OrderSide.BUY:
        return Reject((f"unbekannte Seite {auftrag.side!r}",))

    # ---- ab hier: ein Kauf, also neues Risiko
    if kill_switch:
        return Reject(("Kill-Switch ist aktiv — keine neuen Positionen",))
    if not zustand.leiter.darf_eroeffnen:
        return Reject((f"Verlustleiter steht auf {zustand.leiter.stufe.value}",))
    if not mandat.aktiv:
        return Reject((f"Mandat von {mandat.agent_id!r} ist leer (Kapital oder Instrumente fehlen)",))
    if auftrag.symbol not in mandat.instruments:
        return Reject((f"{auftrag.symbol} steht nicht auf der Instrumentliste",))

    papier = markt.get(auftrag.symbol)
    if papier is None or not _positiv(papier.price):
        return Reject((f"kein gültiger Kurs für {auftrag.symbol}",))
    preis = float(papier.price)  # type: ignore[arg-type]

    gruende: list[str] = []
    if preis < mandat.min_price:
        gruende.append(f"Kurs {preis} unter Mindestkurs {mandat.min_price}")
    for name, wert, grenze in (
        ("ADV", papier.adv_usd, mandat.min_adv_usd),
        ("Marktkapitalisierung", papier.market_cap_usd, mandat.min_market_cap_usd),
    ):
        if grenze <= 0:
            continue
        if not _positiv(wert):
            gruende.append(f"{name} von {auftrag.symbol} unbekannt")
        elif wert < grenze:  # type: ignore[operator]
            gruende.append(f"{name} {wert:,.0f} $ unter Minimum {grenze:,.0f} $")
    if gruende:
        return Reject(tuple(gruende))

    if not _positiv(zustand.equity_usd):
        return Reject(("Eigenkapital des Agenten ist nicht positiv",))
    # Die kleinere der beiden Zahlen: Gewinne erhöhen das Mandat nicht von
    # selbst, Verluste verkleinern, was noch riskiert werden darf.
    basis = min(mandat.capital_usd, zustand.equity_usd)

    brutto = 0.0
    for symbol, menge in zustand.positions.items():
        if not menge:
            continue
        p = markt.get(symbol)
        if p is None or not _positiv(p.price):
            return Reject((f"gehaltene Position {symbol} hat keinen Kurs — Exposure nicht rechenbar",))
        brutto += abs(menge) * float(p.price)  # type: ignore[arg-type]

    raum_brutto = mandat.max_gross_exposure * basis - brutto
    raum_position = mandat.max_position_pct * basis - max(bestand, 0.0) * preis
    max_menge = _ab(min(raum_brutto, raum_position) / preis)
    if max_menge <= 0:
        eng = "Brutto-Exposure" if raum_brutto <= raum_position else "Positionsgrösse"
        return Reject((f"{eng} ist ausgeschöpft",))
    if auftrag.quantity > max_menge:
        eng = "Brutto-Exposure" if raum_brutto <= raum_position else "Positionsgrösse"
        return Reduce(max_menge, (f"auf {max_menge} gekürzt — Grenze: {eng}",))
    return Allow(auftrag.quantity)
