"""Die Verlustleiter als Zustandsmaschine (ADR-018, Entscheidung 4).

    active ──Tagesverlust──▶ paused_today ──Tageswechsel──▶ active
       │                          │
       └────────Drawdown──────────┴──▶ frozen ──▶ retired

Nach unten geht es von selbst, nach oben nur an **einer** Stelle: eine
Tagespause endet mit dem Tag. Aus `frozen` und `retired` führt in diesem Modul
kein Weg heraus — es gibt keine Funktion dafür, nicht nur keine Erlaubnis.
Wiederbelebt wird ein Agent durch einen neuen Eintrag in der hub.yaml, und das
ist ein neuer Versuch, der gezählt wird. Ein Agent, der nach Verlusten neu
startet, probiert, bis er Glück hat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum

from quantrace.hub.limits import Verlustleiter


class Stufe(str, Enum):
    ACTIVE = "active"
    PAUSED_TODAY = "paused_today"
    FROZEN = "frozen"
    RETIRED = "retired"


class Folge(str, Enum):
    """Was beim Übergang zu tun ist. Die Maschine ordnet an, sie führt nicht aus."""

    KEINE_NEUEN_POSITIONEN = "keine_neuen_positionen"
    POSITIONEN_SCHLIESSEN = "positionen_schliessen"
    KAPITAL_EINFRIEREN = "kapital_einfrieren"
    POST_MORTEM = "post_mortem"
    STILLLEGEN = "stilllegen"


@dataclass(frozen=True, slots=True)
class Stand:
    stufe: Stufe = Stufe.ACTIVE
    #: Der Tag, an dem die Stufe erreicht wurde — für das Ende der Tagespause.
    seit: date | None = None

    @property
    def darf_eroeffnen(self) -> bool:
        return self.stufe is Stufe.ACTIVE


def schritt(
    stand: Stand,
    leiter: Verlustleiter,
    *,
    heute: date,
    tagesverlust: float,
    drawdown: float,
    regelbruch: bool = False,
) -> tuple[Stand, tuple[Folge, ...]]:
    """Einen Messpunkt verarbeiten.

    ``tagesverlust`` und ``drawdown`` sind **positive** Anteile des
    Agent-Kapitals (0,03 = drei Prozent Verlust); ein Gewinn ist 0 oder
    negativ. Eine Zahl, die keine ist, wirft: ein Risikomass, das fehlt, darf
    nicht als „kein Verlust" durchgehen.

    Die Folgen werden nur beim **Übergang** gemeldet, nicht bei jedem
    weiteren Messpunkt in derselben Stufe — sonst entstünde jede Minute eine
    neue Post-Mortem-Note.
    """
    for name, wert in (("tagesverlust", tagesverlust), ("drawdown", drawdown)):
        if not isinstance(wert, (int, float)) or not math.isfinite(wert):
            raise ValueError(f"{name} ist keine endliche Zahl ({wert!r})")

    if stand.stufe is Stufe.RETIRED:
        return stand, ()

    if regelbruch or drawdown >= leiter.drawdown_retire:
        folgen = [Folge.STILLLEGEN]
        if stand.stufe is not Stufe.FROZEN:
            # Direkt von oben hierher: was die Stufe darüber angeordnet hätte,
            # gilt mit.
            folgen = [Folge.POSITIONEN_SCHLIESSEN, Folge.KAPITAL_EINFRIEREN, Folge.POST_MORTEM, *folgen]
        return Stand(Stufe.RETIRED, heute), tuple(folgen)

    if stand.stufe is Stufe.FROZEN:
        return stand, ()

    if drawdown >= leiter.drawdown_freeze:
        return Stand(Stufe.FROZEN, heute), (
            Folge.POSITIONEN_SCHLIESSEN,
            Folge.KAPITAL_EINFRIEREN,
            Folge.POST_MORTEM,
        )

    if stand.stufe is Stufe.PAUSED_TODAY:
        if stand.seit is not None and heute <= stand.seit:
            return stand, ()
        stand = Stand(Stufe.ACTIVE, heute)  # neuer Tag — und gleich neu messen

    if tagesverlust >= leiter.daily_loss_pause:
        return Stand(Stufe.PAUSED_TODAY, heute), (Folge.KEINE_NEUEN_POSITIONEN,)

    return stand, ()
