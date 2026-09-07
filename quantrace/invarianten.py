"""Was für echte Kursdaten gelten muss — als Liste statt als Einzelfälle.

**Warum es das gibt.** Zwischen dem 2026-09-05 und dem 2026-09-07 sind fünf
Klassen von Datenfehlern aufgetaucht, jede durch einen anderen Zufall: ein
Absturz, ein Logeintrag, ein Backtest, eine Nebenfrage. Jede bekam einen
eigenen Detektor mit eigener Verdrahtung, und **jeder Detektor lag zwei bis
vier Mal daneben**, bevor er stimmte.

Der Aufwand lag nicht in der Prüfung — die ist meist zehn Zeilen — sondern in
allem drumherum: wo läuft sie, was gibt sie zurück, wie kommt das Ergebnis zum
Aufrufer, wie wird es gemeldet. Das war fünfmal dieselbe Arbeit.

Vier der fünf Klassen waren ausserdem **innere Widersprüche**, keine Fälle für
einen zweiten Anbieter:

===========================  ==========================================
Nullbars (#322)              Zeile aus lauter Nullen
Einheitenfehler (#324)       Nachbarzeilen widersprechen
Erfundene Dividende (#312)   Kurs am Ex-Tag widerspricht
`DIC`-Doppelreihe (#333)     Kurs widerspricht dem Volumen
Fehlender Split (#327)       *nur hier* braucht es eine zweite Quelle
===========================  ==========================================

Sie fielen nicht auf, weil jede bestehende Prüfung **eine Zeile für sich**
ansieht. Eine Zeile mit ``close = 0,67`` neben einer mit ``close = 30.024`` ist
einzeln betrachtet in beiden Fällen tadellos.

## Die Form

Eine Invariante bekommt die Reihe eines Papiers und gibt Befunde zurück. Mehr
nicht. Sie entscheidet **nichts** — was mit einem Befund geschieht, steht in
``data/corrections/`` (siehe ``quantrace.befunde``).

Damit ist Klasse sechs eine Funktion und ein Eintrag in ``ALLE``, kein PR.

## Was eine Invariante nicht ist

Kein Filter für unerwünschte Papiere. ``CHK`` stieg am 2020-06-08 um **182 %**
und fiel am nächsten Tag um 66 % — echt, mit passendem Volumen (21 Mio. Stück),
eine Meme-Rally kurz vor der Insolvenz. Eine Regel, die das wegwirft, hat kein
Datenproblem gelöst, sondern ein Datum gelöscht.

Der Unterschied zu ``DIC`` ist nicht die Höhe der Bewegung, sondern dass dort
**Kurs und Volumen einander widersprechen**. Genau darum geht es: Felder
gegeneinander prüfen, nicht Grenzwerte setzen.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantrace.befunde import Befund

log = logging.getLogger(__name__)

#: Eine Prüfung: bekommt die Reihe **eines** Papiers (nach Datum sortiert,
#: Spalten wie im Lake) und gibt zurück, was ihr auffiel.
Pruefung = Callable[[pd.DataFrame, str, str], list[Befund]]


@dataclass(frozen=True)
class Invariante:
    """Eine Aussage, die für echte Kursdaten gelten muss.

    ``name`` wird zur ``Befund.art`` und damit zum Schlüssel im Register — er
    ist Teil der Schnittstelle und wird nicht beiläufig umbenannt.
    """

    name: str
    #: Ein Satz: was gilt, und woran man das Gegenteil erkennt.
    beschreibung: str
    pruefen: Pruefung
    #: Woher der Fall kam. Ohne das steht in einem Jahr eine Regel da, die
    #: niemand mehr begründen kann — und die dann jemand „aufräumt".
    anlass: str = ""


def _befund(
    teil: pd.DataFrame, instrument: str, code: str, pos: int, art: str, **belege
) -> Befund:
    """Ein Befund an Position ``pos`` der Reihe."""
    return Befund(
        instrument=instrument,
        code=code,
        tag=pd.Timestamp(teil["date"].iloc[pos]).date(),
        art=art,
        belege=belege,
    )



#: Um wieviele Zehnerpotenzen der Kurs vom Median der eigenen Reihe abweichen
#: darf, bevor es ein Befund ist.
#:
#: Zwei Dekaden sind Faktor 100 — grosszügig. Ein Papier, das sich über zwanzig
#: Jahre verhundertfacht, erreicht das gegenüber seinem *Median* nie; wohl aber
#: eines, dessen Spalte zwei Instrumente enthält. ``DIC`` liegt bei 4,6
#: Dekaden, ``CHK`` an seinem Rekordtag bei 0,7.
#:
#: Die Zahl ist gesetzt und nicht abgeleitet — die Verteilung ist stetig, wie
#: am 2026-09-07 über 7,17 Mio. Tagesrenditen gemessen. Sie liegt bewusst
#: **weit** jenseits des Echten: der Zweck ist Melden, nicht Verwerfen, und ein
#: Befund kostet eine Zeile im Bericht.
KURSNIVEAU_DEKADEN = 2.0

# ---------------------------------------------------------------------------
# Die Invarianten
# ---------------------------------------------------------------------------


def _kursniveau(teil: pd.DataFrame, instrument: str, code: str) -> list[Befund]:
    """Der Kurs darf nicht um Grössenordnungen vom eigenen Niveau abweichen (#333).

    **Der Fall.** ``DIC`` springt im Juli 2010 im Tageswechsel zwischen zwei
    Niveaus::

        2010-07-13   close 28.347,20   Volumen    314.998
        2010-07-14   close      0,6718 Volumen  1.141.312   <--
        2010-07-15   close 28.666,60   Volumen    230.466
        2010-07-19   close      0,6793 Volumen  3.552.542   <--
        2010-07-20   close 32.659,10   Volumen    631.625

    Das ist keine volatile Aktie, das sind **zwei Reihen in einer Spalte**. Der
    Fehler steckt in *beiden* Kursspalten gleich, weshalb der
    Einheiten-Detektor (#324) dort blind ist: ``adjusted_close / close``
    bewegt sich nie.

    **Warum gegen den Median der Reihe und nicht gegen den Vortag.** Bei
    ``DIC`` wechselt es täglich; ein Vortagsvergleich hätte je nach Startpunkt
    die falsche Hälfte gemeldet.

    **Welches Niveau das richtige ist, sagt diese Regel nicht.** Der Median
    trifft es nur, solange die kaputten Zeilen in der Minderheit sind. Bei
    ``TRA`` sind es 2.582 von 5.576 — dort liegt der Median auf dem *falschen*
    Niveau, und gemeldet werden die gesunden Zeilen. Der Befund stimmt
    trotzdem: die Spalte enthält zwei Niveaus. Welches Terra Industries ist,
    sieht ein Mensch in einer Sekunde (3,69 $, nicht 12.260 $) — und genau
    dafür gibt es das Register.

    ``n_zeilen`` steht deshalb in den Belegen: nahe der Hälfte ist das ein
    Hinweis, dass der Median gekippt sein könnte.

    **Warum nicht das Dollarvolumen.** Der erste Versuch am 2026-09-07 prüfte
    ``close · volume`` gegen dessen Median und meldete bei ``DIC`` 344 Zeilen —
    darunter hunderte Tage mit 76 gehandelten Stück. Das ist kein Widerspruch,
    sondern ein illiquider Tag. Das Volumen steht deshalb nur noch **als Beleg**
    daneben, nicht als Kriterium.
    """
    if "close" not in teil.columns or len(teil) < 20:
        return []
    close = teil["close"].astype(float)
    brauchbar = close > 0
    if int(brauchbar.sum()) < 20:
        return []

    mitte = close.where(brauchbar).median()
    if not np.isfinite(mitte) or mitte <= 0:
        return []
    dekaden = np.log10((close / mitte).astype(float))
    schlecht = (brauchbar & (dekaden.abs() > KURSNIVEAU_DEKADEN)).fillna(False)
    if not bool(schlecht.any()):
        return []

    hat_volumen = "volume" in teil.columns
    volume = teil["volume"].astype(float) if hat_volumen else None
    return [
        _befund(
            teil, instrument, code, int(i), "kursniveau",
            close=float(close.iloc[i]),
            median_close=float(mitte),
            dekaden=float(dekaden.iloc[i]),
            volume=float(volume.iloc[i]) if hat_volumen else None,
            n_zeilen=int(schlecht.sum()),
        )
        for i in np.flatnonzero(schlecht.to_numpy())[:_MAX_JE_INVARIANTE]
    ]

#: Wieviele Befunde eine Invariante je Reihe höchstens meldet. `DIC` hat 149
#: kaputte Tage; sie alle einzeln zu melden macht den Bericht unlesbar, und die
#: Gesamtzahl steht in `n_zeilen`.
_MAX_JE_INVARIANTE = 5


ALLE: tuple[Invariante, ...] = (
    Invariante(
        name="kursniveau",
        beschreibung=(
            "Der Kurs weicht um mehr als zwei Zehnerpotenzen vom Median der "
            "eigenen Reihe ab — die Spalte enthält zwei Niveaus. Welches das "
            "richtige ist, entscheidet die Regel nicht."
        ),
        pruefen=_kursniveau,
        anlass=(
            "#333 — `DIC` springt im Juli 2010 täglich zwischen 0,67 und 30.000. "
            "Beide Kursspalten tragen denselben Fehler, deshalb ist der "
            "Einheiten-Detektor aus #324 dort blind."
        ),
    ),
)


def pruefe(
    teil: pd.DataFrame,
    instrument: str,
    code: str,
    *,
    invarianten: Sequence[Invariante] = ALLE,
) -> list[Befund]:
    """Alle Invarianten auf die Reihe **eines** Papiers.

    Eine Invariante, die wirft, darf die übrigen nicht mitnehmen: der Sinn der
    Liste ist, dass sie wächst, und eine neue Regel mit einem Randfall soll
    nicht den Load kippen. Sie wird geloggt und übersprungen.
    """
    aus: list[Befund] = []
    for inv in invarianten:
        try:
            aus.extend(inv.pruefen(teil, instrument, code))
        except Exception as exc:  # pragma: no cover - defekte Regel
            log.warning("Invariante %s scheiterte an %s: %s", inv.name, code, exc)
    return aus


__all__ = ["ALLE", "KURSNIVEAU_DEKADEN", "Invariante", "Pruefung", "pruefe"]
