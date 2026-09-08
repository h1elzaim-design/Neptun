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

    **Was der erste volle Lauf ergab (2026-09-08, `us_top500_liquid`,
    2000–2023, 2.008 Instrumente).** Nicht zwei Papiere, sondern **240** — und
    sie zerfallen in zwei Fälle mit verschiedener Behandlung:

    =====================  ========  =====================================
    ein Abschnitt             87     zwei Instrumente **hintereinander**
    zwei bis fünf             80     Mischform
    mehr als fünf             73     Zeilen wechseln sich ab
    =====================  ========  =====================================

    Von den 87 liegt der Abschnitt bei **63** am Rand der Reihe: ``PGD`` ist
    bis 2012 ein Papier und danach ein anderes, ``MABANEE`` (Kuwait) bis 2009
    eines und danach eines — beide unter *einem* Segmentschlüssel
    ``code.*.US.s1``. Das ist kein Zeilenfehler; das ist die Frage nach der
    Identität, und die Segmentierung hat dort nicht gegriffen.

    **Deshalb ``bloecke`` und ``am_rand`` in den Belegen.** Ohne sie sähen im
    Bericht ``DIC`` (160 Abschnitte, täglich alternierend, kaputte Zeilen) und
    ``MABANEE`` (ein Abschnitt am Rand, ein anderes Papier) gleich aus — und
    ``verwerfen`` wäre im zweiten Fall die falsche Antwort: es löschte die
    halbe Historie eines Papiers, das nichts falsch macht.
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

    # **Wie die fernen Zeilen liegen, sagt was sie sind.** Ein einzelner
    # zusammenhängender Abschnitt heisst: bis hier das eine Papier, danach ein
    # anderes — die Spalte trägt zwei Instrumente **hintereinander**. Viele
    # Abschnitte heissen: die Zeilen wechseln sich ab, und dann ist die eine
    # Hälfte kaputt. Die Behandlung ist verschieden, und ohne diese Zahl steht
    # sie nicht im Bericht.
    marke = schlecht.to_numpy()
    bloecke = int((np.diff(np.concatenate(([0], marke.astype(int)))) == 1).sum())
    # Liegt der Abschnitt am Rand der Reihe, ist der Bruch ein Wechsel der
    # Identität und keine Lücke in der Mitte — genau der Fall, für den es die
    # Segmentierung gibt (`resolve.py`, ADR-013) und in dem sie nicht griff.
    am_rand = bool(marke[0] or marke[-1])

    # **Die Grenzen des betroffenen Bereichs**, damit ein Abschnitt im
    # Register (`befunde.Entscheidung.von`/`bis`) sich abschreiben lässt. Ohne
    # sie müsste jemand die Reihe aufmachen, um das Fenster zu bestimmen — und
    # dann schaut niemand hin, dieselbe Begründung wie für die übrigen Belege.
    treffer = np.flatnonzero(marke)
    tage = pd.to_datetime(teil["date"])
    fern_von = tage.iloc[int(treffer[0])].date()
    fern_bis = tage.iloc[int(treffer[-1])].date()

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
            bloecke=bloecke,
            am_rand=am_rand,
            fern_von=fern_von.isoformat(),
            fern_bis=fern_bis.isoformat(),
        )
        for i in treffer
    ]

#: Wieviele Befunde eine Invariante je Reihe höchstens **meldet**. `DIC` hat
#: 1.666 kaputte Tage; sie alle einzeln in den Bericht zu schreiben macht ihn
#: unlesbar, und die Gesamtzahl steht in `n_zeilen`.
#:
#: **Der Deckel sitzt in `pruefe`, nicht in der Invariante** — seit dem
#: 2026-09-08, und der Grund ist die Anwendung: eine Entscheidung über einen
#: Abschnitt (`befunde.Entscheidung.von`/`bis`) muss *jede* betroffene Zeile
#: treffen, nicht die ersten fünf. Eine Invariante findet also alles; wieviel
#: davon in einen Bericht geht, ist eine Frage der Darstellung und gehört
#: nicht in die Regel.
MAX_JE_INVARIANTE = 5


#: Ab wieviel Tagen mit identischem Kurs eine Strecke gemeldet wird.
#:
#: **Gemessen, nicht gesetzt** (2026-09-08, `us_top500_liquid`): Strecken von
#: 5 bis 19 Tagen tragen an 40,7 % der Tage Umsatz — das ist Handel. Ab 20
#: Tagen sind es 13,7 %, ab 100 nur noch 3,9 %. Die Zwanzig ist der Punkt, ab
#: dem die Mehrheit der Strecken tot ist; darunter wäre die Meldung Rauschen.
#:
#: Sie ist **kein** Kriterium für Verwerfen — das entscheidet der Umsatz
#: (`bulk_read._fortgeschriebene_als_luecke`). Hier geht es nur darum, dass
#: auch die Fälle sichtbar werden, in denen das Volumen als Nachweis nicht
#: taugt: `PTT` steht 2.467 Tage auf 5,12, und drei widersprüchliche Nulltage
#: reichen, damit der Lesepfad die Finger davon lässt.
EINGEFROREN_TAGE = 20


def _eingefroren(teil: pd.DataFrame, instrument: str, code: str) -> list[Befund]:
    """Ein Kurs, der wochenlang steht, ist meist keine Beobachtung (#333).

    **Der Fall.** ``LDG`` steht nach dem Delisting von Longs Drug Stores sieben
    Jahre auf 71,51, ``PTT`` 2.467 Tage auf 5,12. Jede Zeile ist für sich
    plausibel, die Nachbarn widersprechen sich nicht, der Aktionsfaktor steht
    still — alle anderen Prüfungen sind dort blind. Für einen Backtest sind es
    Jahre mit Rendite null und Volatilität null.

    **Warum das trotzdem nur eine Meldung ist.** Ein illiquides Papier hält
    seinen Kurs legitim über Wochen: ``UDS`` steht 263 Tage auf 6,02 und
    handelt an 30 % davon. Verworfen wird deshalb nur, wo der Umsatz es belegt
    — und das entscheidet der Lesepfad, nicht diese Regel. Was hier gemeldet
    wird, ist der Rest: Strecken, bei denen das Volumen als Nachweis nicht
    taugt, weil es anderswo in der Reihe einer Kursbewegung widerspricht.

    ``tage_mit_umsatz`` steht in den Belegen: null heisst „nie gehandelt",
    ein nennenswerter Anteil heisst „echt und bloss illiquide".
    """
    if "close" not in teil.columns or len(teil) < EINGEFROREN_TAGE:
        return []
    close = pd.to_numeric(teil["close"], errors="coerce")
    if not bool(close.notna().any()):
        return []

    # Zusammenhängende Abschnitte mit identischem Kurs.
    gruppe = (close != close.shift(1)).cumsum()
    laenge = gruppe.map(gruppe.value_counts())
    lang = (laenge >= EINGEFROREN_TAGE) & close.notna()
    if not bool(lang.any()):
        return []

    hat_volumen = "volume" in teil.columns
    volume = pd.to_numeric(teil["volume"], errors="coerce") if hat_volumen else None

    aus: list[Befund] = []
    for g in gruppe[lang].unique():
        maske = (gruppe == g) & close.notna()
        pos = int(np.flatnonzero(maske.to_numpy())[0])
        tage = int(maske.sum())
        mit_umsatz = int((volume[maske] > 0).sum()) if hat_volumen else None
        # **Nur die toten Strecken.** Wo auch nur an einem Tag gehandelt
        # wurde, ist der stehende Kurs eine Beobachtung — `UDS` hält seinen
        # Kurs 263 Tage und handelt an 80 davon. Das zu melden wäre Rauschen,
        # und der nächste echte Befund ginge darin unter (siehe
        # `data/corrections/README.md`).
        #
        # Ohne Volumenspalte gibt es keine Auskunft, also auch keine Meldung:
        # „unbekannt" ist kein Befund.
        if mit_umsatz is None or mit_umsatz > 0:
            continue
        aus.append(
            _befund(
                teil, instrument, code, pos, "eingefroren",
                close=float(close.iloc[pos]),
                tage=tage,
                tage_mit_umsatz=mit_umsatz,
                fern_von=pd.Timestamp(teil["date"].iloc[pos]).date().isoformat(),
                fern_bis=pd.Timestamp(
                    teil["date"].iloc[int(np.flatnonzero(maske.to_numpy())[-1])]
                ).date().isoformat(),
                n_zeilen=tage,
            )
        )
    return aus


ALLE: tuple[Invariante, ...] = (
    Invariante(
        name="kursniveau",
        beschreibung=(
            "Der Kurs weicht um mehr als zwei Zehnerpotenzen vom Median der "
            "eigenen Reihe ab — die Spalte enthält zwei Niveaus. Welches das "
            "richtige ist, entscheidet die Regel nicht. `bloecke` sagt, ob die "
            "Niveaus aufeinanderfolgen (ein Abschnitt: zwei Instrumente in "
            "einer Spalte) oder sich abwechseln (viele: kaputte Zeilen); "
            "`am_rand`, ob der Abschnitt an einem Ende der Reihe liegt."
        ),
        pruefen=_kursniveau,
        anlass=(
            "#333 — `DIC` springt im Juli 2010 täglich zwischen 0,67 und 30.000. "
            "Beide Kursspalten tragen denselben Fehler, deshalb ist der "
            "Einheiten-Detektor aus #324 dort blind."
        ),
    ),
    Invariante(
        name="eingefroren",
        beschreibung=(
            "Der Kurs steht über 20 Handelstage oder länger still. Meist ein "
            "fortgeschriebener Wert nach dem Delisting, manchmal ein echtes "
            "illiquides Papier — `tage_mit_umsatz` sagt welches."
        ),
        pruefen=_eingefroren,
        anlass=(
            "#333 — `LDG` steht nach dem Delisting 2008 sieben Jahre auf 71,51, "
            "`PTT` 2.467 Tage auf 5,12. Der Lesepfad verwirft solche Zeilen nur, "
            "wo der Umsatz es belegt; gemeldet gehören auch die übrigen."
        ),
    ),
)


def pruefe(
    teil: pd.DataFrame,
    instrument: str,
    code: str,
    *,
    invarianten: Sequence[Invariante] = ALLE,
    je_invariante: int | None = MAX_JE_INVARIANTE,
) -> list[Befund]:
    """Alle Invarianten auf die Reihe **eines** Papiers.

    Eine Invariante, die wirft, darf die übrigen nicht mitnehmen: der Sinn der
    Liste ist, dass sie wächst, und eine neue Regel mit einem Randfall soll
    nicht den Load kippen. Sie wird geloggt und übersprungen.

    ``je_invariante`` deckelt, **wieviel gemeldet wird** — die Vorgabe ist der
    Bericht, ``None`` ist alles. Wer eine Entscheidung anwendet, braucht
    alles: ein Abschnitt im Register (`befunde.Entscheidung`) trifft jede
    betroffene Zeile, nicht die ersten fünf.
    """
    aus: list[Befund] = []
    for inv in invarianten:
        try:
            gefunden = inv.pruefen(teil, instrument, code)
        except Exception as exc:  # pragma: no cover - defekte Regel
            log.warning("Invariante %s scheiterte an %s: %s", inv.name, code, exc)
            continue
        aus.extend(gefunden if je_invariante is None else gefunden[:je_invariante])
    return aus


__all__ = [
    "ALLE",
    "KURSNIVEAU_DEKADEN",
    "MAX_JE_INVARIANTE",
    "Invariante",
    "Pruefung",
    "pruefe",
]
