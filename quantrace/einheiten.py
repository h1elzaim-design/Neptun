"""Rohzeilen, die auf der falschen Stückzahl stehen (#324).

**Der Fall.** Drei aufeinanderfolgende Tage von ``AAPL`` führen die ganze
OHLC-Zeile in split-bereinigten Einheiten, während die Nachbartage roh sind::

    2003-01-06   close 14.9016   adjusted_close 0.2229   adj/close 0.014958
    2003-01-07   close  0.2652   adjusted_close 0.2221   adj/close 0.837481
    2003-01-08   close  0.2598   adjusted_close 0.2176   adj/close 0.837567
    2003-01-09   close  0.2621   adjusted_close 0.2195   adj/close 0.837467
    2003-01-10   close 14.7224   adjusted_close 0.2202   adj/close 0.014957

Der Faktor zwischen den Niveaus ist 56,2 — AAPLs kumulierte Splits seit 2003
sind ``2 · 7 · 4 = 56``. Als Rendite gelesen: **−98,2 %** und drei Tage später
**+5.453 %**, bei einem der liquidesten Namen im Bestand.

**Warum keine bestehende Prüfung das fängt.** Die Zeile ist *in sich*
plausibel: ``high > low``, alle Kurse positiv, keine Nullen. Falsch ist nur ihr
Verhältnis zu den Nachbarn, und danach sieht nichts.

## Der Detektor

``adjusted_close / close`` ist der kumulierte Aktionsfaktor. Zwischen zwei
Corporate Actions **muss** er konstant sein. Springt er an einem Tag ohne
Aktion, steht eine der beiden Spalten in falschen Einheiten.

Trennschärfe gemessen, nicht behauptet — über 15 grosse Namen und 87.675
Kurszeilen springt der Faktor an einem Tag *ohne* Aktion in 0,006 % der Fälle
um mehr als 1 %, an einem Tag *mit* Aktion in 1.349 von 1.351.

## Zwei Unterscheidungen, und beide waren nötig

**Welche Spalte?** Der Sprung sagt nicht, welche der beiden falsch steht.
Entschieden wird am Rohkurs::

    ARNA 2008-12-17   f 1,0 → 10,0     close  37,80 → 39,40   glatt
    AAPL 2003-01-07   f 0,015 → 0,837  close  14,90 →  0,27   Bruch
    J    2001-05-18   f 0,096 → 0,190  close 144,17 → 74,20   Bruch

Bei ``ARNA`` ist der **bereinigte** Kurs des Anbieters kaputt — vor 2008 nicht
auf den Reverse Split von 2017 korrigiert. Den benutzt der Backtest nicht, die
Bereinigung wird aus den Aktionsfeeds rekonstruiert. Ohne diese Unterscheidung
wären 2.127 einwandfreie Zeilen weggeflogen.

**Ausschlag oder Stufe?** Ein Ausschlag ist ein Stück, nach dem die Kurve auf
ihr vorheriges Niveau zurückkommt — dort sind die Zeilen kaputt. Eine Stufe
hält: dort ist die Rohzeile richtig und im Feed **fehlt eine Aktion**
(``MO`` 2002-10-15, die Kraft-Abspaltung). Wer beide gleich behandelt, wirft
bei der Stufe halbe Historien weg.

Über die Segment*länge* getrennt fällt ``IBM`` Januar 2003 auf die falsche
Seite: acht zusammenhängende Tage, nach denen die Kurve *nicht* zurückkehrt.
Acht ist kurz und trotzdem eine Stufe. Die Länge sagt nichts über die Ursache,
die Rückkehr schon.

Dieses Modul ist die **eine** Implementierung. ``scripts/unit_drift.py`` misst
damit die Ausbreitung, ``bulk_read._apply_actions`` verwirft damit die Zeilen —
zwei Fassungen wären zwei Wahrheiten, die auseinanderlaufen.
"""

from __future__ import annotations

import pandas as pd

#: Ab hier gilt ein Sprung der Faktorkurve als Befund und nicht als Rundung.
#: ``adjusted_close`` steht auf vier Nachkommastellen; bei einem Cent-Papier
#: ist das schon Promille. Gemessen (15 Namen, 87.675 Zeilen): oberhalb 1 %
#: liegen 0,006 % der aktionsfreien Tage, und die sind alle echte Befunde.
SPRUNG = 0.01

#: Ab hier gilt der **Rohkurs** als gebrochen. Grosszügiger als ``SPRUNG``,
#: weil ein echter Tagesverlust von 20 % vorkommt — die Einheitenfehler liegen
#: bei −98 % (``AAPL``) und −49 % (``J``), also weit darüber.
CLOSE_BRUCH = 0.30

#: Wie nah zwei Niveaus sein müssen, um als dasselbe zu gelten. Grosszügiger
#: als ``SPRUNG``, weil ein Ausschlag die Kurve nicht exakt zurückgeben muss —
#: dazwischen kann eine echte Aktion liegen.
GLEICH = 0.02

#: EODHDs Platzhalter für „kein Kurs ermittelbar" — kein Nullwert, sondern
#: eine konkrete Zahl, die wie ein echter Kurs aussieht. Steht in **beiden**
#: Kursspalten; `bulk_read.dollar_volume` kennt ihn seit #296.
#:
#: Hier ist er wichtig, weil er die Faktorkurve zerstört: `999999,9999 / 12,50`
#: ist ein Faktor von 80.000, und der sieht aus wie ein Einheitenfehler.
#: Gemessen am 2026-09-06 waren **516 von 2.943** Treffern in Wahrheit dieser
#: Platzhalter — die Rohkurse daneben können völlig in Ordnung sein.
EODHD_NULL_PRICE_SENTINEL = 999999.9999

AUSSCHLAG = "ausschlag"
STUFE = "stufe"


def faktorkurve(adjusted_close: pd.Series, close: pd.Series) -> pd.Series:
    """``adjusted_close / close``, mit ``NaN`` wo die Zahl nichts bedeutet.

    Die **eine** Stelle, an der die Kurve entsteht — sonst filtert eine der
    beiden Aufrufseiten den Platzhalter und die andere nicht.

    ``NaN`` statt eines Ersatzwerts: eine Zeile ohne Faktorauskunft soll keine
    Segmentgrenze setzen und in keinen Median eingehen. Fortgeschrieben wäre
    sie eine Behauptung, gelöscht ginge die Kurszeile mit verloren — und die
    ist womöglich völlig in Ordnung.
    """
    a = adjusted_close.astype(float)
    c = close.astype(float)
    brauchbar = (a > 0) & (c > 0) & (a != EODHD_NULL_PRICE_SENTINEL)
    return (a / c).where(brauchbar)


def klassifiziere(
    f: pd.Series, aktionstag: pd.Series, close: pd.Series | None = None
) -> pd.DataFrame:
    """Je Zeile: ``ausschlag``, ``stufe`` oder ``""``.

    ``f`` ist die Faktorkurve ``adjusted_close / close`` **einer** Reihe, nach
    Datum sortiert. ``aktionstag`` sagt je Zeile, ob an dem Tag ein Split oder
    eine Dividende im Feed steht. ``close`` ist der Rohkurs; ohne ihn wird nur
    die Faktorkurve geprüft, und dann kann nicht entschieden werden, **welche**
    Spalte falsch steht.
    """
    sprung = ((f / f.shift(1) - 1.0).abs() > SPRUNG).fillna(False)

    # Siehe Modulkopf, „Welche Spalte?" — ein Sprung ohne Bruch im Rohkurs
    # betrifft `adjusted_close`, und den benutzt der Backtest nicht.
    if close is not None:
        bruch = ((close / close.shift(1) - 1.0).abs() > CLOSE_BRUCH).fillna(False)
        sprung = sprung & bruch

    # **Jede** Niveauänderung setzt eine Grenze, auch die an einem Aktionstag.
    # Liesse man Aktionstage nicht trennen, spannte das Segment hinter einem
    # Ausschlag über spätere Splits hinweg und sein Median würde sinnlos —
    # `AAPL` reicht bis 2023 und enthält die Splits von 2005, 2014 und 2020.
    # Ein Segment muss ein Stück *konstanter* Kurve sein, sonst hat „Niveau"
    # keine Bedeutung. Erlaubt oder verdächtig entscheidet sich dann an der
    # Grenze, nicht an ihrer Existenz.
    segment = sprung.cumsum()
    laenge = segment.map(segment.value_counts())
    art = pd.Series("", index=f.index, dtype=object)
    if segment.max() == 0:
        return pd.DataFrame({"f": f, "segment": segment, "seg_laenge": laenge, "art": art})

    verdaechtig = sprung & ~aktionstag.astype(bool)
    # Median statt Mittel: ein einzelner Ausreisser **innerhalb** eines
    # Segments soll dessen Niveau nicht verschieben.
    m = f.groupby(segment).median()
    ids = list(m.index)
    startet_verdaechtig = verdaechtig.groupby(segment).first()

    def gleich(a: float, b: float) -> bool:
        return abs(a / b - 1.0) <= GLEICH if b else False

    urteil: dict[int, str] = {}
    i = 1
    while i < len(ids):
        if not bool(startet_verdaechtig.get(ids[i], False)):
            i += 1
            continue
        vorher, dieses = m[ids[i - 1]], m[ids[i]]
        nachher = m[ids[i + 1]] if i + 1 < len(ids) else None
        if nachher is not None and gleich(nachher, vorher) and not gleich(dieses, vorher):
            # Die Kurve kehrt zurück: dieses Segment ist der Ausschlag, das
            # nächste die Rückkehr und damit gesund. Es zu überspringen ist der
            # Punkt — sonst wäre die Erholung selbst ein Befund.
            urteil[ids[i]] = AUSSCHLAG
            i += 2
            continue
        urteil[ids[i]] = STUFE
        i += 1

    for seg, wert in urteil.items():
        art[segment == seg] = wert
    return pd.DataFrame({"f": f, "segment": segment, "seg_laenge": laenge, "art": art})


__all__ = [
    "AUSSCHLAG",
    "CLOSE_BRUCH",
    "EODHD_NULL_PRICE_SENTINEL",
    "GLEICH",
    "SPRUNG",
    "STUFE",
    "faktorkurve",
    "klassifiziere",
]
