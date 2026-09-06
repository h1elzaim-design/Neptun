"""Back-Adjustment — Roh-OHLCV + Corporate Actions → adjustierte Serie.

Warum überhaupt: back-adjustierte Kurse ändern sich RÜCKWIRKEND bei jeder neuen
Dividende/jedem Split. Wer adjustierte Kurse cacht, hat nach dem nächsten
Corporate-Action-Event stille Fehler. Profi-Standard ist deshalb: Roh-OHLCV +
`divCash`/`splitFactor` speichern (die ändern sich nie) und beim LESEN
adjustieren. Dieses Modul ist diese Adjustierung — rein, deterministisch, getestet.

Methode (Total-Return, CRSP-Stil, ankert an der letzten Zeile = Raw == Adjusted):
    Pro Tag t mit Aktion:
        ratio_t = (1 / splitFactor_t) * (1 - divCash_t / close_{t-1})
    Kumulativer Faktor für Tag i = Produkt aller ratio_t mit t > i.
    adj_price_i = price_i * cumFactor_i

Volumen wird nur durch Splits skaliert (mehr Aktien), nicht durch Dividenden.

**Vorbedingung für `volume`, seit #304 ausgesprochen:** die Spalte muss *roh*
hereinkommen, also die am Stichtag tatsächlich gehandelten Stücke. Das galt für
Tiingo und gilt für den EODHD-Bulk **nicht** — dort steht `volume` bereits auf
heutiger Stückzahl. Wer eine solche Reihe hierher gibt, bekommt `V · S²`
zurück. Der Bulk-Lesepfad reicht `volume` deshalb gar nicht erst herein
(`bulk_read._apply_actions`); diese Funktion bleibt reine Mathematik ohne
Provider-Wissen.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

#: Ab welchem Anteil des Vortagsschlusses eine behauptete Ausschüttung gross
#: genug ist, dass der Kurs am Ex-Tag darüber urteilen kann. Darunter
#: entscheidet die Tagesvolatilität und nicht die Ausschüttung — eine
#: 1-%-Dividende verschwindet im Rauschen.
GROSSE_AUSSCHUETTUNG = 0.10

#: Wieviel des behaupteten Rückgangs im Kurs **fehlen** darf, bevor der
#: Eintrag als widerlegt gilt.
#:
#: Gemessen am 2026-09-06 über `us_top500_liquid`, 2000–2023, 55.080
#: Dividendenereignisse: bei den **echten** grossen Ausschüttungen bleibt
#: höchstens 4,8 Prozentpunkte unerklärt (`DF` 2007; `AABA` 2019 mit 51,50 $
#: auf 70,80 $ liegt bei 0,3). Bei den widerlegten sind es mindestens 96.
#: Dazwischen liegt so viel Luft, dass die genaue Zahl nicht entscheidet —
#: 0,20 lässt der echten Seite das Vierfache ihres beobachteten Maximums.
#:
#: Nur **positiv**: ein stärkerer Rückgang als die Dividende ist an einem
#: schlechten Markttag völlig normal und kein Befund.
UNERKLAERT_MAX = 0.20

RAW_COLUMNS = ["open", "high", "low", "close", "volume"]
CORP_COLUMNS = ["divCash", "splitFactor"]
ALL_RAW_COLUMNS = RAW_COLUMNS + CORP_COLUMNS


class UnadjustableActionError(ValueError):
    """Ein Aktionseintrag ergibt keinen positiven Preisfaktor.

    Praktisch heisst das: die gemeldete Dividende ist grösser als der
    Vortagsschluss. ``1 - div/prev_close`` wird dann negativ, und das Produkt
    kippt **alle** früheren Kurse ins Minus.

    Das ist kein extremes, aber gültiges Ereignis, sondern ein defekter
    Eintrag. Nachgemessen am 2026-09-01 an ``WY``: EODHD führt zum 2010-07-20
    eine „Dividende" von 26,42 $ auf ein 16-Dollar-Papier — die REIT-Umwandlung
    von Weyerhaeuser, historisch überwiegend **in Aktien** geleistet. Der Kurs
    fiel an diesem Tag von 16,52 auf 15,94, also gar nicht; auch EODHDs eigener
    ``adjusted_close`` läuft glatt weiter. Die Formel korrigiert hier für einen
    Kurssturz, den es nie gab.

    Deshalb wird geworfen statt gerechnet: eine Reihe mit negativen Kursen ist
    schlimmer als keine. Renditen kehren das Vorzeichen um, Positionsgrössen
    aus ``capital / price`` werden negativ, und jede Kennzahl darüber ist
    bedeutungslos statt bloss ungenau — ohne dass irgendwo etwas rot wird.
    """

    def __init__(self, message: str, *, tage: Sequence[Any] = ()) -> None:
        super().__init__(message)
        #: Die verantwortlichen Tage — damit der Aufrufer sie nennen kann,
        #: statt nur „irgendwo in dieser Reihe".
        self.tage = list(tage)


def _reverse_cumprod_exclusive(series: pd.Series) -> pd.Series:
    """Für jede Position i: Produkt aller Werte mit Index > i (self exklusiv).

    Implementiert über Reverse → cumprod → shift(1) → Reverse zurück. Die
    jüngste Zeile bekommt 1.0 (keine späteren Aktionen)."""
    rev = series[::-1]
    cum = rev.cumprod().shift(1).fillna(1.0)
    return cum[::-1]


def adjust_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Roh-Frame (open/high/low/close/volume + divCash/splitFactor) → adjustiert.

    Erwartet einen DatetimeIndex. Gibt open/high/low/close/volume adjustiert
    zurück (gleicher Index). Idempotent gegenüber actionsfreien Reihen
    (kein Split/keine Dividende → Output == Roh-OHLCV).
    """
    if raw.empty:
        return raw[[c for c in RAW_COLUMNS if c in raw.columns]].copy()

    df = raw.sort_index()
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    split = df.get("splitFactor")
    split = pd.Series(1.0, index=df.index) if split is None else split.astype(float)
    split = split.replace(0.0, 1.0).fillna(1.0)

    # **Der Schutz war asymmetrisch.** Ein nicht-positiver Dividendenfaktor
    # wirft seit dem WY-Fall (siehe `UnadjustableActionError`), ein absurder
    # Split-Faktor lief ungeprueft durch — dabei kippt ein negativer Wert das
    # Vorzeichen der ganzen Reihe genauso, und die Null wurde stillschweigend zu
    # Eins. Eine Reihe mit negativen Kursen ist schlimmer als keine: Renditen
    # kehren sich um, Positionsgroessen aus `capital / price` werden negativ,
    # und jede Kennzahl darueber ist bedeutungslos statt bloss ungenau.
    unmoeglich = split <= 0.0
    if bool(unmoeglich.any()):
        tage = list(df.index[unmoeglich])
        raise UnadjustableActionError(
            f"Split-Faktor {float(split[unmoeglich].iloc[0])} am "
            f"{getattr(tage[0], 'date', lambda: tage[0])()}: ein Split teilt oder "
            f"legt zusammen, er dreht nicht das Vorzeichen. "
            f"({len(tage)} betroffene(r) Tag(e))",
            tage=tage,
        )

    div = df.get("divCash")
    div = pd.Series(0.0, index=df.index) if div is None else div.astype(float).fillna(0.0)

    # **Was der Kurs am Ex-Tag dazu sagt** (#312). Eine Barausschüttung von
    # x % des Kurses zeigt sich als Rückgang von ungefähr x %; eine erfundene
    # zeigt sich gar nicht:
    #
    #   WY   2010-07-20  behauptet -160 %   16,52 → 15,94  =  -3,5 %   Müll
    #   JNUG 2017-03-21  behauptet  -65 %    7,08 →  7,37  =  +4,1 %   Müll
    #   MIL  2014-07-24  behauptet  -96 %    7,82 →  7,68  =  -1,8 %   Müll
    #   AABA 2019-09-24  behauptet  -73 %   70,80 → 19,51  = -72,4 %   echt
    #
    # Der bestehende Guard oben fängt nur, wo das Vorzeichen kippt — gemessen
    # 32 Ereignisse, während 1.888 still durchliefen. Diese Prüfung fängt die
    # stillen: der Eintrag wird verworfen, **die Reihe bleibt**. Bei `JNUG`
    # ist ein einziger Eintrag von 2017 kaputt und die übrigen zwanzig Jahre
    # sind in Ordnung.
    #
    # Der Split muss heraus: an einem 2:1-Tag halbiert sich der Kurs ohne jede
    # Ausschüttung.
    quote = (div / prev_close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bewegung = ((close / split) / prev_close - 1.0).fillna(0.0)
    widerlegt = (quote > GROSSE_AUSSCHUETTUNG) & ((bewegung + quote) > UNERKLAERT_MAX)
    verworfen: list[Any] = []
    if bool(widerlegt.any()):
        verworfen = list(df.index[widerlegt])
        div = div.where(~widerlegt, 0.0)

    # Dividenden-Faktor der Aktion an Tag t (wirkt auf Tage < t). Erste Zeile hat
    # keinen Vortagsschluss → kein Faktor (1.0).
    div_factor = (1.0 - div / prev_close).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # **Ein nicht-positiver Faktor ist ein Datenfehler, kein Extremfall.**
    # Siehe `UnadjustableActionError`: eine Dividende über dem Vortagsschluss
    # kippt jede frühere Zeile ins Negative. Lieber keine Reihe als eine mit
    # negativen Kursen.
    #
    # **Warum das jetzt das Netz ist und nicht mehr die Regel.** Bis zum
    # 2026-09-06 stand dieser Guard vorn und warf bei `WY` die *ganze Reihe*
    # weg — zwanzig Jahre Kurse wegen eines Eintrags von 2010. Die Kursprüfung
    # oben entscheidet dieselben Fälle besser: sie verwirft den Eintrag und
    # behält die Reihe. Was bis hierher kommt, hat sie nicht beurteilen können
    # (kein Vortagsschluss, oder eine Ausschüttung unter der Grössenschwelle,
    # die den Faktor trotzdem kippt), und dann bleibt es beim alten Urteil.
    schlecht = div_factor <= 0.0
    if bool(schlecht.any()):
        tage = list(df.index[schlecht])
        beispiel = tage[0]
        raise UnadjustableActionError(
            f"Aktionseintrag am {getattr(beispiel, 'date', lambda: beispiel)()}: "
            f"Dividende {float(div[schlecht].iloc[0]):.4f} bei Vortagsschluss "
            f"{float(prev_close[schlecht].iloc[0]):.4f} ergibt den Preisfaktor "
            f"{float(div_factor[schlecht].iloc[0]):.4f}. Eine Dividende über dem "
            f"Vortagsschluss ist keine Bardividende — die Reihe wäre ab hier "
            f"rückwärts negativ. ({len(tage)} betroffene(r) Tag(e))",
            tage=tage,
        )

    day_ratio = (1.0 / split) * div_factor

    price_factor = _reverse_cumprod_exclusive(day_ratio)
    # Volumen: nur Splits skalieren die Stückzahl.
    vol_factor = _reverse_cumprod_exclusive(split)

    out = pd.DataFrame(index=df.index)
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            out[col] = (df[col].astype(float) * price_factor).astype(float)
    if "volume" in df.columns:
        out["volume"] = (df["volume"].astype(float) * vol_factor).astype(float)
    # Die verworfenen Aktionstage reisen am Ergebnis mit, nicht in einem Log.
    # Eine Annahme, die man später nicht mehr ablesen kann, ist keine.
    out.attrs["verworfene_aktionen"] = verworfen
    return out
