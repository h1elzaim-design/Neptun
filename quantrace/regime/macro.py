"""Makro-Features für die Regime-Erkennung — opt-in, Default aus.

Die Regime-Engine läuft bis heute ausschließlich auf Kursen: Trend und
realisierte Vola eines Benchmarks. Zinsstruktur und Credit-Spreads sind aber die
klassischen Regime-Treiber — eine inverse Zinskurve und ausweitende Spreads
sagen mehr über den Zustand des Marktes als die Vola des S&P allein.

**Warum das trotzdem abgeschaltet bleibt.** Issue #231 hat gemessen, was
passiert, wenn man diesem HMM mehr Freiheit gibt: Multi-Restart verbesserte die
Trainings-Likelihood in 21 von 36 Fällen und verschlechterte die
Out-of-Sample-Likelihood in 16 von 30. Mehr Dimensionen sind dieselbe Art von
Geschenk. Ein Gauß-HMM mit drei Zuständen schätzt pro zusätzlicher Dimension
eine ganze Zeile der Kovarianzmatrix; von zwei auf vier Features wächst die
Parameterzahl deutlich schneller als die Information.

Deshalb: gebaut, getestet, **standardmäßig aus**. Freischalten heißt, die
Messung aus #231 zu wiederholen — mit echten Reihen und einem begründeten
Auswahlmaß, nicht mit Trainings-Likelihood.

**Warum Spreads und keine Zinsniveaus.** Ein Gauß-HMM auf `DGS10` würde „hohe
Zinsen" zu einem eigenen Regime machen und es nie wieder verlassen: die Reihe
läuft von 15 % (1981) auf 0 % (2020) und ist über Jahrzehnte nicht stationär.
Spreads sind mittelwertrückkehrend und bleiben über die Zeit vergleichbar —
eine inverse Kurve 1989 bedeutet dasselbe wie eine inverse Kurve 2019.

**Ausrichtung nur vorwärts.** FRED-Reihen haben Lücken (Feiertage, verspätete
Veröffentlichungen). Rückwärts zu füllen wäre Look-ahead — der Wert von morgen
stünde heute. `align_macro` füllt deshalb ausschließlich vorwärts.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)

#: Bewusst klein gehalten. Zwei Spreads statt zehn Reihen: beide sind
#: mittelwertrückkehrend, beide messen etwas, das die Kursfeatures nicht sehen,
#: und beide zusammen verdoppeln die Dimensionszahl schon von 2 auf 4.
MACRO_SERIES: dict[str, str] = {
    "T10Y2Y": "term_spread",
    "BAMLH0A0HYM2": "credit_spread",
}


def align_macro(macro: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Makroreihen auf den Kursindex legen — ausschließlich vorwärts gefüllt.

    Ein Feiertag oder eine verspätete Veröffentlichung darf den Wert des
    *nächsten* Tages nicht rückwirkend sichtbar machen. Was am Stichtag noch
    nicht da war, bleibt NaN und fällt später beim `dropna` heraus.
    """
    target = pd.DatetimeIndex(index)
    if macro.empty:
        return pd.DataFrame(index=target)

    src = macro.copy()
    src.index = pd.DatetimeIndex(src.index)
    src = src.sort_index()

    combined = src.reindex(src.index.union(target)).ffill()
    return combined.reindex(target)


def fetch_macro_features(
    start: date,
    end: date,
    *,
    series: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Die kuratierten Makroreihen von FRED, umbenannt auf sprechende Spalten.

    Braucht ``FRED_API_KEY``. Wirft nicht bei Netzproblemen — Makro ist ein
    Zusatz, kein kritischer Pfad; ein leerer Frame lässt die Regime-Erkennung
    auf den Kursfeatures weiterlaufen, statt den Lauf zu kippen.
    """
    series = series or MACRO_SERIES
    try:
        from quantrace.providers import fred

        raw = fred.fetch_many(list(series), start, end)
    except Exception as exc:  # noqa: BLE001 — bewusst breit, siehe Docstring
        log.warning("Makro-Features übersprungen (%s) — Regime läuft auf Kursen.", exc)
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    present = {k: v for k, v in series.items() if k in raw.columns}
    missing = set(series) - set(present)
    if missing:
        log.warning("FRED lieferte %s nicht — Feature fehlt.", ", ".join(sorted(missing)))
    return raw[list(present)].rename(columns=present)
