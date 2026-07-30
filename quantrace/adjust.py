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
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS = ["open", "high", "low", "close", "volume"]
CORP_COLUMNS = ["divCash", "splitFactor"]
ALL_RAW_COLUMNS = RAW_COLUMNS + CORP_COLUMNS


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

    div = df.get("divCash")
    div = pd.Series(0.0, index=df.index) if div is None else div.astype(float).fillna(0.0)

    # Dividenden-Faktor der Aktion an Tag t (wirkt auf Tage < t). Erste Zeile hat
    # keinen Vortagsschluss → kein Faktor (1.0).
    div_factor = (1.0 - div / prev_close).replace([np.inf, -np.inf], np.nan).fillna(1.0)
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
    return out
