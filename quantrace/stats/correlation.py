"""Asset-Korrelationen (#178) — Matrix + hierarchische Cluster-Sortierung.

Pure, from scratch (kein scipy, konsistent zum Rest von quantrace/stats):
Average-Linkage-Clustering auf der Distanz d = 1 − |ρ|. Die Blattreihenfolge
des Dendrogramms sortiert die Matrix so, dass korrelierte Blöcke nebeneinander
liegen — Cluster werden als Blöcke sichtbar statt über die Matrix verstreut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Log-Returns einer Close-Matrix (Index=Zeit, Spalten=Symbol)."""
    return np.log(closes / closes.shift(1)).iloc[1:]


def correlation_matrix(returns: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Pearson-Korrelation, optional nur über die letzten `window` Zeilen.

    Zeilen mit irgendeinem NaN werden verworfen (gemeinsame Handelstage) —
    paarweise Korrelation über verschiedene Zeiträume wäre still inkonsistent.
    """
    clean = returns.dropna(how="any")
    if window is not None and window > 0:
        clean = clean.iloc[-window:]
    if len(clean) < 3:
        raise ValueError(
            f"Zu wenige gemeinsame Beobachtungen für eine Korrelation ({len(clean)} < 3)."
        )
    return clean.corr()


def rolling_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
    """Rollierendes ρ **eines Paares** über die gemeinsamen Handelstage.

    Erwartet genau zwei Spalten. Der Grund für die Enge: `DataFrame.rolling`
    liefert für n Spalten eine MultiIndex-Matrix je Zeitpunkt, und die daraus
    wieder auf ein Paar herunterzuschneiden lädt zu genau dem Fehler ein, den
    `correlation_matrix` vermeidet — paarweise Korrelationen über verschiedene
    Zeiträume. Hier werden **erst** die gemeinsamen Tage gebildet, **dann**
    gerollt.

    Der Rückgabewert enthält bewusst die führenden NaN nicht: die ersten
    `window - 1` Tage haben kein volles Fenster, und eine Kurve, die dort bei 0
    beginnt, behauptete Unkorreliertheit, wo schlicht nichts gemessen wurde.
    """
    if returns.shape[1] != 2:
        raise ValueError(f"Rolling-Korrelation braucht genau 2 Spalten, bekam {returns.shape[1]}.")
    if window < 3:
        raise ValueError(f"Fenster muss mindestens 3 Beobachtungen umfassen (war {window}).")

    clean = returns.dropna(how="any")
    if len(clean) < window:
        raise ValueError(
            f"Zu wenige gemeinsame Beobachtungen für ein Fenster von {window} "
            f"({len(clean)} verfügbar)."
        )

    a, b = clean.columns
    series = clean[a].rolling(window).corr(clean[b])
    # Eine konstante Reihe im Fenster ergibt 0/0 → ±inf oder NaN. Beides ist
    # „nicht messbar", nicht „0" — und muss aus der Kurve fallen, statt sie
    # nach oben oder unten zu reißen.
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def cluster_order(corr: pd.DataFrame) -> list[str]:
    """Blattreihenfolge eines Average-Linkage-Dendrogramms auf d = 1 − |ρ|."""
    labels = list(corr.columns)
    n = len(labels)
    if n <= 2:
        return labels

    dist = 1.0 - corr.abs().to_numpy()
    # Cluster als (Mitglieds-Indizes, Blattliste-in-Reihenfolge)
    clusters: dict[int, tuple[list[int], list[int]]] = {
        i: ([i], [i]) for i in range(n)
    }
    next_id = n

    def avg_dist(a: list[int], b: list[int]) -> float:
        return float(np.mean([dist[i, j] for i in a for j in b]))

    while len(clusters) > 1:
        ids = sorted(clusters)
        best: tuple[float, int, int] | None = None
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                d = avg_dist(clusters[ids[x]][0], clusters[ids[y]][0])
                if best is None or d < best[0]:
                    best = (d, ids[x], ids[y])
        assert best is not None
        _, a, b = best
        members = clusters[a][0] + clusters[b][0]
        leaves = clusters[a][1] + clusters[b][1]
        del clusters[a], clusters[b]
        clusters[next_id] = (members, leaves)
        next_id += 1

    (_, leaf_order) = next(iter(clusters.values()))
    return [labels[i] for i in leaf_order]
