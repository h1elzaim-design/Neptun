"""Proper Scoring Rules und Kalibrierung von Wahrscheinlichkeitsprognosen.

Eine Scoring Rule ist *proper*, wenn die ehrliche Wahrscheinlichkeit den
erwarteten Score optimiert. Wer sie als Massstab nimmt, belohnt keine Prognose,
die mutiger klingt, als sie ist.

Quellen
-------
- Brier (1950), "Verification of Forecasts Expressed in Terms of Probability"
- Murphy (1973), "A New Vector Partition of the Probability Score"
- Cox (1958), "Two Further Applications of a Model for Binary Regression"
- Gneiting & Raftery (2007), "Strictly Proper Scoring Rules, Prediction, and
  Estimation"

Abhängige Beobachtungen
-----------------------
Prognosen zum selben Ereignis sind nicht unabhängig: von 150 Golfern landen in
den Top 20 ungefähr 20, egal wie gut der Markt ist. Ein Intervall, das jede
Prognose als eigene Beobachtung zählt, ist deshalb zu schmal.
:func:`cluster_bootstrap` zieht darum ganze Cluster.

Alle Masse nehmen optionale Gewichte. Ein Cluster-Bootstrap ist damit nur ein
Satz von Gewichten: dieselbe Rechnung, keine kopierten Datenzeilen.

"Nicht definiert" ist kein Wert
-------------------------------
Haben alle Beobachtungen denselben Ausgang, ist der Referenz-Brier null und der
Skill nicht definiert. Die Funktion gibt dann ``None`` zurück, nicht ``0``: ein
Skill von null hiesse "so gut wie die Basisrate", und das ist hier unbekannt.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

#: Grenze für den Log-Score. Ein Preis von exakt 0 oder 1 ergäbe sonst
#: ``-inf``, und ein einziger Fall bestimmte das ganze Mittel.
LOG_CLIP = 1e-3


def _als_arrays(
    p: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    gewichte: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eingaben prüfen. Ungültiges wirft, statt still herauszufallen."""
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if p_arr.ndim != 1 or p_arr.shape != y_arr.shape:
        raise ValueError("p und y müssen eindimensional und gleich lang sein")
    if p_arr.size == 0:
        raise ValueError("keine Beobachtungen")
    if np.isnan(p_arr).any() or np.isnan(y_arr).any():
        raise ValueError("p oder y enthält NaN")
    if ((p_arr < 0) | (p_arr > 1)).any():
        raise ValueError("p muss in [0, 1] liegen")
    if not np.isin(y_arr, (0.0, 1.0)).all():
        raise ValueError("y muss 0 oder 1 sein")
    if gewichte is None:
        return p_arr, y_arr, np.ones_like(p_arr)
    w = np.asarray(gewichte, dtype=float)
    if w.shape != p_arr.shape or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Gewichte passen nicht zu p oder summieren nicht auf > 0")
    return p_arr, y_arr, w


def brier_score(p, y, gewichte: np.ndarray | None = None) -> float:
    """``BS = (1/N) Σ (p_i − y_i)²``. Kleiner ist besser, 0 ist perfekt."""
    p_arr, y_arr, w = _als_arrays(p, y, gewichte)
    return float(np.average((p_arr - y_arr) ** 2, weights=w))


def log_score(p, y, gewichte: np.ndarray | None = None, *, clip: float = LOG_CLIP) -> float:
    """``LS = (1/N) Σ [y·ln p + (1−y)·ln(1−p)]``. Grösser ist besser, 0 ist perfekt."""
    p_arr, y_arr, w = _als_arrays(p, y, gewichte)
    q = np.clip(p_arr, clip, 1.0 - clip)
    return float(np.average(y_arr * np.log(q) + (1.0 - y_arr) * np.log(1.0 - q), weights=w))


def brier_skill(
    p, y, gewichte: np.ndarray | None = None, *, referenz: float | None = None
) -> float | None:
    """``BSS = 1 − BS / BS_ref``. Ohne ``referenz`` ist das die Basisrate.

    ``None``, wenn der Referenz-Brier null ist: dann ist Skill nicht definiert.
    Die Basisrate stammt aus derselben Stichprobe und ist damit eine Referenz, die
    im Vorhinein niemand kennen konnte. Der Skill fällt dadurch eher zu niedrig aus.
    """
    p_arr, y_arr, w = _als_arrays(p, y, gewichte)
    if referenz is None:
        rate = float(np.average(y_arr, weights=w))
        bs_ref = rate * (1.0 - rate)
    else:
        bs_ref = brier_score(np.full_like(p_arr, referenz), y_arr, w)
    if bs_ref <= 0.0:
        return None
    return 1.0 - brier_score(p_arr, y_arr, w) / bs_ref


@dataclass(frozen=True, slots=True)
class MurphyZerlegung:
    """``BS = reliability − resolution + uncertainty + rest``.

    ``rest`` ist der Fehler durch das Klassieren. Er ist null, wenn alle Prognosen
    einer Klasse gleich sind, und klein, wenn die Klassen eng sind.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    rest: float


def murphy_zerlegung(
    p, y, grenzen: Sequence[float], gewichte: np.ndarray | None = None
) -> MurphyZerlegung:
    """Zerlegt den Brier-Score über die Klassen ``grenzen``."""
    p_arr, y_arr, w = _als_arrays(p, y, gewichte)
    klasse = _klassen(p_arr, grenzen)
    gesamt = w.sum()
    rate = float(np.average(y_arr, weights=w))
    rel = res = 0.0
    for k in np.unique(klasse):
        maske = klasse == k
        gew = w[maske].sum()
        if gew <= 0:
            continue
        p_k = float(np.average(p_arr[maske], weights=w[maske]))
        y_k = float(np.average(y_arr[maske], weights=w[maske]))
        rel += gew * (p_k - y_k) ** 2
        res += gew * (y_k - rate) ** 2
    rel /= gesamt
    res /= gesamt
    unc = rate * (1.0 - rate)
    bs = brier_score(p_arr, y_arr, w)
    return MurphyZerlegung(bs, rel, res, unc, bs - (rel - res + unc))


def _klassen(p: np.ndarray, grenzen: Sequence[float]) -> np.ndarray:
    """Klassenindex je Prognose: ``[a, b)``, die letzte Klasse schliesst ``b`` ein."""
    g = np.asarray(grenzen, dtype=float)
    if g.size < 2 or (np.diff(g) <= 0).any():
        raise ValueError("grenzen müssen aufsteigen und mindestens zwei Werte haben")
    if p.min() < g[0] or p.max() > g[-1]:
        raise ValueError("eine Prognose liegt ausserhalb der Grenzen")
    return np.minimum(np.searchsorted(g, p, side="right") - 1, g.size - 2)


@dataclass(frozen=True, slots=True)
class CoxKalibrierung:
    """``P(Y=1) = logistic(alpha + beta · logit(p))``.

    ``beta > 1``: die Prognosen sind zur Mitte hin gestaucht, die Wahrheit ist
    extremer (Favourite-Longshot-Bias). ``beta < 1``: sie sind zu extrem.
    Perfekt kalibriert ist ``alpha = 0`` und ``beta = 1``.

    ``konvergiert`` ist falsch bei (fast) perfekter Trennung, wenn die Schätzer
    nach unendlich laufen. Die Werte sind dann ``nan``, nicht eine grosse Zahl.
    """

    alpha: float
    beta: float
    konvergiert: bool


def _logit(p: np.ndarray, clip: float = LOG_CLIP) -> np.ndarray:
    q = np.clip(p, clip, 1.0 - clip)
    return np.log(q / (1.0 - q))


def cox_kalibrierung(
    p,
    y,
    gewichte: np.ndarray | None = None,
    *,
    max_iter: int = 50,
    toleranz: float = 1e-8,
    max_betrag: float = 50.0,
) -> CoxKalibrierung:
    """Logistische Regression von ``y`` auf ``logit(p)`` per IRLS."""
    p_arr, y_arr, w = _als_arrays(p, y, gewichte)
    x = np.column_stack((np.ones_like(p_arr), _logit(p_arr)))
    b = np.zeros(2)
    for _ in range(max_iter):
        eta = x @ b
        mu = 1.0 / (1.0 + np.exp(-eta))
        v = np.clip(mu * (1.0 - mu), 1e-10, None)
        gradient = x.T @ (w * (y_arr - mu))
        hesse = (x.T * (w * v)) @ x
        try:
            schritt = np.linalg.solve(hesse, gradient)
        except np.linalg.LinAlgError:
            return CoxKalibrierung(float("nan"), float("nan"), False)
        b = b + schritt
        if np.abs(b).max() > max_betrag:
            return CoxKalibrierung(float("nan"), float("nan"), False)
        if np.abs(schritt).max() < toleranz:
            return CoxKalibrierung(float(b[0]), float(b[1]), True)
    return CoxKalibrierung(float("nan"), float("nan"), False)


@dataclass(frozen=True, slots=True)
class Intervall:
    """Punktschätzer mit Perzentil-Intervall aus einem Cluster-Bootstrap.

    ``n_ungueltig`` zählt Ziehungen, in denen die Statistik nicht bestimmbar war.
    Die Grenzen stammen nur aus den gültigen, und bei vielen ungültigen ist das
    Intervall entsprechend wenig wert.
    """

    punkt: float
    unten: float
    oben: float
    n_cluster: int
    n_ziehungen: int
    n_ungueltig: int

    def enthaelt(self, wert: float) -> bool:
        return self.unten <= wert <= self.oben


def cluster_bootstrap(
    statistik: Callable[[np.ndarray], float],
    cluster: Sequence | np.ndarray,
    *,
    n_ziehungen: int = 2000,
    konfidenz: float = 0.95,
    seed: int = 7,
) -> Intervall:
    """Bootstrap über ganze Cluster.

    ``statistik`` bekommt je Beobachtung ein Gewicht und gibt eine Zahl zurück,
    oder ``nan``, wenn sie in dieser Ziehung nicht bestimmbar ist.
    """
    if not 0.0 < konfidenz < 1.0:
        raise ValueError("konfidenz muss in (0, 1) liegen")
    _, codes = np.unique(np.asarray(cluster), return_inverse=True)
    n_cluster = int(codes.max()) + 1 if codes.size else 0
    if n_cluster == 0:
        raise ValueError("keine Cluster")
    punkt = float(statistik(np.ones(codes.size, dtype=float)))
    rng = np.random.default_rng(seed)
    gleich = np.full(n_cluster, 1.0 / n_cluster)
    werte = np.empty(n_ziehungen, dtype=float)
    for i in range(n_ziehungen):
        anzahl = rng.multinomial(n_cluster, gleich)
        werte[i] = statistik(anzahl[codes].astype(float))
    gueltig = werte[~np.isnan(werte)]
    n_ungueltig = int(n_ziehungen - gueltig.size)
    if gueltig.size == 0:
        return Intervall(punkt, float("nan"), float("nan"), n_cluster, n_ziehungen, n_ungueltig)
    rand = (1.0 - konfidenz) / 2.0
    unten, oben = np.quantile(gueltig, [rand, 1.0 - rand])
    return Intervall(punkt, float(unten), float(oben), n_cluster, n_ziehungen, n_ungueltig)


def _nan_falls_leer(gewichte: np.ndarray) -> bool:
    return float(gewichte.sum()) <= 0.0


@dataclass(frozen=True, slots=True)
class Klassenzeile:
    """Eine Zeile der Kalibrierungstabelle."""

    von: float
    bis: float
    n: int
    n_cluster: int
    preis_mittel: float
    haeufigkeit: float
    #: ``preis − häufigkeit``. Positiv heisst: die Prognose war zu hoch.
    abweichung: Intervall


def kalibrierungstabelle(
    p,
    y,
    cluster: Sequence | np.ndarray,
    grenzen: Sequence[float],
    *,
    n_ziehungen: int = 2000,
    seed: int = 7,
) -> list[Klassenzeile]:
    """Prognose gegen Häufigkeit je Klasse, das Intervall über Cluster gezogen.

    Klassen ohne Beobachtung fehlen in der Liste; sie sind nicht null.
    """
    p_arr, y_arr, _ = _als_arrays(p, y, None)
    cluster_arr = np.asarray(cluster)
    if cluster_arr.shape != p_arr.shape:
        raise ValueError("cluster passt nicht zu p")
    klasse = _klassen(p_arr, grenzen)
    zeilen: list[Klassenzeile] = []
    for k in np.unique(klasse):
        maske = klasse == k
        p_k, y_k = p_arr[maske], y_arr[maske]

        def abweichung(w: np.ndarray, p_k=p_k, y_k=y_k) -> float:
            if _nan_falls_leer(w):
                return float("nan")
            return float(np.average(p_k - y_k, weights=w))

        iv = cluster_bootstrap(abweichung, cluster_arr[maske], n_ziehungen=n_ziehungen, seed=seed)
        zeilen.append(
            Klassenzeile(
                von=float(grenzen[int(k)]),
                bis=float(grenzen[int(k) + 1]),
                n=int(maske.sum()),
                n_cluster=iv.n_cluster,
                preis_mittel=float(p_k.mean()),
                haeufigkeit=float(y_k.mean()),
                abweichung=iv,
            )
        )
    return zeilen


__all__ = [
    "LOG_CLIP",
    "CoxKalibrierung",
    "Intervall",
    "Klassenzeile",
    "MurphyZerlegung",
    "brier_score",
    "brier_skill",
    "cluster_bootstrap",
    "cox_kalibrierung",
    "kalibrierungstabelle",
    "log_score",
    "murphy_zerlegung",
]
