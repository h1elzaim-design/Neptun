"""Feature engineering for regime detection.

A regime is characterised by *trend* (are returns drifting up or down?) and
*risk* (how violent are the moves?). We feed the HMM two trailing, fully causal
features so the resulting regimes are persistent and interpretable rather than
flickering on daily noise:

    trend  — annualised mean log-return over a trailing window
    vol    — annualised realised volatility over the same window

Both are computed from a single benchmark price series. For a multi-asset
universe the benchmark is the equal-weight average price (see
:func:`benchmark_series`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252.0


def benchmark_series(prices: pd.Series | pd.DataFrame) -> pd.Series:
    """Collapse prices to a single benchmark series.

    A ``Series`` is returned as-is. A ``DataFrame`` (columns = symbols) becomes
    the equal-weight path via :func:`quantrace.data_agent.equal_weight_benchmark`.

    **Über Renditen, nicht über Kurse.** Der Mittelwert einer Kursmatrix ist
    kursgewichtet (Dow-Logik). Solange die Zusammensetzung fest ist, ist das
    eine Ungenauigkeit im Namen; sobald sie sich ändert, springt der Mittelwert
    ohne dass sich ein Kurs bewegt hat — und für ein Modell, das genau Sprünge
    lernt, ist das an jedem Rekonstitutionsdatum ein Regimewechsel, den es nie
    gab (#255). Seit #255 kann sich die Zusammensetzung ändern.

    Für den HMM ist die *Höhe* der Reihe gleichgültig — ``regime_features``
    rechnet auf ``log(bench).diff()``, und ein konstanter Faktor kürzt sich
    heraus. Verglichen wird also nur die Form, und die ist über Renditen die
    richtige.

    Der Import steht in der Funktion: ``data_agent`` zieht Storage, Provider
    und Credentials nach, und ein Feature-Modul soll das nicht beim Import
    eines Regime-Fits tun. Eine zweite Implementierung derselben Rechnung wäre
    die teurere Lösung — zwei Quellen, die nichts voneinander wissen, sind das
    Muster, an dem hier schon einmal eine Zahl auseinanderlief.
    """
    if isinstance(prices, pd.Series):
        return prices.astype(float)

    from quantrace.data_agent import equal_weight_benchmark

    return equal_weight_benchmark(prices.astype(float))


def regime_features(
    prices: pd.Series | pd.DataFrame,
    *,
    window: int = 21,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Trailing trend + realised-vol features, indexed by date.

    The first ``window`` rows are dropped (insufficient history). Both columns
    are annualised so their scale is intuitive (e.g. vol ≈ 0.15 is a calm 15 %).

    ``macro`` fügt optional Makro-Spalten hinzu (Zinskurve, Credit-Spreads —
    siehe ``quantrace.regime.macro``). **Default ist ``None``, und das bleibt
    so**: Issue #231 hat gemessen, dass dieses HMM auf zusätzliche Freiheit mit
    Überanpassung reagiert. Jede weitere Dimension kostet eine Zeile der
    Kovarianzmatrix pro Zustand. Einschalten heißt, die Messung aus #231 zu
    wiederholen — nicht, ein Flag zu setzen.

    Die Makro-Spalten werden **nur vorwärts** auf den Kursindex gelegt; ein
    Feiertag darf den Wert des Folgetags nicht rückwirkend sichtbar machen.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    bench = benchmark_series(prices)
    log_ret = np.log(bench).diff()

    trend = log_ret.rolling(window).mean() * _TRADING_DAYS
    vol = log_ret.rolling(window).std(ddof=1) * np.sqrt(_TRADING_DAYS)

    feats = pd.DataFrame({"trend": trend, "vol": vol})

    if macro is not None and not macro.empty:
        from quantrace.regime.macro import align_macro

        feats = feats.join(align_macro(macro, feats.index))

    return feats.replace([np.inf, -np.inf], np.nan).dropna()
