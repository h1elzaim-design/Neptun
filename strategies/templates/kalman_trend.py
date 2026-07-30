"""Kalman-Filter-Trend — adaptiver gleitender Trend ohne festes Lookback.

Statt eines SMA mit fixem Fenster schätzt ein lokales lineares Trendmodell
(Level + Slope) per Kalman-Filter den "wahren" Preispfad und dessen Steigung.
Der Filter passt seine Reaktionsgeschwindigkeit datengetrieben an: in ruhigen
Phasen glättet er stark, in schnellen Moves zieht er nach. Long, solange die
geschätzte Steigung positiv ist; Exit, wenn sie dreht.

State-Space-Modell (pro Symbol, auf Log-Preisen):
    Level_t  = Level_{t-1} + Slope_{t-1}
    Slope_t  = Slope_{t-1}
    y_t      = Level_t + ε_t,  ε_t ~ N(0, meas_var)

`delta` steuert die Adaptivität über das Verhältnis Prozess-/Messrauschen
(Q = delta/(1-delta) · meas_var · I). Klein = träge/glatt, groß = nervös/schnell.
Referenz: West & Harrison, *Bayesian Forecasting and Dynamic Models*; die delta-
Parametrisierung ist aus der Pairs-Trading-Literatur (Chan) entlehnt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantrace.models import MarketData
from quantrace.strategy import Strategy


def _kalman_trend_slope(y: np.ndarray, delta: float, meas_var: float) -> np.ndarray:
    """Gefilterte Slope-Schätzung eines lokalen linearen Trends, kausal.

    Führende NaNs (Symbol startet später) werden übersprungen; an Beobachtungs-
    lücken läuft nur der Predict-Schritt. Gibt die Steigung je Zeitschritt zurück.
    """
    n = len(y)
    slope = np.full(n, np.nan)
    f_mat = np.array([[1.0, 1.0], [0.0, 1.0]])
    h_vec = np.array([1.0, 0.0])
    ratio = delta / (1.0 - delta) if 0.0 < delta < 1.0 else delta
    q_cov = np.eye(2) * (ratio * meas_var)

    x_state: np.ndarray | None = None
    p_cov = np.eye(2)
    for t in range(n):
        obs = y[t]
        if x_state is None:
            if np.isnan(obs):
                continue
            x_state = np.array([obs, 0.0])
            p_cov = np.eye(2)
            slope[t] = 0.0
            continue

        # Predict
        x_state = f_mat @ x_state
        p_cov = f_mat @ p_cov @ f_mat.T + q_cov

        # Update (nur bei vorhandener Beobachtung)
        if not np.isnan(obs):
            s_var = float(h_vec @ p_cov @ h_vec + meas_var)
            k_gain = (p_cov @ h_vec) / s_var
            resid = obs - float(h_vec @ x_state)
            x_state = x_state + k_gain * resid
            p_cov = (np.eye(2) - np.outer(k_gain, h_vec)) @ p_cov

        slope[t] = x_state[1]
    return slope


class KalmanTrend(Strategy):
    defaults = {"delta": 1e-4, "meas_var": 1e-3}

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = self.close(data)
        log_px = np.log(close.where(close > 0))
        delta = float(self.params["delta"])
        meas_var = float(self.params["meas_var"])

        slopes = {
            col: _kalman_trend_slope(log_px[col].to_numpy(), delta, meas_var)
            for col in close.columns
        }
        slope_df = pd.DataFrame(slopes, index=close.index)

        long_now = (slope_df > 0).fillna(False).astype(bool)
        long_prev = long_now.shift(1, fill_value=False).astype(bool)
        entries = long_now & ~long_prev
        exits = ~long_now & long_prev
        return entries.astype(bool), exits.astype(bool)
