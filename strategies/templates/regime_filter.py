"""Regime-gefilterter Trend-Follower — HMM-Makro-Overlay über einem SMA-Trend.

Basissignal: simpler Long-Trend (Close > SMA(trend_lookback)). Darüber liegt ein
Hidden-Markov-Regime-Filter (:mod:`quantrace.regime`): Long-Exposure ist nur
erlaubt, solange der Markt **nicht** in einem Risk-off-Regime steckt. So umgeht
die Strategie die großen Drawdowns, die reine Trendfolger in Volatilitäts-Spikes
einfahren — der klassische Grund, warum Trend-Strategien 2008/2020 wehtaten.

Das Regime wird auf einem Equal-Weight-Benchmark des Universums geschätzt und auf
alle Symbole gebroadcastet: es ist ein *Makro*-Schalter, kein Per-Symbol-Signal.

Look-ahead-Disziplin: Die HMM-*Parameter* und die State→Label-Zuordnung werden
ausschließlich auf einem **anchored Train-Fenster** (die ersten
``regime_train_window`` Bars) geschätzt und dann eingefroren. Der Regime-*Pfad*
außerhalb des Train-Fensters ist die gefilterte (kausale) Posterior unter diesen
eingefrorenen Parametern — am Tag *t* fließt nur Information bis *t* ein, und die
Regime-*Definition* selbst enthält keine Zukunft. Innerhalb des Train-Fensters
wird **nicht** gegated (die Labels dort sind in-sample), sodass kein In-Sample-
Regime einen Trade beeinflusst. (Ein periodischer Expanding-Window-Refit wäre noch
strenger, ändert den qualitativen Verlauf aber selten — bewusster Trade-off.)
"""

from __future__ import annotations

import pandas as pd

from quantrace.models import MarketData
from quantrace.regime import RegimeDetector
from quantrace.regime.features import benchmark_series
from quantrace.strategy import Strategy


class RegimeFilter(Strategy):
    defaults = {
        "trend_lookback": 100,
        "n_states": 3,
        "feature_window": 21,
        # ~3 trading years of burn-in to estimate the regime model before any
        # of its labels are allowed to gate a trade. Series shorter than this
        # simply trade the raw trend (no causal regime available yet).
        "regime_train_window": 756,
    }

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = self.close(data)
        n = int(self.params["trend_lookback"])

        sma = close.rolling(n).mean()
        uptrend = (close > sma).fillna(False).astype(bool)

        # Risk-off-Maske aus dem HMM, auf den Index ausgerichtet, vor dem Warmup
        # bewusst False (nicht blockieren — das Basissignal entscheidet).
        risk_off = self._risk_off_mask(close)
        allow = (~risk_off).astype(int)

        long_now = uptrend.mul(allow, axis=0).astype(bool)
        long_prev = long_now.shift(1, fill_value=False).astype(bool)
        entries = long_now & ~long_prev
        exits = ~long_now & long_prev
        return entries.astype(bool), exits.astype(bool)

    def _risk_off_mask(self, close: pd.DataFrame) -> pd.Series:
        # Equal-Weight-Markt-Proxy — über `benchmark_series`, nicht über
        # `close.mean(axis=1)`. Das Kursmittel ist kursgewichtet und springt bei
        # jedem Mitgliederwechsel; auf einem rekonstituierten Universum (#255)
        # lernte das HMM dort einen Regimewechsel, den es nie gab, und genau
        # dieses Regime schaltet hier die Position.
        bench = benchmark_series(close)
        train_window = int(self.params["regime_train_window"])

        det = RegimeDetector(
            n_states=int(self.params["n_states"]),
            feature_window=int(self.params["feature_window"]),
        )
        try:
            # Parameter + Label-Map NUR auf dem Train-Fenster schätzen …
            det.fit(bench.iloc[:train_window])
            # … und mit eingefrorenen Parametern kausal über die volle Serie filtern.
            regime = det.regime_series(bench, mode="filter")
        except ValueError:
            # Zu wenig Historie für ein HMM → kein Gating.
            return pd.Series(False, index=close.index)

        mask = regime.isin(det.risk_off_labels)
        # In-Sample-Region (das Train-Fenster) darf nicht gaten — ihre Labels
        # wurden mit Hindsight relativ zu genau diesen Bars geschätzt. Ohne OOS-
        # Region (Serie ≤ Train-Fenster) gibt es nichts kausal Gatebares.
        if train_window < len(bench):
            train_end = bench.index[train_window - 1]
            mask.loc[mask.index <= train_end] = False
        else:
            mask.loc[:] = False
        # fill_value=False statt fillna → kein object-Downcast, bleibt bool.
        return mask.reindex(close.index, fill_value=False).astype(bool)
