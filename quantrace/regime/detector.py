"""RegimeDetector — high-level, semantic regime labels from prices.

Wraps :class:`~quantrace.regime.hmm.GaussianHMM` with two pieces of
domain knowledge:

1. **Feature construction** (trailing trend + realised vol) via
   :mod:`quantrace.regime.features`.
2. **Semantic labelling** — the HMM's anonymous states are sorted into a fixed
   risk ladder so "state 2" always means the same thing across fits. States are
   scored by ``z(trend) − z(vol)`` (high return, low vol = best) and mapped onto
   a label ladder sized to ``n_states``:

       2 states → ["risk_off", "risk_on"]
       3 states → ["risk_off", "neutral", "risk_on"]
       4 states → ["crisis", "risk_off", "neutral", "risk_on"]

The series read off here is the **causal/filtered** posterior, so a consumer
reading it at time *t* sees only information available at *t* *given the fitted
parameters*.

Important: ``fit`` estimates parameters (and the state→label map) over whatever
series it is handed. Calling ``fit(full_series)`` then reading a "filtered" path
is fine for **analysis / visualization** (e.g. the Regime-Lab API), but it is
*not* leak-free for a trading signal, because the regime *definition* then
embeds the whole sample. Signal paths must fit on a train window and decode the
rest with frozen parameters — see ``strategies/templates/regime_filter.py`` for
the anchored train-split pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantrace.regime.features import regime_features
from quantrace.regime.hmm import GaussianHMM

_LABEL_LADDERS: dict[int, list[str]] = {
    1: ["neutral"],
    2: ["risk_off", "risk_on"],
    3: ["risk_off", "neutral", "risk_on"],
    4: ["crisis", "risk_off", "neutral", "risk_on"],
    5: ["crisis", "risk_off", "neutral", "risk_on", "euphoria"],
}


@dataclass(frozen=True)
class RegimeSnapshot:
    """The regime as of a single date."""

    as_of: pd.Timestamp
    label: str
    state: int
    probabilities: dict[str, float]

    @property
    def confidence(self) -> float:
        return float(self.probabilities.get(self.label, 0.0))


class RegimeDetector:
    """Fit on prices, read off semantic market regimes.

    Parameters
    ----------
    n_states:
        Number of regimes (2–5 supported by the label ladder).
    feature_window:
        Trailing window (days) for the trend/vol features.
    n_iter:
        Max Baum-Welch iterations.
    """

    def __init__(
        self,
        *,
        n_states: int = 3,
        feature_window: int = 21,
        n_iter: int = 50,
    ) -> None:
        if n_states not in _LABEL_LADDERS:
            raise ValueError(f"n_states must be one of {sorted(_LABEL_LADDERS)}")
        self.n_states = n_states
        self.feature_window = feature_window
        self.hmm = GaussianHMM(n_states=n_states, n_iter=n_iter)
        self.state_to_label_: dict[int, str] = {}
        self._features: pd.DataFrame | None = None

    # -- fit ------------------------------------------------------------------

    def fit(self, prices: pd.Series | pd.DataFrame) -> RegimeDetector:
        feats = regime_features(prices, window=self.feature_window)
        if len(feats) < self.n_states + 1:
            raise ValueError(
                f"Not enough history: {len(feats)} feature rows for "
                f"{self.n_states} states (need ≥ {self.n_states + 1})."
            )
        self._features = feats
        self.hmm.fit(feats.to_numpy())
        self._assign_labels()
        return self

    def _assign_labels(self) -> None:
        """Rank states best→worst on z(trend) − z(vol) and map to the ladder."""
        means = self.hmm.means_
        assert means is not None
        trend, vol = means[:, 0], means[:, 1]

        def _z(a: np.ndarray) -> np.ndarray:
            sd = a.std(ddof=0)
            return (a - a.mean()) / sd if sd > 1e-12 else np.zeros_like(a)

        score = _z(trend) - _z(vol)
        # Ascending score → worst regime first, matching the label ladder order.
        order = np.argsort(score)
        ladder = _LABEL_LADDERS[self.n_states]
        self.state_to_label_ = {int(state): ladder[rank] for rank, state in enumerate(order)}

    # -- inference ------------------------------------------------------------

    def _proba_frame(self, prices: pd.Series | pd.DataFrame, mode: str) -> pd.DataFrame:
        if not self.state_to_label_:
            raise RuntimeError("RegimeDetector is not fitted — call fit() first")
        feats = regime_features(prices, window=self.feature_window)
        proba = self.hmm.predict_proba(feats.to_numpy(), mode=mode)
        # Each state maps to a distinct ladder label, so columns are unique;
        # reorder them worst→best for stable downstream consumption.
        cols = [self.state_to_label_[s] for s in range(self.n_states)]
        df = pd.DataFrame(proba, index=feats.index, columns=cols)
        ordered = [lab for lab in _LABEL_LADDERS[self.n_states] if lab in df.columns]
        return df[ordered]

    def regime_series(
        self,
        prices: pd.Series | pd.DataFrame,
        *,
        mode: str = "filter",
    ) -> pd.Series:
        """Causal (``mode='filter'``) regime label per date — the argmax label."""
        proba = self._proba_frame(prices, mode)
        return proba.idxmax(axis=1).rename("regime")

    def probabilities(
        self,
        prices: pd.Series | pd.DataFrame,
        *,
        mode: str = "filter",
    ) -> pd.DataFrame:
        """Full per-label posterior probability frame (causal by default)."""
        return self._proba_frame(prices, mode)

    def current_regime(
        self,
        prices: pd.Series | pd.DataFrame,
        *,
        mode: str = "filter",
    ) -> RegimeSnapshot:
        """Regime snapshot for the most recent available date."""
        proba = self._proba_frame(prices, mode)
        last = proba.iloc[-1]
        label = str(last.idxmax())
        inv = {v: k for k, v in self.state_to_label_.items()}
        return RegimeSnapshot(
            as_of=proba.index[-1],
            label=label,
            state=int(inv.get(label, -1)),
            probabilities={k: float(v) for k, v in last.items()},
        )

    @property
    def labels(self) -> list[str]:
        """Risk ladder for this detector, worst → best."""
        return list(_LABEL_LADDERS[self.n_states])

    @property
    def risk_off_labels(self) -> set[str]:
        """Labels considered 'risk-off' (the bottom half of the ladder)."""
        ladder = _LABEL_LADDERS[self.n_states]
        cut = max(1, len(ladder) // 2)
        return set(ladder[:cut])
