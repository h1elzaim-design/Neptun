"""Causality guarantee for the regime signal path (fix for the full-sample leak).

The regime filter fits HMM parameters on an anchored train window and freezes
them, then decodes the rest with a causal forward filter. The defining property
of that design — and the thing the old full-sample fit violated — is that
*future data cannot change past regime labels*. If parameters were re-estimated
over the whole sample, appending future observations would shift the fitted
means/transitions and silently rewrite history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantrace.regime import RegimeDetector


def _two_regime_series(n: int, seed: int = 0) -> pd.Series:
    """Calm → volatile → calm synthetic price path with a clear regime shift."""
    rng = np.random.default_rng(seed)
    third = n // 3
    vols = np.concatenate([
        np.full(third, 0.004),
        np.full(third, 0.020),
        np.full(n - 2 * third, 0.004),
    ])
    drift = 0.0003
    rets = rng.normal(drift, vols)
    prices = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx)


def test_frozen_params_decode_is_causal():
    full = _two_regime_series(420)
    train = full.iloc[:200]

    det = RegimeDetector(n_states=2, feature_window=10)
    det.fit(train)  # parameters + label map estimated on train ONLY, then frozen

    labels_short = det.regime_series(full.iloc[:300], mode="filter")
    labels_long = det.regime_series(full, mode="filter")  # same 300-bar prefix + future

    common = labels_short.index.intersection(labels_long.index)
    assert len(common) > 50  # sanity: a real overlap to compare
    # Frozen params + causal forward filter ⇒ the extra future bars leave every
    # earlier label byte-identical. This fails under a full-sample refit.
    assert (labels_short.loc[common] == labels_long.loc[common]).all()


def test_train_only_fit_differs_from_full_sample_fit():
    """Sanity check that fitting on the train window is genuinely train-only:
    a fit over the full sample lands on different parameters."""
    full = _two_regime_series(420, seed=1)

    det_train = RegimeDetector(n_states=2, feature_window=10)
    det_train.fit(full.iloc[:200])

    det_full = RegimeDetector(n_states=2, feature_window=10)
    det_full.fit(full)

    assert det_train.hmm.means_ is not None and det_full.hmm.means_ is not None
    # Different data in → different fitted means out (the leak we removed).
    assert not np.allclose(
        np.sort(det_train.hmm.means_[:, 0]),
        np.sort(det_full.hmm.means_[:, 0]),
    )
