"""Tests for RegimeDetector — semantic labelling and causal regime series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantrace.regime import RegimeDetector, regime_features


def _two_regime_prices(seed: int = 0) -> pd.Series:
    """A price path: long calm bull, then a sharp high-vol drawdown, then recovery."""
    rng = np.random.default_rng(seed)
    bull1 = rng.normal(0.0006, 0.006, 400)   # calm uptrend
    crash = rng.normal(-0.004, 0.030, 120)   # violent downtrend
    bull2 = rng.normal(0.0007, 0.007, 400)   # calm recovery
    rets = np.concatenate([bull1, crash, bull2])
    idx = pd.bdate_range("2020-01-01", periods=len(rets))
    return pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx, name="px")


def test_features_are_causal_and_annualised():
    px = _two_regime_prices()
    feats = regime_features(px, window=21)
    assert list(feats.columns) == ["trend", "vol"]
    # First 21 rows dropped (warmup).
    assert len(feats) == len(px) - 21
    # Vol is annualised and positive.
    assert (feats["vol"] > 0).all()


def test_labels_track_risk():
    px = _two_regime_prices()
    det = RegimeDetector(n_states=3, feature_window=21).fit(px)
    series = det.regime_series(px)  # causal by default

    # The crash window sits roughly at rows 400..520 of the price series.
    crash_dates = px.index[400:520]
    crash_labels = series.reindex(crash_dates).dropna()
    # During the crash, risk_off should dominate over risk_on.
    n_risk_off = (crash_labels == "risk_off").sum()
    n_risk_on = (crash_labels == "risk_on").sum()
    assert n_risk_off > n_risk_on


def test_probabilities_sum_to_one_and_match_labels():
    px = _two_regime_prices()
    det = RegimeDetector(n_states=3).fit(px)
    proba = det.probabilities(px)
    np.testing.assert_allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-8)
    # idxmax of the proba frame must equal the regime series.
    series = det.regime_series(px)
    pd.testing.assert_series_equal(
        proba.idxmax(axis=1).rename("regime"), series, check_names=True
    )


def test_current_regime_snapshot():
    px = _two_regime_prices()
    det = RegimeDetector(n_states=3).fit(px)
    snap = det.current_regime(px)
    assert snap.label in det.labels
    assert 0.0 <= snap.confidence <= 1.0
    assert abs(sum(snap.probabilities.values()) - 1.0) < 1e-8


def test_label_ladder_sizes():
    px = _two_regime_prices()
    assert RegimeDetector(n_states=2).fit(px).labels == ["risk_off", "risk_on"]
    assert RegimeDetector(n_states=4).fit(px).labels == [
        "crisis",
        "risk_off",
        "neutral",
        "risk_on",
    ]


def test_risk_off_labels_are_bottom_half():
    px = _two_regime_prices()
    det = RegimeDetector(n_states=4).fit(px)
    assert det.risk_off_labels == {"crisis", "risk_off"}


def test_invalid_n_states():
    with pytest.raises(ValueError, match="n_states"):
        RegimeDetector(n_states=9)


def test_fit_rejects_too_little_history():
    px = pd.Series(
        [100.0, 101.0, 102.0, 103.0],
        index=pd.bdate_range("2020-01-01", periods=4),
    )
    with pytest.raises(ValueError, match="Not enough history"):
        RegimeDetector(n_states=3, feature_window=21).fit(px)


def test_accepts_multi_asset_frame():
    px = _two_regime_prices()
    frame = pd.DataFrame({"A": px, "B": px * 1.5})
    det = RegimeDetector(n_states=3).fit(frame)
    series = det.regime_series(frame)
    assert not series.empty
