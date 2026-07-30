"""Factor attribution — coefficient recovery, alpha inference, HAC errors."""

from __future__ import annotations

import numpy as np
import pytest

from quantrace.stats.attribution import (
    default_hac_lags,
    factor_attribution,
    rolling_factor_attribution,
)


def _factors(t: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0003, 0.01, (t, k))


def test_recovers_known_betas_and_alpha():
    t = 3000
    f = _factors(t, 2, seed=1)
    rng = np.random.default_rng(2)
    alpha_daily = 0.0004
    y = alpha_daily + 0.8 * f[:, 0] - 0.3 * f[:, 1] + rng.normal(0, 0.002, t)

    res = factor_attribution(y, f, ["mkt", "value"])
    assert res.exposures[0].beta == pytest.approx(0.8, abs=0.03)
    assert res.exposures[1].beta == pytest.approx(-0.3, abs=0.03)
    assert res.alpha_annual == pytest.approx(alpha_daily * 252, rel=0.25)
    assert res.alpha_t_stat > 3.0  # real alpha, long sample → significant
    assert res.alpha_p_value < 0.01


def test_pure_beta_has_no_alpha_and_high_r2():
    t = 2000
    f = _factors(t, 1, seed=3)
    rng = np.random.default_rng(4)
    y = 1.2 * f[:, 0] + rng.normal(0, 0.0005, t)  # beta in a trench coat

    res = factor_attribution(y, f, ["spy"])
    assert res.r_squared > 0.95
    assert abs(res.alpha_t_stat) < 2.0
    assert abs(res.residual_sharpe_annual) < 1.0


def test_orthogonal_strategy_low_r2():
    t = 2000
    f = _factors(t, 3, seed=5)
    rng = np.random.default_rng(6)
    y = rng.normal(0.0005, 0.01, t)  # independent of the factors

    res = factor_attribution(y, f, ["a", "b", "c"])
    assert res.r_squared < 0.02
    for e in res.exposures:
        assert abs(e.beta) < 0.1


def test_hac_widens_errors_under_autocorrelation():
    # AR(1) residuals inflate the naive OLS t-stat; Newey–West should shrink
    # it relative to lags=0 (which is plain heteroskedasticity-robust).
    t = 3000
    f = _factors(t, 1, seed=7)
    rng = np.random.default_rng(8)
    eps = np.zeros(t)
    shocks = rng.normal(0, 0.004, t)
    for i in range(1, t):
        eps[i] = 0.7 * eps[i - 1] + shocks[i]
    y = 0.0003 + 0.5 * f[:, 0] + eps

    res_hac = factor_attribution(y, f, ["mkt"], hac_lags=20)
    res_lag0 = factor_attribution(y, f, ["mkt"], hac_lags=0)
    assert abs(res_hac.alpha_t_stat) < abs(res_lag0.alpha_t_stat)


def test_default_hac_lags_rule():
    assert default_hac_lags(100) == 4
    assert default_hac_lags(1000) == 6
    assert default_hac_lags(10) < 4


def test_single_factor_1d_input_accepted():
    t = 500
    f = _factors(t, 1, seed=9)
    y = 0.5 * f[:, 0]
    res = factor_attribution(y, f[:, 0], ["mkt"])
    assert res.exposures[0].beta == pytest.approx(0.5, abs=1e-6)
    assert res.r_squared == pytest.approx(1.0, abs=1e-9)


def test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        factor_attribution(np.zeros(100), _factors(99, 1), ["x"])


def test_rejects_name_count_mismatch():
    with pytest.raises(ValueError):
        factor_attribution(np.zeros(100), _factors(100, 2), ["only_one"])


def test_rejects_nan():
    y = np.zeros(100)
    f = _factors(100, 1)
    y[5] = np.nan
    with pytest.raises(ValueError):
        factor_attribution(y, f, ["x"])


def test_rejects_too_few_obs():
    with pytest.raises(ValueError):
        factor_attribution(np.zeros(8), _factors(8, 2), ["a", "b"])


def test_to_dict_json_friendly():
    t = 300
    f = _factors(t, 2, seed=10)
    y = 0.3 * f[:, 0] + 0.1 * f[:, 1]
    d = factor_attribution(y, f, ["a", "b"]).to_dict()
    assert d["method"] == "ols_newey_west"
    assert len(d["exposures"]) == 2
    assert {"name", "beta", "t_stat", "p_value"} == set(d["exposures"][0])


# -----------------------------------------------------------------------------
# Rolling attribution (beta timeline)
# -----------------------------------------------------------------------------


def test_rolling_recovers_constant_beta():
    t = 800
    f = _factors(t, 1, seed=11)
    rng = np.random.default_rng(12)
    y = 0.7 * f[:, 0] + rng.normal(0, 0.001, t)

    pts = rolling_factor_attribution(y, f, window=126, step=21)
    assert len(pts) == (t - 126) // 21 + 1
    for p in pts:
        assert p.betas[0] == pytest.approx(0.7, abs=0.1)
        assert p.r_squared > 0.9
    # end_index chronologisch aufsteigend, letzter innerhalb T
    ends = [p.end_index for p in pts]
    assert ends == sorted(ends)
    assert ends[-1] <= t - 1


def test_rolling_tracks_beta_regime_change():
    # Erste Hälfte β=+1, zweite Hälfte β=−1 — Full-Sample-OLS mittelt das zu
    # ~0 weg, die Timeline muss den Umschwung zeigen.
    t = 1000
    f = _factors(t, 1, seed=13)
    rng = np.random.default_rng(14)
    beta_path = np.where(np.arange(t) < t // 2, 1.0, -1.0)
    y = beta_path * f[:, 0] + rng.normal(0, 0.001, t)

    pts = rolling_factor_attribution(y, f, window=126, step=21)
    first = pts[0].betas[0]
    last = pts[-1].betas[0]
    assert first == pytest.approx(1.0, abs=0.15)
    assert last == pytest.approx(-1.0, abs=0.15)

    full = factor_attribution(y, f, ["mkt"])
    assert abs(full.exposures[0].beta) < 0.3  # der Durchschnitt verschleiert


def test_rolling_rejects_short_series_and_bad_params():
    f = _factors(100, 1)
    y = np.zeros(100)
    with pytest.raises(ValueError, match="shorter than window"):
        rolling_factor_attribution(y, f, window=126)
    with pytest.raises(ValueError, match="too small"):
        rolling_factor_attribution(y, f, window=5)
    with pytest.raises(ValueError, match="step"):
        rolling_factor_attribution(y, f, window=50, step=0)
