"""Tests für die Walk-Forward-Validation."""

from __future__ import annotations

import pandas as pd
import pytest

from quantrace.models import StrategySpec, Timeframe
from quantrace.walk_forward import WalkForwardResult, _slice_market_data, walk_forward


def test_slice_market_data(synthetic_md):
    """Testet das Herausschneiden eines Zeitraums aus MarketData."""
    start = pd.Timestamp("2019-06-01")
    end = pd.Timestamp("2019-12-31")

    sliced = _slice_market_data(synthetic_md, start, end)

    assert sliced.start >= start.date()
    assert sliced.end <= end.date()
    assert len(sliced.frame) < len(synthetic_md.frame)
    assert sliced.symbols == synthetic_md.symbols


def test_slice_market_data_empty(synthetic_md):
    """Sollte ValueError werfen, wenn der Slice leer ist."""
    start = pd.Timestamp("2018-01-01")
    end = pd.Timestamp("2018-12-31")

    with pytest.raises(ValueError, match="ist leer"):
        _slice_market_data(synthetic_md, start, end)


def test_walk_forward_basic(synthetic_md):
    """Testet einen vollständigen Walk-Forward-Durchlauf."""
    spec = StrategySpec(
        strategy_id="wf_test",
        name="WF Test",
        class_path="strategies.templates.sma_crossover:SmaCrossover",
        strategy_class="trend_following",
        universe="synthetic",
        timeframe=Timeframe.DAILY,
        params={"fast": 10, "slow": 50},
        param_space={"fast": [10, 20], "slow": [50]},
    )

    # synthetic_md hat 5 Jahre Daten. Der degenerierte erste Fold wird jetzt
    # übersprungen, also bleiben ≥1 (typisch 2) saubere Folds.
    result = walk_forward(spec, synthetic_md, n_folds=3, train_ratio=0.5)

    assert isinstance(result, WalkForwardResult)
    assert 1 <= len(result.folds) <= 3
    assert result.n_folds == len(result.folds)  # spiegelt tatsächlich evaluierte Folds
    assert result.rank_by == "sharpe"

    # Prüfe Fold-Inhalte — jeder Fold ist nicht-degeneriert (kein 1-Bar-Train).
    for i, fold in enumerate(result.folds, 1):
        assert fold.fold_index == i
        assert fold.train_end < fold.test_start  # Embargo-/No-Overlap-Disziplin
        assert "fast" in fold.chosen_params
        assert "slow" in fold.chosen_params


def test_walk_forward_oos_fdr(synthetic_md):
    """Jeder Fold trägt einen p-Wert aus den echten OOS-Returns; über die
    Folds läuft Benjamini-Hochberg (nur definiert für ≥ 2 Folds)."""
    spec = StrategySpec(
        strategy_id="wf_fdr_test",
        name="WF FDR Test",
        class_path="strategies.templates.sma_crossover:SmaCrossover",
        strategy_class="trend_following",
        universe="synthetic",
        timeframe=Timeframe.DAILY,
        params={"fast": 10, "slow": 50},
        param_space={"fast": [10, 20], "slow": [50]},
    )

    result = walk_forward(spec, synthetic_md, n_folds=3, train_ratio=0.5)

    for fold in result.folds:
        assert fold.test_n_obs is not None and fold.test_n_obs >= 3
        assert fold.test_p_value is not None
        assert 0.0 <= fold.test_p_value <= 1.0

    if len(result.folds) >= 2:
        assert result.fdr is not None
        assert result.fdr["method"] == "benjamini_hochberg"
        assert result.fdr["n_tests"] == len(result.folds)
        assert result.fdr["scope"] == "oos_folds"
        n_sig = 0
        for fold in result.folds:
            assert fold.test_q_value is not None
            assert fold.test_q_value >= fold.test_p_value - 1e-12  # BH verkleinert nie
            assert fold.fdr_significant == (fold.test_q_value <= result.fdr["alpha"])
            n_sig += bool(fold.fdr_significant)
        assert result.fdr["n_significant"] == n_sig
        assert result.fdr["all_significant"] == (n_sig == len(result.folds))
    else:
        assert result.fdr is None


def test_walk_forward_stitched_oos_inference(synthetic_md):
    """Der gestitchte OOS-Pfad (alle Test-Fenster konkateniert) trägt Sharpe +
    Stationary-Bootstrap-KI — und T ist die Summe der Fold-Beobachtungen."""
    spec = StrategySpec(
        strategy_id="wf_stitched_test",
        name="WF Stitched Test",
        class_path="strategies.templates.sma_crossover:SmaCrossover",
        strategy_class="trend_following",
        universe="synthetic",
        timeframe=Timeframe.DAILY,
        params={"fast": 10, "slow": 50},
        param_space={"fast": [10, 20], "slow": [50]},
    )

    result = walk_forward(spec, synthetic_md, n_folds=3, train_ratio=0.5)

    inf = result.oos_inference
    assert inf is not None
    assert inf["method"] == "stitched_oos_stationary_bootstrap"
    assert inf["ci_low"] <= inf["sharpe_annual"] <= inf["ci_high"]
    assert 0.0 < inf["p_value"] <= 1.0
    assert inf["n_folds_stitched"] == len(result.folds)
    assert inf["n_obs"] == sum(f.test_n_obs for f in result.folds)

    # Persistiert im JSON-Roundtrip (Vault-Note-Quelle).
    import json

    payload = json.loads(result.model_dump_json())
    assert payload["oos_inference"]["n_obs"] == inf["n_obs"]

    # Der gestitchte OOS-Equity-Pfad selbst wird mitpersistiert: chronologisch,
    # kettennormiert über die Fold-Grenzen (kein Sprung zurück auf 1.0).
    eq = payload["oos_equity"]
    assert eq is not None and len(eq) >= 8
    dates = [p["date"] for p in eq]
    assert dates == sorted(dates)
    values = [p["value"] for p in eq]
    assert all(v > 0 for v in values)
    # Kettennormierung: erster Punkt startet bei 1.0 (skaliert), und die
    # Anzahl der Punkte entspricht der Summe der Fold-Kurvenlängen.
    assert values[0] == pytest.approx(1.0, rel=1e-9)


def test_walk_forward_no_train_test_overlap(synthetic_md):
    """Out-of-Sample darf nicht am selben Bar wie train_end starten — sonst
    leakt der letzte Trainings-Bar ins Test-Set (pandas .loc ist inklusiv)."""
    spec = StrategySpec(
        strategy_id="wf_test",
        name="WF Test",
        class_path="strategies.templates.sma_crossover:SmaCrossover",
        strategy_class="trend_following",
        universe="synthetic",
        timeframe=Timeframe.DAILY,
        params={"fast": 10, "slow": 50},
        param_space={"fast": [10, 20], "slow": [50]},
    )

    result = walk_forward(spec, synthetic_md, n_folds=3, train_ratio=0.5)

    for fold in result.folds:
        assert fold.test_start > fold.train_end, (
            f"Fold {fold.fold_index}: test_start {fold.test_start} überlappt "
            f"train_end {fold.train_end}"
        )


def test_walk_forward_not_enough_data(synthetic_md):
    """split_walk_forward wirft ValueError bei zu wenig Daten."""
    spec = StrategySpec(
        strategy_id="wf_test",
        name="WF Test",
        class_path="strategies.templates.sma_crossover:SmaCrossover",
        strategy_class="trend_following",
        universe="synthetic",
        timeframe=Timeframe.DAILY,
        param_space={"fast": [10], "slow": [50]},
    )

    # Wir schneiden künstlich auf 50 Tage runter
    small_md = _slice_market_data(
        synthetic_md, synthetic_md.frame.index[0], synthetic_md.frame.index[49]
    )

    with pytest.raises(ValueError, match="Zu wenig Daten"):
        walk_forward(spec, small_md, n_folds=3)
