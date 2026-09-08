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


# ---------------------------------------------------------------------------
# Das Embargo: deklariert statt geraten (#320)
# ---------------------------------------------------------------------------


def _spec(**kw) -> StrategySpec:
    grund = dict(
        strategy_id="embargo_test",
        name="Embargo",
        class_path="strategies.templates.sma_crossover:SmaCrossover",
        strategy_class="trend_following",
        universe="us_core_etfs",
        timeframe=Timeframe.DAILY,
    )
    return StrategySpec(**{**grund, **kw})


class TestEmbargoWirdDeklariert:
    """Der Lookback wird deklariert, nicht aus dem Grid geraten (#320).

    **Der Fehler.** `_infer_embargo` nahm die grösste ganze Zahl im
    `param_space`. Das trifft, solange dort nur Lookbacks stehen, und geht in
    beide Richtungen schief, sobald etwas anderes dazukommt — und beide
    Richtungen sind teuer:

    * Zu gross: `n_positions: [10, 50, 500]` ergibt 500 Bars Embargo, zwei
      Jahre fallen aus jedem Fold, und weil `min_train` mitwächst und
      degenerierte Folds übersprungen werden, verschwindet die Validierung
      **still**.
    * Zu klein: ein Float-Lookback wurde verworfen, das Embargo war 0, und der
      OOS-Rand las Train-Preise — genau der Leak, gegen den es gebaut ist.
    """

    def test_ein_grosser_nicht_lookback_erzeugt_kein_embargo(self):
        """Der Fall aus dem Ticket: `n_positions: 500` ist kein Rückblick."""
        from quantrace.walk_forward import _infer_embargo

        spec = _spec(
            param_space={"window": [10, 20], "n_positions": [10, 50, 500]},
            lookback_keys=("window",),
        )
        bars, quelle = _infer_embargo(spec)
        assert bars == 20, "nur `window` blickt zurück"
        assert quelle == "deklariert"

    def test_ohne_deklaration_wird_weiter_geraten(self):
        """Der Rückfall bleibt — ein fehlendes Embargo wäre schlimmer als ein
        zu grosses. Aber er sagt, dass er geraten hat."""
        from quantrace.walk_forward import _infer_embargo

        spec = _spec(param_space={"window": [10, 20], "n_positions": [10, 50, 500]})
        bars, quelle = _infer_embargo(spec)
        assert bars == 500
        assert quelle == "geraten"

    def test_ein_float_lookback_ergibt_kein_null_embargo(self):
        """`halflife: 20.5` kam vorher als *nichts* an. Aufgerundet, weil ein
        Rückblick von 20,5 Bars bis in den 21. reicht."""
        from quantrace.walk_forward import _infer_embargo

        spec = _spec(param_space={"halflife": [12.5, 20.5]}, lookback_keys=("halflife",))
        bars, quelle = _infer_embargo(spec)
        assert bars == 21
        assert quelle == "deklariert"

    def test_ein_fester_lookback_zaehlt_auch(self):
        """Nicht gesweept heisst nicht „blickt nicht zurück"."""
        from quantrace.walk_forward import _infer_embargo

        spec = _spec(
            params={"window": 200},
            param_space={"entry_z": [1.0, 2.0]},
            lookback_keys=("window",),
        )
        assert _infer_embargo(spec)[0] == 200

    def test_ein_tippfehler_in_der_deklaration_wird_gemeldet(self, caplog):
        """Sonst wäre er ein stilles Embargo von 0 — dieselbe Undichtigkeit
        wie vorher, nur mit Zeremonie."""
        import logging

        from quantrace.walk_forward import _infer_embargo

        spec = _spec(param_space={"window": [10, 20]}, lookback_keys=("windwo",))
        with caplog.at_level(logging.WARNING, logger="quantrace.walk_forward"):
            bars, quelle = _infer_embargo(spec)
        assert bars == 0 and quelle == "deklariert"
        assert "windwo" in caplog.text

    def test_bools_zaehlen_nicht_als_bars(self):
        """`bool` ist in Python ein `int` — `use_log: [True, False]` wäre
        sonst ein Embargo von einem Bar."""
        from quantrace.walk_forward import _infer_embargo

        spec = _spec(param_space={"use_log": [True, False]}, lookback_keys=("use_log",))
        assert _infer_embargo(spec)[0] == 0

    def test_ein_leerer_raum_ergibt_null(self):
        from quantrace.walk_forward import _infer_embargo

        assert _infer_embargo(_spec())[0] == 0


class TestDasErgebnisTraegtDieHerkunft:
    """Geraten darf nicht aussehen wie belegt (#320)."""

    def test_die_herkunft_steht_im_ergebnis(self, synthetic_md):
        spec = _spec(param_space={"fast": [5], "slow": [20]}, lookback_keys=("slow",))
        res = walk_forward(spec, synthetic_md, n_folds=2)
        assert res.embargo == 20
        assert res.embargo_source == "deklariert"

    def test_ein_vorgegebenes_embargo_ist_als_solches_kenntlich(self, synthetic_md):
        spec = _spec(param_space={"fast": [5], "slow": [20]})
        res = walk_forward(spec, synthetic_md, n_folds=2, embargo=7)
        assert res.embargo == 7
        assert res.embargo_source == "vorgegeben"

    def test_angeforderte_und_gerechnete_folds_stehen_beide_da(self, synthetic_md):
        """Eine Validierung über zwei statt sechs Folds ist eine andere
        Aussage, nicht dieselbe mit weniger Zeilen."""
        spec = _spec(param_space={"fast": [5], "slow": [20]}, lookback_keys=("slow",))
        res = walk_forward(spec, synthetic_md, n_folds=2)
        assert res.n_folds_requested == 2
        assert res.n_folds == len(res.folds)

    def test_verschwundene_folds_werden_gemeldet(self, synthetic_md, caplog):
        """Folds fallen aus, und das stand bisher nirgends.

        Gemessen am Fixture (523 Bars): bei acht angeforderten Folds bleiben
        sieben — schon bei kleinem Embargo, weil `min_train` mitwächst. Genau
        diese Sorte Reduktion ist die stille: sie wirft keinen Fehler, sie
        liefert einfach weniger Validierung.
        """
        import logging

        spec = _spec(param_space={"fast": [5], "slow": [20]})
        with caplog.at_level(logging.WARNING, logger="quantrace.walk_forward"):
            res = walk_forward(spec, synthetic_md, n_folds=8, embargo=20)
        assert res.n_folds_requested == 8
        assert res.n_folds < 8, "das Fixture traegt keine acht Folds"
        assert "von 8 angeforderten Folds" in caplog.text

    def test_ein_zu_grosses_embargo_scheitert_laut(self, synthetic_md):
        """Bleibt **kein** Fold uebrig, ist das ein Fehler und keine leere
        Antwort — sonst saehe „nichts validiert" aus wie „nichts gefunden"."""
        import pytest as _pytest

        spec = _spec(param_space={"fast": [5], "slow": [20]})
        with _pytest.raises(ValueError, match="Kein gültiger Walk-Forward-Fold"):
            walk_forward(spec, synthetic_md, n_folds=8, embargo=400)


def test_ein_geratenes_embargo_von_null_wird_gemeldet(caplog):
    """`kalman_trend` ist der Fall: `delta: 1e-4`, `meas_var: 1e-3` — keine
    ganze Zahl im Grid, also raet der Rückfall **0**.

    Die alte Warnung sprang nur bei Werten über 0 an; ein Embargo von 0 sah
    damit aus wie „braucht keins" statt wie „konnte nicht bestimmt werden".
    Der Filter läuft rekursiv über die ganze Reihe und hat gar kein endliches
    Fenster — das ist eine fehlende Antwort, keine Null.
    """
    import logging

    from quantrace.walk_forward import _infer_embargo

    spec = _spec(param_space={"delta": [1e-5, 1e-4], "meas_var": [1e-3]})
    with caplog.at_level(logging.WARNING, logger="quantrace.walk_forward"):
        bars, quelle = _infer_embargo(spec)
    assert (bars, quelle) == (0, "geraten")
    assert "Embargo **0**" in caplog.text
    assert "ungeschützt" in caplog.text
