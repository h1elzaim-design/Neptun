from __future__ import annotations

import pytest

from quantrace.models import (
    BacktestConfig,
    KnowledgeNote,
    StrategySpec,
    StrategyStatus,
    Timeframe,
)


def test_market_data_hash_stable(synthetic_md):
    h1 = synthetic_md.content_hash
    assert len(h1) == 16
    # Re-construct must give identical hash
    from quantrace.models import MarketData

    md2 = MarketData(
        universe=synthetic_md.universe,
        symbols=synthetic_md.symbols,
        timeframe=synthetic_md.timeframe,
        start=synthetic_md.start,
        end=synthetic_md.end,
        provider=synthetic_md.provider,
        frame=synthetic_md.frame,
    )
    assert md2.content_hash == h1


def test_strategy_spec_validates_id():
    StrategySpec(
        strategy_id="sma_20_100",
        name="x",
        class_path="strategies.templates.sma_crossover:SmaCrossover",
        strategy_class="trend_following",
        universe="us_core_etfs",
        timeframe=Timeframe.DAILY,
    )
    with pytest.raises(Exception):
        StrategySpec(
            strategy_id="Bad ID!",
            name="x",
            class_path="strategies.templates.sma_crossover:SmaCrossover",
            strategy_class="trend_following",
            universe="us_core_etfs",
            timeframe=Timeframe.DAILY,
        )


def test_backtest_config_defaults():
    c = BacktestConfig()
    assert c.fees_bps == 2.0
    assert c.slippage_bps == 5.0
    assert c.size_type == "percent"


def test_knowledge_note_markdown_roundtrip():
    n = KnowledgeNote(
        folder="02 Strategien",
        title="Test Strat",
        frontmatter={"strategy_id": "abc", "status": "draft"},
        tags=["test"],
        body="## Idee\nfoo",
    )
    md = n.to_markdown()
    assert md.startswith("---\n")
    assert "strategy_id: abc" in md
    assert "tags:" in md
    assert "## Idee" in md

    # Ohne `purpose` ist eine Note ein Testlauf und landet unter `_smoke/`
    # (vault_layout). Research ist eine bewusste Angabe — siehe
    # tests/test_vault_layout.py für die vollständige Begründung.
    assert n.path == "Trading Research/_smoke/02 Strategien/Test Strat.md"
    assert (
        n.model_copy(update={"purpose": "research"}).path
        == "Trading Research/02 Strategien/Test Strat.md"
    )


def test_strategy_status_enum():
    assert StrategyStatus.APPROVED.value == "approved"


# -----------------------------------------------------------------------------
# Datenabdeckung: angefordert gegen tatsächlich (#307)
# -----------------------------------------------------------------------------


def test_coverage_meldet_ein_gekapptes_fenster(synthetic_md):
    """Der Fall, der am 2026-08-30 unbemerkt durchlief: ein Sweep über
    2000–2019 auf Daten, die 2015 endeten."""
    from datetime import date

    md = synthetic_md.model_copy(update={"end": date(2030, 1, 1)})
    cov = md.coverage

    assert cov.requested_end == date(2030, 1, 1)
    assert cov.actual_end == synthetic_md.frame.index[-1].date()
    assert cov.truncated_end and not cov.truncated_start
    assert not cov.complete
    fehlt = cov.shortfall()
    assert fehlt and "endet schon" in fehlt and "2030-01-01" in fehlt


def test_coverage_zaehlt_fehlende_symbole_mit(synthetic_md):
    """Ein Korb aus 16 ETFs, von denen 15 Daten haben, ist ein anderer Korb.
    Bis hierher stand das nur im Log des Loaders."""
    md = synthetic_md.model_copy(update={"missing_symbols": ["XLRE"]})
    cov = md.coverage

    assert not cov.complete
    assert cov.n_symbols_requested == len(synthetic_md.symbols) + 1
    assert cov.n_symbols_loaded == len(synthetic_md.symbols)
    assert "XLRE" in (cov.shortfall() or "")


def test_coverage_traegt_den_ausschluss_ins_ergebnis(synthetic_md):
    """Ein verworfenes Symbol gehört ins JSON, nicht nur in den CLI-Ausdruck.

    Seit #322 wirft ein kaputter Bar ein Symbol statt des Universums heraus,
    und seit #312 tut ein defekter Aktionseintrag dasselbe. Beides ist genau
    die Art von Auskunft, für die `DataCoverage` gebaut wurde: gerechnet wurde
    auf weniger als angefordert, und wer die Zahl später liest, muss das sehen
    können — ohne das Log von damals.
    """
    md = synthetic_md.model_copy(
        update={"unusable_symbols": {"PNLYY": "high_lt_low: 1 Bars mit high < low"}}
    )
    cov = md.coverage

    assert not cov.complete
    assert cov.unusable_symbols == {"PNLYY": "high_lt_low: 1 Bars mit high < low"}
    fehlt = cov.shortfall() or ""
    assert "PNLYY" in fehlt
    assert "high_lt_low" in fehlt, "mit Grund — ein Name allein schickt in den Lake"


def test_coverage_zaehlt_verworfene_zur_anforderung(synthetic_md):
    """Sonst schrumpft die Anforderung still mit jedem Ausschluss.

    Ein verworfenes Symbol steht weder in `symbols` noch in `missing_symbols`.
    Ohne den dritten Summanden läse sich der Lauf als „950 von 950" — und
    hätte über ein Papier hinweggelesen, das da war und verworfen wurde.
    """
    md = synthetic_md.model_copy(update={"unusable_symbols": {"PNLYY": "high_lt_low"}})

    assert md.coverage.n_symbols_requested == len(synthetic_md.symbols) + 1
    assert md.coverage.n_symbols_loaded == len(synthetic_md.symbols)


def test_die_beiden_luecken_werden_nicht_vermischt(synthetic_md):
    """„Nie geladen" und „geladen und verworfen" bleiben getrennte Zeilen.

    In einen Topf geworfen wäre keine der beiden Auskünfte noch etwas wert:
    die eine schickt in den Lake, die andere in den Actions-Feed.
    """
    md = synthetic_md.model_copy(
        update={"missing_symbols": ["XLRE"], "unusable_symbols": {"WY": "unadjustable_action"}}
    )
    cov = md.coverage

    assert cov.missing_symbols == ["XLRE"]
    assert "XLRE" not in cov.unusable_symbols
    assert cov.n_symbols_requested == len(synthetic_md.symbols) + 2
    fehlt = cov.shortfall() or ""
    assert "ohne Daten" in fehlt and "verworfen" in fehlt


def test_coverage_schweigt_wenn_alles_da_ist(synthetic_md):
    """`shortfall()` ist keine Statuszeile, sondern ein Vorbehalt — ohne
    Vorbehalt gibt es nichts zu sagen."""
    cov = synthetic_md.coverage

    assert cov.complete
    assert cov.shortfall() is None
