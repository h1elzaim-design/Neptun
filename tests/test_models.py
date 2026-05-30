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
    assert n.path == "Trading Research/02 Strategien/Test Strat.md"


def test_strategy_status_enum():
    assert StrategyStatus.APPROVED.value == "approved"
