"""Per-Asset-Class-Kosten: Resolver (config/costs.yaml) + Runner-Integration.

Der Resolver klassifiziert Symbole (Override > Klasse > default_class); der
Runner baut daraus per-Spalte-Fees/Slippage-Arrays mit effektiver Slippage
= slippage + spread/2 und persistiert die aufgelöste Tabelle in der Config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantrace.costs import DEFAULT_COSTS_PATH, resolve_symbol_costs
from quantrace.models import BacktestConfig, StrategySpec, SymbolCosts, Timeframe

# --- Resolver gegen das echte config/costs.yaml -----------------------------------


def test_known_symbols_resolve_to_their_class():
    costs = resolve_symbol_costs(["SPY", "GLD", "TLT", "AAPL", "EEM"])
    assert costs["SPY"].asset_class == "equity_index_etf"
    assert costs["GLD"].asset_class == "commodity_etf"
    assert costs["TLT"].asset_class == "bond_etf"
    assert costs["AAPL"].asset_class == "us_equity_single"
    assert costs["EEM"].asset_class == "intl_equity_etf"


def test_symbol_override_wins_over_class():
    costs = resolve_symbol_costs(["GLD", "DBC"])
    # DBC hat ein Voll-Override (breiter Rohstoffkorb) — teurer als GLD.
    assert costs["DBC"].spread_bps > costs["GLD"].spread_bps
    assert costs["DBC"].total_per_side_bps > costs["GLD"].total_per_side_bps


def test_unknown_symbol_falls_back_to_default_class():
    costs = resolve_symbol_costs(["ZZZTEST"])
    assert costs["ZZZTEST"].asset_class == "equity_index_etf"  # default_class


def test_all_universe_symbols_are_classified():
    """Jedes Symbol aus data/universes/ muss explizit klassifiziert sein —
    ein Fallback wäre eine stille Fehlkalkulation der Kosten."""
    import yaml

    universes_dir = Path(DEFAULT_COSTS_PATH).parents[1] / "data" / "universes"
    mapped = set((yaml.safe_load(DEFAULT_COSTS_PATH.read_text()) or {}).get("symbols", {}))
    unmapped = {
        sym
        for path in universes_dir.glob("*.yaml")
        for sym in (yaml.safe_load(path.read_text()) or {}).get("symbols", [])
        if sym not in mapped
    }
    assert not unmapped, f"Symbole ohne Kosten-Klassifikation: {sorted(unmapped)}"


def test_effective_slippage_includes_half_spread():
    sc = SymbolCosts(asset_class="x", fees_bps=1.0, slippage_bps=2.0, spread_bps=4.0)
    assert sc.effective_slippage_bps == pytest.approx(4.0)  # 2 + 4/2
    assert sc.total_per_side_bps == pytest.approx(5.0)


def test_broken_costs_yaml_raises(tmp_path: Path):
    bad = tmp_path / "costs.yaml"
    bad.write_text("default_class: nope\nclasses:\n  a: {fees_bps: 1, slippage_bps: 1, spread_bps: 1}\n")
    with pytest.raises(ValueError, match="default_class"):
        resolve_symbol_costs(["SPY"], config_path=bad)

    bad2 = tmp_path / "costs2.yaml"
    bad2.write_text(
        "default_class: a\n"
        "classes:\n  a: {fees_bps: 1, slippage_bps: 1, spread_bps: 1}\n"
        "symbols:\n  SPY: ghost_class\n"
    )
    with pytest.raises(ValueError, match="unbekannte Klassen"):
        resolve_symbol_costs(["SPY"], config_path=bad2)


# --- Runner-Integration -------------------------------------------------------------


def _spec() -> StrategySpec:
    return StrategySpec(
        strategy_id="sma_cost_test",
        name="SMA Cost Test",
        class_path="strategies.templates.sma_crossover:SmaCrossover",
        strategy_class="trend_following",
        universe="synthetic",
        timeframe=Timeframe.DAILY,
        params={"fast": 10, "slow": 50},
    )


def test_per_asset_class_resolves_and_persists_table(synthetic_md):
    from quantrace.backtest_runner import run_backtest

    result = run_backtest(_spec(), synthetic_md, BacktestConfig(cost_model="per_asset_class"))

    table = result.config.symbol_costs
    assert table is not None and set(table) == {"SPY", "QQQ"}
    assert all(sc.asset_class == "equity_index_etf" for sc in table.values())
    # JSON-Roundtrip: die Kosten-Annahmen stehen im persistierten Ergebnis.
    dumped = result.model_dump(mode="json", exclude={"equity_curve"})
    assert dumped["config"]["cost_model"] == "per_asset_class"
    assert dumped["config"]["symbol_costs"]["SPY"]["fees_bps"] == 0.5


def test_flat_model_unchanged_and_default(synthetic_md):
    from quantrace.backtest_runner import run_backtest

    result = run_backtest(_spec(), synthetic_md, BacktestConfig())
    assert result.config.cost_model == "flat"
    assert result.config.symbol_costs is None


def test_higher_costs_hurt_performance(synthetic_md):
    """Per-Spalte-Kosten wirken wirklich: ein teures Override auf beide Symbole
    drückt Total-Return gegenüber einem billigen."""
    from quantrace.backtest_runner import run_backtest

    def table(bps: float) -> dict[str, SymbolCosts]:
        return {
            s: SymbolCosts(asset_class="t", fees_bps=bps, slippage_bps=bps, spread_bps=bps)
            for s in ("SPY", "QQQ")
        }

    cheap = run_backtest(
        _spec(), synthetic_md,
        BacktestConfig(cost_model="per_asset_class", symbol_costs=table(0.1)),
    )
    expensive = run_backtest(
        _spec(), synthetic_md,
        BacktestConfig(cost_model="per_asset_class", symbol_costs=table(50.0)),
    )
    assert expensive.total_return < cheap.total_return


def test_presupplied_table_must_cover_all_symbols(synthetic_md):
    from quantrace.backtest_runner import run_backtest

    partial = {"SPY": SymbolCosts(asset_class="t", fees_bps=1, slippage_bps=1, spread_bps=1)}
    with pytest.raises(ValueError, match="QQQ"):
        run_backtest(
            _spec(), synthetic_md,
            BacktestConfig(cost_model="per_asset_class", symbol_costs=partial),
        )


def test_old_config_json_without_cost_fields_parses():
    cfg = BacktestConfig.model_validate({"cash": 50_000.0, "fees_bps": 2.0, "slippage_bps": 5.0})
    assert cfg.cost_model == "flat"
    assert cfg.symbol_costs is None
