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
    """Jedes Symbol aus data/universes/ muss klassifiziert sein — ein stiller
    Rückfall wäre eine Fehlkalkulation der Kosten.

    **Die Ausnahme und warum sie die Zusage nicht aufweicht.** Ein Universum
    mit ``cost_class:`` deklariert seine Klasse für alle Mitglieder auf einmal.
    Das ist nur für *konstruierte* Universen zulässig — die mit einem
    ``construction:``-Block, deren Mitglieder eine Liquiditätsregel bestimmt
    hat. Dort **ist** die Regel die Klassifikation: wer eine Untergrenze fürs
    Dollarvolumen setzt, hat über Spread und Impact bereits entschieden.

    Eine handverlesene Liste hat dieses Argument nicht — dort hat ein Mensch
    jedes Symbol einzeln gewählt und kann es einzeln einordnen. Der Test
    besteht deshalb auf beidem: ``cost_class`` **und** ``construction``.
    """
    import yaml

    universes_dir = Path(DEFAULT_COSTS_PATH).parents[1] / "data" / "universes"
    costs_raw = yaml.safe_load(DEFAULT_COSTS_PATH.read_text()) or {}
    mapped = set(costs_raw.get("symbols", {}))
    klassen = set(costs_raw.get("classes", {}))

    unmapped: dict[str, list[str]] = {}
    falsch_deklariert: list[str] = []
    for path in universes_dir.glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text()) or {}
        cost_class = cfg.get("cost_class")
        if cost_class is not None:
            if not cfg.get("construction"):
                falsch_deklariert.append(
                    f"{path.name}: cost_class ohne construction-Block"
                )
            elif cost_class not in klassen:
                falsch_deklariert.append(
                    f"{path.name}: cost_class '{cost_class}' ist keine Klasse in costs.yaml"
                )
            continue
        fehlend = [s for s in cfg.get("symbols", []) if s not in mapped]
        if fehlend:
            unmapped[path.name] = sorted(fehlend)

    assert not falsch_deklariert, f"Unzulässige cost_class-Angaben: {falsch_deklariert}"
    assert not unmapped, f"Symbole ohne Kosten-Klassifikation: {unmapped}"


def test_fallback_class_greift_nur_fuer_unklassifizierte():
    """Ein deklariertes `cost_class` überschreibt keine eigene Zuordnung.

    Sonst bekäme SPY in einem konstruierten Universum Small-Cap-Spreads —
    und das wäre wieder eine Zahl, die niemand nachvollziehen kann.
    """
    costs = resolve_symbol_costs(
        ["SPY", "ZZZTEST"], fallback_class="us_equity_smallcap"
    )
    assert costs["SPY"].asset_class == "equity_index_etf"
    assert costs["ZZZTEST"].asset_class == "us_equity_smallcap"


def test_unbekannte_fallback_klasse_wird_abgelehnt():
    with pytest.raises(ValueError, match="fallback_class"):
        resolve_symbol_costs(["SPY"], fallback_class="gibts_nicht")


class TestKlasseAusLiquiditaet:
    """Der Boden bestimmt die Klasse — nicht der Durchschnitt, nicht die Zusage."""

    @pytest.mark.parametrize(
        ("volumen", "erwartet"),
        [
            (500_000_000.0, "us_equity_single"),
            (25_000_000.0, "us_equity_single"),
            (24_999_999.0, "us_equity_liquid"),
            (5_000_000.0, "us_equity_liquid"),
            (1_000_000.0, "us_equity_smallcap"),
        ],
    )
    def test_schwellen(self, volumen, erwartet):
        from quantrace.costs import class_for_liquidity

        assert class_for_liquidity(volumen) == erwartet

    def test_zu_duenn_ist_ein_fehler_keine_grosse_zahl(self):
        """Eine erfundene bps-Zahl wäre schlimmer als eine Fehlermeldung."""
        from quantrace.costs import UnpriceableError, class_for_liquidity

        with pytest.raises(UnpriceableError, match="keine feste bps-Zahl"):
            class_for_liquidity(200_000.0)

    def test_jede_klasse_existiert_wirklich_in_costs_yaml(self):
        """Sonst wirft der Resolver erst beim ersten echten Backtest."""
        import yaml

        from quantrace.costs import _LIQUIDITY_CLASSES

        klassen = set((yaml.safe_load(DEFAULT_COSTS_PATH.read_text()) or {})["classes"])
        assert {k for _, k in _LIQUIDITY_CLASSES} <= klassen

    def test_teurer_je_duenner(self):
        """Die Ordnung ist der eigentliche Inhalt der drei Klassen."""
        from quantrace.costs import _LIQUIDITY_CLASSES

        klassen = [k for _, k in _LIQUIDITY_CLASSES]
        profile = resolve_symbol_costs(["X"], fallback_class=klassen[0])["X"]
        vorher = profile.total_per_side_bps
        for klasse in klassen[1:]:
            jetzt = resolve_symbol_costs(["X"], fallback_class=klasse)["X"]
            assert jetzt.total_per_side_bps > vorher
            vorher = jetzt.total_per_side_bps


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
