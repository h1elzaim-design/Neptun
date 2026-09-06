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


# --- Der Spread skaliert mit dem Kursniveau (#323) --------------------------------


class TestSpreadUntergrenze:
    """Die Tickgrösse ist eine Marktregel, keine Annahme.

    **Der Anlass.** Gemessen am 2026-09-05 über `us_top500_liquid`: 0,5–0,7 %
    der Kurszellen tragen 99,7 % der Rendite, und es sind durchweg insolvente
    Papiere kurz vor dem Delisting. `buy_and_hold` kam über 2007–2012 auf
    **CAGR +2.333 % bei 99 % Drawdown**. Beides zusammen gibt es nicht — also
    hat keine der beiden Zahlen gemessen, was sie behauptet.

    Die Kostenklassen stehen in bps pro Symbol und gelten für den ganzen
    Backtest. Das stimmt, solange ein Papier auf normalem Niveau handelt.
    Fällt es auf zwei Zehntelcent, ist eine Bewegung von 0,0002 auf 0,0003
    **+50 %** und dabei genau ein Tick.

    SEC Rule 612 schreibt die Mindestpreisschritte vor: ein Cent ab 1,00 $, ein
    Hundertstelcent darunter. Wer über den Spread handelt, zahlt mindestens
    einen halben davon. Das ist die eine Zahl in `costs.py`, die nicht
    geschätzt ist — der Rest der Datei sagt selbst, dass er es ist.
    """

    @pytest.mark.parametrize(
        "preis,erwartet_bps",
        [
            (100.0, 0.5),      # ein Tick ist hier bedeutungslos
            (10.0, 5.0),
            (5.0, 10.0),
            (1.0, 50.0),       # genau an der Sub-Penny-Grenze
            (0.02, 25.0),      # darunter gilt der kleinere Tick
            (0.001, 500.0),
            (0.0002, 2500.0),  # 25 % je Seite — und das ist die Untergrenze
        ],
    )
    def test_die_gerechneten_stufen(self, preis, erwartet_bps):
        from quantrace.costs import spread_untergrenze_bps

        assert float(spread_untergrenze_bps(preis)) == pytest.approx(erwartet_bps)

    def test_der_tickwechsel_sitzt_bei_einem_dollar(self):
        """Ohne den Sprung wären Papiere knapp unter einem Dollar mit einem
        Cent-Tick bepreist, den es dort nicht gibt."""
        from quantrace.costs import spread_untergrenze_bps

        assert float(spread_untergrenze_bps(0.999)) < float(spread_untergrenze_bps(1.001))
        assert float(spread_untergrenze_bps(1.001)) == pytest.approx(49.95, rel=1e-3)

    def test_sie_kann_kosten_nur_erhoehen(self):
        """Dieselbe Konvention wie bei `dollar_volume`: was geschätzt ist, wird
        so geschätzt, dass der Fehler gegen die Strategie läuft."""
        import numpy as np

        from quantrace.costs import spread_untergrenze_bps

        assert (spread_untergrenze_bps(np.array([0.0001, 0.01, 1.0, 50.0, 5000.0])) >= 0).all()

    def test_ein_nicht_positiver_kurs_ergibt_keine_unendlichkeit(self):
        """Im Lesepfad gibt es die seit #322/#324 nicht mehr — käme doch einer
        durch, wäre eine unendliche Kostenzahl schlimmer als keine."""
        import numpy as np

        from quantrace.costs import spread_untergrenze_bps

        raus = spread_untergrenze_bps(np.array([0.0, -1.0, float("nan")]))
        assert np.isfinite(raus).all() and (raus == 0.0).all()

    def test_ein_ganzer_rahmen_geht_zellenweise_durch(self):
        """Genau das ist der Unterschied zur Klassenzahl: die steht pro Symbol
        fest, und ein Papier, das von 40 $ auf 2 Cent fällt, behält sie."""
        import numpy as np

        from quantrace.costs import spread_untergrenze_bps

        rahmen = np.array([[40.0, 0.5], [4.0, 0.05], [0.02, 0.005]])
        raus = spread_untergrenze_bps(rahmen)
        assert raus.shape == rahmen.shape
        assert raus[0, 0] < raus[2, 0], "je tiefer der Kurs, desto teurer"
        assert raus[2, 0] == pytest.approx(25.0)


class TestUntergrenzeImRunner:
    """Sie muss beim Backtest ankommen, nicht nur richtig gerechnet sein."""

    def _md(self, preise: list[list[float]]):
        import pandas as pd

        from quantrace.models import MarketData, Timeframe

        idx = pd.date_range("2020-01-01", periods=len(preise), freq="B")
        rahmen = pd.concat(
            {
                sym: pd.DataFrame(
                    {"open": sp, "high": sp, "low": sp, "close": sp, "volume": [1e6] * len(sp)},
                    index=idx,
                )
                for sym, sp in zip(("TEUER", "CENT"), zip(*preise, strict=True), strict=True)
            },
            axis=1,
        )
        rahmen.columns.names = ["symbol", "field"]
        return MarketData(
            universe="probe",
            provider="eodhd",
            symbols=["TEUER", "CENT"],
            timeframe=Timeframe.DAILY,
            start=idx[0].date(),
            end=idx[-1].date(),
            calendar="us_equity",
            frame=rahmen,
        )

    def test_das_cent_papier_bekommt_mehr_slippage_als_das_teure(self):
        from quantrace.backtest_runner import _cost_inputs

        md = self._md([[100.0, 0.002]] * 5)
        _fees, slippage, _cfg = _cost_inputs(
            md.frame.xs("close", level="field", axis=1),
            BacktestConfig(cost_model="per_asset_class"),
            md,
        )
        assert slippage.shape == (5, 2), "pro Zelle, nicht pro Spalte"
        assert slippage["CENT"].iloc[0] > slippage["TEUER"].iloc[0] * 100

    def test_ein_papier_das_faellt_wird_unterwegs_teurer(self):
        """Der eigentliche Punkt: die Klassenzahl steht fest, der Kurs nicht.

        Ein Papier, das von 40 $ auf zwei Zehntelcent fällt, ist am Ende ein
        anderes Instrument als am Anfang — und wurde bis heute so bepreist wie
        am Anfang.
        """
        from quantrace.backtest_runner import _cost_inputs

        md = self._md([[10.0, 40.0], [10.0, 4.0], [10.0, 0.4], [10.0, 0.0002]])
        _f, slippage, _c = _cost_inputs(
            md.frame.xs("close", level="field", axis=1),
            BacktestConfig(cost_model="per_asset_class"),
            md,
        )
        verlauf = slippage["CENT"].tolist()
        assert verlauf[-1] > verlauf[0] * 1000, "am Boden ist es tausendfach teurer"
        assert slippage["TEUER"].nunique() == 1, "ein stabiler Kurs, eine Zahl"

    def test_unterhalb_eines_dollars_sinkt_die_untergrenze_wieder(self):
        """Nicht monoton — und das ist die Regel, nicht ein Fehler.

        SEC Rule 612 erlaubt unter 1,00 $ einen hundertfach kleineren Tick::

            4,00 $  →  Tick 0,01 $     →  12,50 bps
            0,40 $  →  Tick 0,0001 $   →   1,25 bps

        Ein 40-Cent-Papier hat also einen *kleineren* Mindestspread als ein
        4-Dollar-Papier. Wer hier Monotonie erwartet, hat die Marktregel durch
        eine Intuition ersetzt — und genau das soll diese Datei verhindern.

        Der Steilanstieg kommt erst darunter: bei 0,0002 $ sind es 2.500 bps.
        """
        from quantrace.costs import spread_untergrenze_bps

        assert float(spread_untergrenze_bps(4.0)) == pytest.approx(12.5)
        assert float(spread_untergrenze_bps(0.4)) == pytest.approx(1.25)
        assert float(spread_untergrenze_bps(0.0002)) == pytest.approx(2500.0)

    def test_flat_bleibt_flat(self):
        """`cost_model="flat"` reproduziert weiter exakt das alte Verhalten —
        sonst wäre jeder Altvergleich stillschweigend kaputt."""
        from quantrace.backtest_runner import _cost_inputs

        md = self._md([[100.0, 0.002]] * 3)
        fees, slippage, _c = _cost_inputs(
            md.frame.xs("close", level="field", axis=1), BacktestConfig(cost_model="flat"), md
        )
        assert isinstance(fees, float) and isinstance(slippage, float)
