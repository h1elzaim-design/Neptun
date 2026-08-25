"""Capital-model semantics of the backtest runner (ADR-003).

``capital_model="shared"`` must behave like ONE account: capital competes for
a single pool, is split equally across the currently-active positions, stays
fully invested up to ``size`` (k=1 → that one position gets ~95%), rebalances
only when signal membership changes, and goes to cash when nothing is active.
``capital_model="independent"`` is the documented legacy rollback: every
column backtests with the full init_cash, equity = mean of the sleeves.

The end-to-end scenarios use zero costs and ``execution_lag=0`` so expected
equities can be computed by hand and asserted exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantrace.backtest_runner import _held_mask, _shared_target_sizes, run_inline
from quantrace.models import BacktestConfig, MarketData, Timeframe

CASH = 100_000.0


# -----------------------------------------------------------------------------
# Fixtures / helpers
# -----------------------------------------------------------------------------

class ScriptedStrategy:
    """Strategy stub that replays pre-scripted entry/exit matrices."""

    def __init__(self, entries: pd.DataFrame, exits: pd.DataFrame) -> None:
        self._entries = entries
        self._exits = exits

    def generate_signals(self, data: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._entries, self._exits


def _md(prices: dict[str, np.ndarray]) -> MarketData:
    """MarketData from scripted close paths (OHLC collapsed onto close)."""
    n = len(next(iter(prices.values())))
    idx = pd.bdate_range("2021-01-01", periods=n)
    frames = {
        s: pd.DataFrame(
            {
                "open": p, "high": p, "low": p, "close": p,
                "volume": np.full(n, 1_000_000.0),
            },
            index=idx,
        )
        for s, p in prices.items()
    }
    combined = pd.concat(frames, axis=1)
    combined.columns.names = ["symbol", "field"]
    return MarketData(
        universe="scripted",
        symbols=list(prices),
        timeframe=Timeframe.DAILY,
        start=idx[0].date(),
        end=idx[-1].date(),
        provider="synthetic",
        frame=combined,
    )


def _signals(index, columns, on: dict[str, list[int]]) -> pd.DataFrame:
    df = pd.DataFrame(False, index=index, columns=columns)
    for col, bars in on.items():
        df.iloc[bars, df.columns.get_loc(col)] = True
    return df


def _config(capital_model: str) -> BacktestConfig:
    # Zero costs + lag opt-out → hand-computable equities. The look-ahead lag
    # has its own dedicated tests; here it would only shift the script by a bar.
    return BacktestConfig(
        fees_bps=0.0, slippage_bps=0.0, execution_lag=0, capital_model=capital_model
    )


def _run(md: MarketData, entries, exits, capital_model: str):
    return run_inline(
        "scripted", ScriptedStrategy(entries, exits), md, _config(capital_model)
    )


@pytest.fixture
def staggered_md() -> MarketData:
    """A: flat 100 until entry, doubles to 200 by bar 10, flat after.
    B: flat 100 throughout. 16 bars."""
    a = np.concatenate([[100.0, 100.0], np.linspace(112.5, 200.0, 8), [200.0] * 6])
    b = np.full(16, 100.0)
    return _md({"A": a, "B": b})


# -----------------------------------------------------------------------------
# Pure helpers: held mask + target sizes
# -----------------------------------------------------------------------------

class TestHeldMask:
    def test_entry_persists_until_exit(self):
        idx = pd.date_range("2021-01-01", periods=6)
        entries = _signals(idx, ["A"], {"A": [1]})
        exits = _signals(idx, ["A"], {"A": [4]})
        held = _held_mask(entries, exits)
        assert held["A"].tolist() == [False, True, True, True, False, False]

    def test_exit_wins_on_simultaneous_bar(self):
        idx = pd.date_range("2021-01-01", periods=4)
        entries = _signals(idx, ["A"], {"A": [1]})
        exits = _signals(idx, ["A"], {"A": [1]})
        held = _held_mask(entries, exits)
        assert not held["A"].any()  # in doubt, flat

    def test_starts_flat_without_signal(self):
        idx = pd.date_range("2021-01-01", periods=3)
        empty = _signals(idx, ["A", "B"], {})
        assert not _held_mask(empty, empty).any().any()


class TestSharedTargetSizes:
    def test_orders_only_on_membership_change(self):
        idx = pd.date_range("2021-01-01", periods=8)
        entries = _signals(idx, ["A", "B"], {"A": [1], "B": [1]})
        exits = _signals(idx, ["A", "B"], {})
        size = _shared_target_sizes(_held_mask(entries, exits), 0.95)
        # Entry bar: both get 0.95/2; every other bar: NaN everywhere (drift).
        assert size.iloc[1].tolist() == pytest.approx([0.475, 0.475])
        assert size.drop(index=idx[1]).isna().all().all()

    def test_membership_change_reslices_survivors_and_zeroes_leavers(self):
        idx = pd.date_range("2021-01-01", periods=6)
        entries = _signals(idx, ["A", "B", "C"], {"A": [1], "B": [1]})
        exits = _signals(idx, ["A", "B", "C"], {"B": [3]})
        size = _shared_target_sizes(_held_mask(entries, exits), 0.95)
        # Bar 3: B exits → target 0; A re-slices to full 0.95; C never held → NaN.
        assert size.iloc[3]["B"] == 0.0
        assert size.iloc[3]["A"] == pytest.approx(0.95)
        assert np.isnan(size.iloc[3]["C"])

    def test_all_exit_targets_zero(self):
        idx = pd.date_range("2021-01-01", periods=5)
        entries = _signals(idx, ["A", "B"], {"A": [1], "B": [1]})
        exits = _signals(idx, ["A", "B"], {"A": [3], "B": [3]})
        size = _shared_target_sizes(_held_mask(entries, exits), 0.95)
        assert size.iloc[3].tolist() == pytest.approx([0.0, 0.0])


# -----------------------------------------------------------------------------
# End-to-end account semantics (needs vectorbt)
# -----------------------------------------------------------------------------

class TestSharedAccount:
    def test_single_active_position_is_fully_invested(self, staggered_md):
        """k=1 → the lone active position gets ~size, not size/n_symbols.

        A enters alone at bar 1 @100 and doubles: with 95% deployed the account
        must end at 1.95× cash. Under fixed 1/N slots it would only be 1.475×.
        """
        idx = staggered_md.frame.index
        cols = ["A", "B"]
        entries = _signals(idx, cols, {"A": [1], "B": [10]})
        exits = _signals(idx, cols, {})
        res = _run(staggered_md, entries, exits, "shared")
        assert float(res.equity_curve.iloc[-1]) == pytest.approx(1.95 * CASH, rel=1e-9)

    def test_late_entrant_shares_the_pool_not_fresh_capital(self, staggered_md):
        """Shared vs independent differ exactly by B's phantom second account.

        Independent: sleeve A 1.95×, sleeve B (own full 100k, flat) 1.0× →
        mean 1.475×. Shared ends 1.95× (see above) — the gap is the audit
        finding made executable.
        """
        idx = staggered_md.frame.index
        cols = ["A", "B"]
        entries = _signals(idx, cols, {"A": [1], "B": [10]})
        exits = _signals(idx, cols, {})
        res_ind = _run(staggered_md, entries, exits, "independent")
        assert float(res_ind.equity_curve.iloc[-1]) == pytest.approx(1.475 * CASH, rel=1e-9)

    def test_simultaneous_entries_split_equally_not_by_column_order(self):
        """3 identical symbols entering together must NOT give 95/4.75/0.24.

        A doubles, B and C stay flat: equal split (0.95/3 each) ends at
        cash·(1 + 0.95/3). Column-order hogging (95% into A) would end ~1.95×.
        """
        n = 10
        a = np.concatenate([[100.0, 100.0], np.linspace(112.5, 200.0, n - 2)])
        md = _md({"A": a, "B": np.full(n, 100.0), "C": np.full(n, 100.0)})
        idx = md.frame.index
        cols = ["A", "B", "C"]
        entries = _signals(idx, cols, {c: [1] for c in cols})
        exits = _signals(idx, cols, {})
        res = _run(md, entries, exits, "shared")
        expected = CASH * (1.0 + 0.95 / 3.0)
        assert float(res.equity_curve.iloc[-1]) == pytest.approx(expected, rel=1e-9)

    def test_no_active_positions_means_cash(self):
        """After the last exit the account sits in cash — equity goes flat."""
        n = 12
        a = np.linspace(100.0, 220.0, n)  # keeps rising AFTER the exit
        md = _md({"A": a, "B": np.full(n, 100.0)})
        idx = md.frame.index
        cols = ["A", "B"]
        entries = _signals(idx, cols, {"A": [1], "B": [1]})
        exits = _signals(idx, cols, {"A": [5], "B": [5]})
        res = _run(md, entries, exits, "shared")
        eq = res.equity_curve
        post_exit = eq.iloc[5:]
        assert post_exit.nunique() == 1  # flat: no exposure, no drift
        assert res.trades.n_trades == 2  # two closed round trips, no extras

    def test_drift_without_signal_change_produces_no_extra_trades(self, staggered_md):
        """Weights drifting (A doubles vs B flat) must not trigger rebalances."""
        idx = staggered_md.frame.index
        cols = ["A", "B"]
        entries = _signals(idx, cols, {"A": [1], "B": [1]})
        exits = _signals(idx, cols, {})
        res = _run(staggered_md, entries, exits, "shared")
        # 2 entries, no exits, no signal change afterwards → exactly 2 trades
        # (both still open). Daily re-slicing would multiply this number.
        assert res.trades.n_trades == 2


class TestRollbackAndEquivalence:
    def test_single_symbol_shared_equals_independent(self):
        n = 10
        md = _md({"A": np.linspace(100.0, 150.0, n)})
        idx = md.frame.index
        entries = _signals(idx, ["A"], {"A": [1]})
        exits = _signals(idx, ["A"], {"A": [7]})
        eq_shared = _run(md, entries, exits, "shared").equity_curve
        eq_ind = _run(md, entries, exits, "independent").equity_curve
        pd.testing.assert_series_equal(eq_shared, eq_ind)

    def test_default_capital_model_is_shared(self):
        assert BacktestConfig().capital_model == "shared"


# -----------------------------------------------------------------------------
# Was keinen Kurs hat, kann nicht gehalten werden (#255)
# -----------------------------------------------------------------------------


class TestUntradable:
    """Ein Papier ohne Kurs muss die Position verlassen — nicht sie einfrieren.

    Der Fall entsteht auf zwei Wegen, die im Rahmen identisch aussehen: eine
    Pleite im survivorship-freien Lake und ein Ausscheiden aus einem
    rekonstituierten Universum. Beide enden in einer Spalte, die ab einem Bar
    nur noch ``NaN`` trägt.

    Ohne Gegenmaßnahme feuert keine Strategie einen Exit (jeder Vergleich mit
    ``NaN`` ist ``False``), ``_held_mask`` liest nur Signale, und
    ``from_orders`` kann bei ``NaN`` nicht verkaufen: die Position überlebt das
    Papier und bindet Kapital zum letzten bekannten Kurs. Die Equity-Kurve
    läuft dabei weiter — der Fehler sieht aus wie ein Ergebnis.
    """

    @staticmethod
    def _szenario() -> tuple[MarketData, pd.DataFrame, pd.DataFrame]:
        n = 40
        a = np.full(n, 100.0)
        a[20:] = np.linspace(100.0, 200.0, n - 20)  # A verdoppelt sich ab Bar 20
        b = np.full(n, 50.0)
        b[20:] = np.nan  # B handelt ab Bar 20 nicht mehr

        md = _md({"A": a, "B": b})
        idx = md.frame.index
        entries = _signals(idx, ["A", "B"], {"A": [2], "B": [2]})
        exits = _signals(idx, ["A", "B"], {})
        return md, entries, exits

    def test_kapital_wird_frei_und_arbeitet_weiter(self):
        """Von Hand: 95.000 in A ab Bar 20, A verdoppelt → 195.000 Endwert.

        Ohne den Zwangs-Exit blieben 47.500 in B eingefroren; A käme nur auf
        95.000 und das Buch auf 147.500 — eine Gesamtrendite von 0,475 statt
        0,95. Exakt die Hälfte des Ergebnisses, ohne Fehlermeldung.
        """
        md, entries, exits = self._szenario()
        res = _run(md, entries, exits, "shared")

        assert res.total_return == pytest.approx(0.95, abs=1e-3)

    def test_der_exit_faellt_auf_den_ersten_bar_ohne_kurs(self):
        from quantrace.backtest_runner import _close_untradable

        md, entries, exits = self._szenario()
        close = md.frame.xs("close", level="field", axis=1)
        _, e2, x2 = _close_untradable(close, entries, exits)

        assert bool(x2["B"].iloc[20]), "Exit genau dort, wo der Kurs verschwindet"
        assert not x2["B"].iloc[19], "und keinen Bar früher — das wäre Vorwissen"
        assert not x2["A"].any(), "A ist unberührt"

    def test_ohne_kurs_wird_nicht_eingestiegen(self):
        from quantrace.backtest_runner import _close_untradable

        md, _, exits = self._szenario()
        close = md.frame.xs("close", level="field", axis=1)
        spaet = _signals(close.index, ["A", "B"], {"B": [25]})  # Einstieg ins Nichts

        _, e2, _ = _close_untradable(close, spaet, exits)
        assert not e2["B"].any(), "eine Order auf NaN wird nie gefüllt"

    def test_ausgefuehrt_wird_zum_letzten_bekannten_kurs(self):
        from quantrace.backtest_runner import _close_untradable

        md, entries, exits = self._szenario()
        close = md.frame.xs("close", level="field", axis=1)
        c2, _, _ = _close_untradable(close, entries, exits)

        assert c2["B"].iloc[20] == pytest.approx(50.0)
        assert c2["B"].notna().all(), "ffill, damit der Verkauf einen Preis hat"

    def test_ein_vollstaendiger_rahmen_bleibt_unangetastet(self):
        """Die Regel darf im Normalfall nichts kosten und nichts ändern."""
        from quantrace.backtest_runner import _close_untradable

        md = _md({"A": np.full(10, 100.0), "B": np.full(10, 50.0)})
        close = md.frame.xs("close", level="field", axis=1)
        entries = _signals(close.index, ["A", "B"], {"A": [1]})
        exits = _signals(close.index, ["A", "B"], {})

        c2, e2, x2 = _close_untradable(close, entries, exits)
        assert c2 is close and e2 is entries and x2 is exits

    def test_eine_einzelne_luecke_verkauft_nicht(self):
        """Eine fehlende Bar ist kein Delisting.

        Der Bulk-Lake liefert ständig Lücken: Handelsaussetzungen, fehlende
        Partitionen, ein Tag an dem dieses Papier nicht handelte während andere
        es taten. Die Qualitätsprüfung lässt sie durch — sie meldet `NaN` als
        Warnung, nicht als Fehler. Wer hier verkauft, verkauft wegen eines
        Datenlochs, und eine Crossover-Strategie steigt danach nie wieder ein.
        """
        from quantrace.backtest_runner import _close_untradable

        idx = pd.bdate_range("2007-01-01", periods=6)
        close = pd.DataFrame({"A": [10.0, 11.0, np.nan, 12.0, 13.0, 14.0]}, index=idx)
        entries = _signals(idx, ["A"], {"A": [0]})
        exits = _signals(idx, ["A"], {})

        c2, e2, x2 = _close_untradable(close, entries, exits)

        assert not x2["A"].any(), "durchhalten, nicht verkaufen"
        assert c2["A"].iloc[2] == pytest.approx(11.0), "letzter Kurs wird fortgeschrieben"
        assert not e2["A"].iloc[2], "einsteigen geht trotzdem nicht"

    def test_das_ende_der_reihe_verkauft_sehr_wohl(self):
        """Kein Kurs mehr, nie wieder — das ist der Unterschied."""
        from quantrace.backtest_runner import _close_untradable

        idx = pd.bdate_range("2007-01-01", periods=6)
        close = pd.DataFrame({"A": [10.0, 11.0, 12.0, np.nan, np.nan, np.nan]}, index=idx)
        entries = _signals(idx, ["A"], {"A": [0]})
        exits = _signals(idx, ["A"], {})

        _, _, x2 = _close_untradable(close, entries, exits)

        assert list(x2["A"]) == [False, False, False, True, False, False]

    def test_ausscheiden_aus_dem_universum_verkauft_auch_mit_kursen_danach(self):
        """Die dritte Ursache — und sie steht nicht in den Kursen.

        Ein Papier, das die Rekonstitution herausnimmt und später wieder
        aufnimmt, handelt durchgehend. Aus dem Rahmen allein wäre das nicht von
        einer Lücke zu unterscheiden; deshalb reist die Mitgliedschaft als
        eigene Maske mit.
        """
        from quantrace.backtest_runner import _close_untradable

        idx = pd.bdate_range("2007-01-01", periods=6)
        close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}, index=idx)
        tradable = pd.DataFrame({"A": [True, True, False, False, True, True]}, index=idx)
        entries = _signals(idx, ["A"], {"A": [0]})
        exits = _signals(idx, ["A"], {})

        _, e2, x2 = _close_untradable(close, entries, exits, tradable)

        assert list(x2["A"]) == [False, False, True, False, False, False]
        assert not e2["A"].iloc[2], "während der Abwesenheit kein Einstieg"
