"""Was ein Delisting einbringt — und was es nicht einbringt (#318).

Bis zum 2026-09-04 wurde die Zwangsschliessung zum **letzten beobachteten
Schluss** gebucht. Das ist der Kurs, zu dem zuletzt *jemand* gehandelt hat, und
nicht der Erlös: CRSP-Delisting-Renditen liegen bei performance-bedingten
Streichungen im Mittel um −30 % darunter (Shumway 1997; Shumway & Warther 1999).

**Warum das ausgerechnet in diesem Repo zählt.** Der Katalog führt über 8.045
delistete Common Stocks; der ganze Lake existiert, um Survivorship im
*Universum* loszuwerden. Der Ausstieg trug ihn weiter — einseitig nach oben, und
am stärksten bei Strategien, die Verlierer halten. Also bei genau denen, für die
man einen survivorship-freien Datensatz überhaupt lädt.

Die Tests hier prüfen drei Dinge getrennt: dass der Abschlag **greift**, dass er
**nur** das Delisting trifft (nicht jede Datenlücke, nicht das Ausscheiden aus
einem Universum), und dass er **im Ergebnis steht** — eine Annahme, die man
später nicht mehr ablesen kann, ist keine.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantrace.backtest_runner import _close_untradable, run_inline
from quantrace.models import BacktestConfig, MarketData

from .test_capital_model import ScriptedStrategy, _md, _signals

CASH = 100_000.0


def _szenario() -> tuple[MarketData, pd.DataFrame, pd.DataFrame]:
    """A hält 100, B verschwindet nach Bar 20 bei 50. Beide ab Bar 2 im Buch."""
    n = 40
    a = np.full(n, 100.0)
    b = np.full(n, 50.0)
    b[20:] = np.nan

    md = _md({"A": a, "B": b})
    idx = md.frame.index
    return md, _signals(idx, ["A", "B"], {"A": [2], "B": [2]}), _signals(idx, ["A", "B"], {})


def _close(md: MarketData) -> pd.DataFrame:
    return md.frame.xs("close", level="field", axis=1)


# ---------------------------------------------------------------------------
# Der Mechanismus


def test_der_zwangsverkauf_fillt_nicht_zum_letzten_print():
    md, entries, exits = _szenario()
    c, _, x = _close_untradable(_close(md), entries, exits, delisting_return=-0.30)

    assert bool(x["B"].iloc[20]), "der Exit sitzt weiter auf dem ersten Bar ohne Kurs"
    assert c["B"].iloc[20] == pytest.approx(35.0), "50 minus 30 % — nicht 50"
    assert c["A"].iloc[20] == pytest.approx(100.0), "A ist unberührt"


def test_ohne_abschlag_bleibt_alles_wie_vorher():
    """Der Rückwärtsgang. Alt-Ergebnisse müssen vergleichbar bleiben."""
    md, entries, exits = _szenario()
    c, _, _ = _close_untradable(_close(md), entries, exits, delisting_return=0.0)

    assert c["B"].iloc[20] == pytest.approx(50.0)
    assert c["B"].notna().all(), "ffill, damit der Verkauf einen Preis hat"


def test_eine_luecke_ist_kein_delisting():
    """Der Unterschied, an dem die ganze Regel hängt.

    Handelsaussetzung, fehlende Partition, ein Tag ohne Handel im Bulk-Lake:
    dort liegt später wieder ein Kurs. Wer da einen Abschlag bucht, bestraft
    eine Datenlücke — und die Strategie steigt oft nie wieder ein.
    """
    n = 30
    b = np.full(n, 50.0)
    b[10:13] = np.nan  # drei Tage Lücke, danach handelt es weiter
    md = _md({"A": np.full(n, 100.0), "B": b})
    idx = md.frame.index
    entries = _signals(idx, ["A", "B"], {"A": [2], "B": [2]})
    exits = _signals(idx, ["A", "B"], {})

    c, _, x = _close_untradable(_close(md), entries, exits, delisting_return=-0.30)

    assert not x["B"].any(), "kein Zwangs-Exit auf einer Lücke"
    assert c["B"].iloc[10:13].eq(50.0).all(), "und kein Abschlag"


def test_der_grund_schlaegt_die_vorgabe():
    """Eine Übernahme zahlt aus, eine Insolvenz nicht.

    Die Unterscheidung ist keine Meinung des Runners — sie kommt aus dem
    Katalog und reist als ``delisting_returns`` mit den Marktdaten.
    """
    md, entries, exits = _szenario()
    c, _, _ = _close_untradable(
        _close(md), entries, exits, delisting_return=-0.30, je_symbol={"B": 0.0}
    )

    assert c["B"].iloc[20] == pytest.approx(50.0), "Fusion: der letzte Kurs ist der Erlös"


# ---------------------------------------------------------------------------
# Was es kostet — von Hand nachgerechnet


def test_der_abschlag_bewegt_das_ergebnis_und_zwar_um_die_richtige_zahl():
    """95.000 aufgeteilt, B verliert 30 % seiner 47.500 beim Ausstieg.

    Ohne Abschlag endet das Buch bei 95.000 (nichts wächst hier). Mit −30 % auf
    B fehlen 14.250, also 0,1425 Gesamtrendite gegenüber dem Lauf ohne Abschlag.
    Kein Prozentzeichen-Gefühl, sondern eine Subtraktion.
    """
    md, entries, exits = _szenario()
    cfg = dict(fees_bps=0.0, slippage_bps=0.0, execution_lag=0, capital_model="shared")

    ohne = run_inline(
        "scripted",
        ScriptedStrategy(entries, exits),
        md,
        BacktestConfig(**cfg, delisting_return=0.0),
    )
    mit = run_inline(
        "scripted",
        ScriptedStrategy(entries, exits),
        md,
        BacktestConfig(**cfg, delisting_return=-0.30),
    )

    assert ohne.total_return - mit.total_return == pytest.approx(0.1425, abs=1e-3)
    assert mit.total_return < ohne.total_return, "der Fehler wirkte nach oben"


def test_ein_vollstaendiger_rahmen_kostet_nichts():
    """Wo nichts verschwindet, darf die Regel nichts ändern."""
    md = _md({"A": np.full(20, 100.0), "B": np.full(20, 50.0)})
    idx = md.frame.index
    entries = _signals(idx, ["A", "B"], {"A": [1]})
    exits = _signals(idx, ["A", "B"], {})

    c, e, x = _close_untradable(_close(md), entries, exits, delisting_return=-0.30)

    pd.testing.assert_frame_equal(c, _close(md))
    assert not x.any().any()


# ---------------------------------------------------------------------------
# Die Annahme muss ablesbar sein


def test_die_annahme_steht_im_ergebnis():
    """Eine Kostenannahme, die man später nicht mehr ablesen kann, ist keine.

    Genau daran ist #317 gescheitert: das JSON trug `per_asset_class`, gerechnet
    wurde `flat`. Für den Delisting-Abschlag gilt dasselbe — er verschiebt jede
    Kennzahl und muss deshalb neben ihr stehen.
    """
    md, entries, exits = _szenario()
    res = run_inline(
        "scripted",
        ScriptedStrategy(entries, exits),
        md,
        BacktestConfig(fees_bps=0.0, slippage_bps=0.0, execution_lag=0, delisting_return=-0.30),
    )

    assert res.config.delisting_return == pytest.approx(-0.30)
    assert "delisting_return" in res.model_dump_json()


def test_die_vorgabe_ist_vorsichtig():
    """Ohne bekannten Grund gilt die performance-bedingte Annahme.

    Dieselbe Abwägung wie bei der Kostenklasse: zu teuer gerechnet verwirft eine
    gute Strategie, zu billig gerechnet gibt eine schlechte frei.
    """
    assert BacktestConfig().delisting_return == pytest.approx(-0.30)


@pytest.mark.parametrize("wert", [0.5, -1.5, -1.0])
def test_unmoegliche_abschlaege_werden_abgelehnt(wert):
    """Ein Delisting bringt weder mehr als den letzten Kurs noch weniger als nichts.

    Die ``-1.0`` steht dabei nicht wegen der Finanzlogik in der Liste, sondern
    wegen der Ausführung: siehe der Test darunter.
    """
    with pytest.raises(ValueError):
        BacktestConfig(delisting_return=wert)


def test_ein_totalverlust_bleibt_eine_order():
    """Bei ``-1`` ist der Fill-Kurs null — und eine Order zum Preis null ist keine.

    vectorbt weist sie ab (``order.price must be finite and greater than 0``),
    und zwar mitten im Lauf, nicht beim Bauen der Config. Ein Wert, den das
    Modell annimmt und der Runner nicht ausführen kann, ist keine Annahme,
    sondern eine Falle: der Abbruch käme Minuten nach dem Load, auf einem
    Universum, das eine halbe Stunde zum Laden gebraucht hat.

    Nachgemessen am 2026-09-05: derselbe Fehler, den ein nicht-positiver Kurs
    im Universum auslöst — nur diesmal von uns selbst erzeugt.
    """
    md, entries, exits = _szenario()
    knapp = run_inline(
        "scripted",
        ScriptedStrategy(entries, exits),
        md,
        BacktestConfig(
            fees_bps=0.0, slippage_bps=0.0, execution_lag=0, delisting_return=-0.999
        ),
    )
    assert knapp.total_return < 0, "fast alles weg, aber gerechnet"
    c, _, _ = _close_untradable(
        _close(md), entries, exits, delisting_return=-0.999
    )
    assert c["B"].iloc[20] > 0.0, "der Fill-Kurs bleibt handelbar"


# ---------------------------------------------------------------------------
# Ein Abgang aus dem Index ist kein Delisting


def _ausgeschieden(ab: int, n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Der Rahmen, wie ``membership.apply`` ihn hinterlässt.

    B ist ab Bar ``ab`` nicht mehr Mitglied. ``apply`` setzt seine Kurse dort
    auf ``NaN`` — im Rahmen sieht das Papier damit aus, als hätte es aufgehört
    zu handeln, obwohl es weiterhandelt.
    """
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame({"A": np.full(n, 100.0), "B": np.full(n, 50.0)}, index=idx)
    tradable = pd.DataFrame(True, index=idx, columns=["A", "B"])
    close.loc[idx[ab:], "B"] = np.nan
    tradable.loc[idx[ab:], "B"] = False
    return close, tradable


def test_wer_aus_dem_universum_faellt_zahlt_keinen_abschlag():
    """Der teuerste Fall, den die Vorgabe allein falsch bucht.

    `us_top500_liquid` rekonstituiert quartalsweise und führt über zwanzig
    Jahre 1.934 Kürzel. Jeder Abgang hinterlässt denselben NaN-Schwanz wie ein
    Delisting — ohne die Mitgliedschaftsmaske bekämen sie alle den
    Insolvenz-Abschlag, und der Fehler zeigte diesmal nach unten.
    """
    close, tradable = _ausgeschieden(ab=20)
    idx = close.index
    entries = pd.DataFrame(False, index=idx, columns=["A", "B"])
    entries.iloc[2] = True
    exits = pd.DataFrame(False, index=idx, columns=["A", "B"])

    c, _, x = _close_untradable(close, entries, exits, tradable, delisting_return=-0.30)

    assert c["B"].iloc[20] == pytest.approx(50.0), "ausgeschieden, nicht pleite"
    assert bool(x["B"].iloc[20]), "verkauft wird trotzdem — nur zum letzten Kurs"


def test_ein_delisting_im_universum_zahlt_ihn_weiter():
    """Die Gegenprobe: der Wächter darf nicht jeden Abschlag abschalten.

    B bleibt Mitglied bis zum Schluss und hört trotzdem auf zu handeln. Genau
    das ist der Fall, für den ``delisting_return`` da ist.
    """
    n = 30
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame({"A": np.full(n, 100.0), "B": np.full(n, 50.0)}, index=idx)
    close.loc[idx[20:], "B"] = np.nan
    tradable = pd.DataFrame(True, index=idx, columns=["A", "B"])
    entries = pd.DataFrame(False, index=idx, columns=["A", "B"])
    entries.iloc[2] = True
    exits = pd.DataFrame(False, index=idx, columns=["A", "B"])

    c, _, x = _close_untradable(close, entries, exits, tradable, delisting_return=-0.30)

    assert c["B"].iloc[20] == pytest.approx(35.0)
    assert bool(x["B"].iloc[20])


# ---------------------------------------------------------------------------
# Woher der Grund kommt: aus der Karte, nicht aus einer Liste im Code


def _karte(zeilen: list[tuple[str, str, str, bool]]) -> pd.DataFrame:
    """Ein Manifest-Ausschnitt: (instrument, code, last, active)."""
    return pd.DataFrame(
        [
            {
                "instrument": i,
                "code": c,
                "last": pd.Timestamp(last).date(),
                "active": aktiv,
            }
            for i, c, last, aktiv in zeilen
        ]
    )


class TestMitKursenNach:
    """Die eine Frage, die der Katalog über ein verschwundenes Papier beantwortet."""

    KARTE = _karte(
        [
            ("isin.US1", "LEB", "2008-09-15", False),  # endete, und zwar endgültig
            ("isin.US2", "SPY", "2026-09-01", True),  # handelt weiter
            ("isin.US3", "XYZ", "2021-06-30", False),  # endete nach dem Fenster
        ]
    )

    def test_wer_danach_noch_kurse_hat_wird_genannt(self):
        from quantrace import resolve

        gefunden = resolve.mit_kursen_nach(
            ["isin.US1", "isin.US2", "isin.US3"], date(2020, 12, 31), manifest=self.KARTE
        )
        assert gefunden == {"isin.US2", "isin.US3"}

    def test_wer_vorher_endete_wird_nicht_genannt(self):
        """Der Normalfall — und der, bei dem der Abschlag stehen bleiben muss."""
        from quantrace import resolve

        assert resolve.mit_kursen_nach(["isin.US1"], date(2020, 12, 31), manifest=self.KARTE) == set()

    def test_am_lake_rand_traegt_active_die_aussage(self):
        """`last` allein reicht nicht: liegt das Fensterende auf der Front, hat
        auch ein lebendes Papier keine Kurse mehr *danach*."""
        from quantrace import resolve

        gefunden = resolve.mit_kursen_nach(
            ["isin.US2", "isin.US3"], date(2026, 9, 1), manifest=self.KARTE
        )
        assert gefunden == {"isin.US2"}

    def test_eine_leere_karte_behauptet_nichts(self):
        """Ohne Schicht 2 gibt es keine Widerlegung — also bleibt die Vorgabe."""
        from quantrace import resolve

        leer = pd.DataFrame(columns=["instrument", "code", "last", "active"])
        assert resolve.mit_kursen_nach(["isin.US1"], date(2020, 1, 1), manifest=leer) == set()
        assert resolve.mit_kursen_nach([], date(2020, 1, 1), manifest=self.KARTE) == set()

    def test_ein_unbekanntes_instrument_ist_keine_aussage(self):
        from quantrace import resolve

        assert resolve.mit_kursen_nach(["isin.NIX"], date(2000, 1, 1), manifest=self.KARTE) == set()


class TestErloesAnnahmen:
    """Was der Loader daraus macht — die Verdrahtung, nicht die Regel."""

    def _frame(self):
        md, _, _ = _szenario()  # B endet nach Bar 20, A läuft durch
        return md.frame

    def test_nur_das_widerlegte_symbol_bekommt_einen_eintrag(self, monkeypatch):
        from quantrace import data_agent, resolve

        monkeypatch.setattr(resolve, "mit_kursen_nach", lambda instr, ende: {"isin.B"})
        annahmen = data_agent._erloes_annahmen(
            self._frame(), {"A": "isin.A", "B": "isin.B"}, date(2021, 12, 31)
        )
        assert annahmen == {"B": 0.0}, "A bricht nicht ab und darf nicht auftauchen"

    def test_ohne_widerspruch_bleibt_die_karte_still(self, monkeypatch):
        """Kein Eintrag heißt „nichts Besseres bekannt", nicht „kein Abschlag"."""
        from quantrace import data_agent, resolve

        monkeypatch.setattr(resolve, "mit_kursen_nach", lambda instr, ende: set())
        assert (
            data_agent._erloes_annahmen(
                self._frame(), {"A": "isin.A", "B": "isin.B"}, date(2021, 12, 31)
            )
            == {}
        )

    def test_ein_vollstaendiger_rahmen_fragt_die_karte_gar_nicht(self, monkeypatch):
        """Ein LIST über die Karte für 1.900 Symbole, von denen keines abbricht,
        wäre Arbeit ohne Frage."""
        from quantrace import data_agent, resolve

        def _nie(*_a, **_k):
            raise AssertionError("die Karte wurde ohne Anlass gelesen")

        monkeypatch.setattr(resolve, "mit_kursen_nach", _nie)
        md = _md({"A": np.full(20, 100.0), "B": np.full(20, 50.0)})
        assert data_agent._erloes_annahmen(md.frame, {"A": "x", "B": "y"}, date(2021, 12, 31)) == {}

    def test_der_eintrag_wirkt_bis_in_den_kurs(self, monkeypatch):
        """Vom Loader bis zur Buchung, ohne Handgriff dazwischen."""
        from quantrace import data_agent, resolve

        monkeypatch.setattr(resolve, "mit_kursen_nach", lambda instr, ende: {"isin.B"})
        md, entries, exits = _szenario()
        je_symbol = data_agent._erloes_annahmen(
            md.frame, {"A": "isin.A", "B": "isin.B"}, date(2021, 12, 31)
        )
        c, _, _ = _close_untradable(
            _close(md), entries, exits, delisting_return=-0.30, je_symbol=je_symbol
        )
        assert c["B"].iloc[20] == pytest.approx(50.0)


def test_die_annahme_ueberlebt_das_beschneiden():
    """`membership.apply` baut ein neues MarketData — und liess dabei zwei
    Aussagen über die Daten liegen, die das Beschneiden nicht berührt."""
    from quantrace.membership import Membership, MembershipPeriod

    md, _, _ = _szenario()
    md = md.model_copy(update={"delisting_returns": {"B": 0.0}, "missing_symbols": ["C"]})
    m = Membership(
        periods=(MembershipPeriod(start=md.start, end=None, symbols=frozenset({"A", "B"})),)
    )

    beschnitten = m.apply(md)

    assert beschnitten.delisting_returns == {"B": 0.0}
    assert beschnitten.missing_symbols == ["C"]


def test_der_loader_haengt_die_annahme_auch_an():
    """Die Verdrahtung selbst, weil kein Test sie sonst berührt.

    `_load_via_bulk` braucht Schicht 2 und läuft deshalb in keinem Test. Fiele
    das Schlüsselwort weg, bliebe `_erloes_annahmen` eine korrekte Funktion,
    die niemand ruft — und der Abschlag träfe wieder jedes Symbol, dessen Reihe
    im Rahmen endet. Dieselbe Bauform-Prüfung wie bei #317.
    """
    import ast
    from pathlib import Path

    quelle = Path(__file__).resolve().parents[2] / "quantrace" / "data_agent.py"
    baum = ast.parse(quelle.read_text())
    for knoten in ast.walk(baum):
        if isinstance(knoten, ast.FunctionDef) and knoten.name == "_load_via_bulk":
            aufrufe = [
                k
                for k in ast.walk(knoten)
                if isinstance(k, ast.Call)
                and isinstance(k.func, ast.Name)
                and k.func.id == "MarketData"
            ]
            assert aufrufe, "_load_via_bulk baut kein MarketData mehr"
            assert any(
                kw.arg == "delisting_returns" for a in aufrufe for kw in a.keywords
            ), "MarketData ohne delisting_returns — _erloes_annahmen läuft ins Leere"
            return
    raise AssertionError("_load_via_bulk nicht gefunden")


# ---------------------------------------------------------------------------
# Über die CLI erreichbar — sonst ist der Vergleich mit Altergebnissen keiner


class TestCliOption:
    """`--delisting-return` ist die einzige Art, einen Lauf ohne Abschlag zu
    fahren. Ohne sie stünde in der Doku „wer vergleichen muss, setzt ihn auf 0"
    und es gäbe keinen Weg dorthin — und die Messung aus #318 (vorher/nachher
    auf `us_top500_liquid`) liesse sich gar nicht durchführen.
    """

    def test_ohne_angabe_gilt_die_vorgabe_aus_dem_modell(self):
        from quantrace.cli import _backtest_config

        assert _backtest_config("flat", "shared").delisting_return == pytest.approx(-0.30)

    def test_null_rechnet_wie_vor_318(self):
        from quantrace.cli import _backtest_config

        assert _backtest_config("flat", "shared", 0.0).delisting_return == pytest.approx(0.0)

    @pytest.mark.parametrize("wert", [0.5, -1.5])
    def test_ein_unmoeglicher_wert_nennt_die_option_nicht_das_feld(self, wert):
        """Wer über die CLI kommt, sucht nach `--delisting-return`, nicht nach
        `BacktestConfig.delisting_return`."""
        import typer

        from quantrace.cli import _backtest_config

        with pytest.raises(typer.BadParameter, match="--delisting-return"):
            _backtest_config("flat", "shared", wert)

    def test_alle_drei_kommandos_kennen_die_option(self):
        """`backtest`, `sweep` und `walkforward` — ein Vergleich, der nur für
        eines der drei geht, ist keiner."""
        import inspect

        from quantrace import cli

        for name in ("backtest", "sweep", "walkforward"):
            sig = inspect.signature(getattr(cli, name))
            assert "delisting_return" in sig.parameters, name
