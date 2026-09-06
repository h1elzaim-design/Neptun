"""Ein Bar aus lauter Nullen ist keine Beobachtung (#322, zweite Wand).

**Der Fall.** EODHD schreibt für manche Tage eine Zeile mit
``open=high=low=close=volume=0``. Am 2026-09-05 nachgezählt: 136 von 2.245
Zeilen bei `ANG`, `DIC` und `IVL` — roh, vor jeder Adjustierung. Eine Aktie hat
an keinem Tag zu 0,00 $ gehandelt; die Null ist ein Platzhalter für „nichts
geliefert".

**Was sie angerichtet hat.** Das Qualitätstor stufte nicht-positive Kurse als
*Warnung* ein, also kamen sie bis in den Backtest, und dort brach vectorbt ab:
``order.price must be finite and greater than 0`` — Minuten nach einem Load,
der eine halbe Stunde gedauert hatte. Genau daran ist die Messung aus #318 im
zweiten Anlauf gescheitert.

Zwei Verteidigungslinien, und beide werden hier geprüft: die Null wird in der
richtigen Schicht zur Lücke, und was trotzdem als nicht-positiver *Schlusskurs*
durchkommt, ist ein Fehler und keine Warnung.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantrace.bulk_read import _nullbars_als_luecke
from quantrace.quality import check_symbol


def _lang(zeilen: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Long-Format wie aus Schicht 2: (code, date, kurs) — OHLC alle gleich."""
    return pd.DataFrame(
        [
            {
                "code": c,
                "date": pd.Timestamp(d).date(),
                "open": k,
                "high": k,
                "low": k,
                "close": k,
                "volume": 0.0 if k == 0 else 1_000.0,
            }
            for c, d, k in zeilen
        ]
    )


class TestNullbarWirdLuecke:
    def test_die_nullzeile_verschwindet(self):
        roh = _lang(
            [
                ("ANG", "2007-03-20", 12.5),
                ("ANG", "2007-03-21", 0.0),
                ("ANG", "2007-03-22", 12.7),
            ]
        )
        sauber = _nullbars_als_luecke(roh)
        assert len(sauber) == 2
        assert list(sauber["close"]) == [12.5, 12.7]

    def test_gesunde_zeilen_bleiben_unangetastet(self):
        roh = _lang([("SPY", "2007-01-03", 140.0), ("SPY", "2007-01-04", 141.0)])
        pd.testing.assert_frame_equal(_nullbars_als_luecke(roh), roh)

    def test_ein_ruhiger_tag_ist_kein_nullbar(self):
        """Volumen null bei positivem Kurs ist der Normalfall eines dünnen
        Papiers — 709 von 2.245 Zeilen im gemessenen Ausschnitt. Wer die
        wegwirft, wirft die halbe Reihe weg."""
        roh = _lang([("DIC", "2010-09-02", 8.0)])
        roh.loc[0, "volume"] = 0.0
        assert len(_nullbars_als_luecke(roh)) == 1

    def test_ein_einzelner_nullwert_bleibt_stehen(self):
        """Ein kaputter Eröffnungskurs neben drei gesunden ist ein anderer
        Defekt — und die Warnung dafür soll ihn weiter treffen, statt dass ihn
        diese Regel wegräumt."""
        roh = _lang([("X", "2010-01-04", 5.0)])
        roh.loc[0, "open"] = 0.0
        assert len(_nullbars_als_luecke(roh)) == 1

    def test_ohne_treffer_wird_der_index_nicht_angefasst(self):
        roh = _lang([("SPY", "2007-01-03", 140.0)])
        assert _nullbars_als_luecke(roh) is roh, "kein unnötiger Kopiervorgang"

    def test_der_index_wird_nach_dem_loeschen_neu_gezaehlt(self):
        roh = _lang(
            [("A", "2010-01-04", 1.0), ("A", "2010-01-05", 0.0), ("A", "2010-01-06", 2.0)]
        )
        assert list(_nullbars_als_luecke(roh).index) == [0, 1]


class TestNichtPositiverSchlusskurs:
    """Die zweite Linie — was trotzdem durchkommt, hält den Lauf nicht mehr auf."""

    def _frame(self, closes: list[float]) -> pd.DataFrame:
        idx = pd.bdate_range("2010-01-04", periods=len(closes))
        return pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes,
             "volume": [1000.0] * len(closes)},
            index=idx,
        )

    def test_ein_nicht_positiver_close_ist_ein_fehler(self):
        """An `close` wird ausgeführt. Ein Preis <= 0 ist dort keine Ungenauigkeit."""
        r = check_symbol("X", self._frame([10.0, 0.0, 12.0]), calendar="us_equity")
        treffer = [i for i in r.issues if i.kind == "nonpositive_price"]
        assert any(i.severity == "error" for i in treffer)
        assert not r.ok

    def test_open_high_low_bleiben_warnung(self):
        """Ein Symbol wegen eines kaputten Eröffnungskurses zu verwerfen, wäre
        teurer als die Auskunft wert ist — gehandelt wird an `close`."""
        f = self._frame([10.0, 11.0, 12.0])
        f.loc[f.index[1], "open"] = 0.0
        r = check_symbol("X", f, calendar="us_equity")
        treffer = [i for i in r.issues if i.kind == "nonpositive_price"]
        assert treffer and all(i.severity == "warning" for i in treffer)
        assert r.ok, "das darf keinen Lauf aufhalten"

    @pytest.mark.parametrize("wert", [0.0, -1.0])
    def test_null_und_negativ_gelten_gleich(self, wert):
        r = check_symbol("X", self._frame([10.0, wert, 12.0]), calendar="us_equity")
        assert not r.ok
