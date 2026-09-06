"""Ein kaputtes Symbol wirft ein Symbol raus, nicht das Universum (#322).

**Der Fall.** `us_top500_liquid` führt 1.934 Kürzel. Eines davon, `PNLYY`,
trägt am 2011-06-08 `open=10,00 · high=9,74 · low=10,00 · close=9,74` — high
und low vertauscht, ein schlechter Tick von EODHD in Schicht 1. Das Gate stuft
`high < low` als Fehler ein, und der Loader brach bei jedem Fehler den ganzen
Load ab. Ergebnis: **kein** Backtest über dieses Universum konnte ein Fenster
umfassen, das den 2011-06-08 enthält — für alle Strategien, dauerhaft.

Das skaliert falsch herum. Ein survivorship-freies Universum über zwanzig Jahre
enthält mit Sicherheit weitere solche Ticks; je länger das Fenster, desto
sicherer der Abbruch. Der Querschnitts-Backtest aus #300 wäre daran nie
angekommen.

Was hier geprüft wird, ist deshalb **nicht** „die Prüfung ist milder geworden" —
sie ist es nicht. Geprüft wird die Reichweite: der Ausschluss trifft das Symbol,
er bleibt sichtbar, und wenn nichts übrig bleibt, wird weiterhin abgebrochen.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantrace.data_agent import _unbrauchbare_symbole
from quantrace.quality import QualityReport

from .test_capital_model import _md


def _bericht(*eintraege: tuple[str, str, str, str]) -> QualityReport:
    r = QualityReport()
    for symbol, kind, detail, severity in eintraege:
        r.add(symbol, kind, detail, severity)
    return r


class TestAuswahl:
    def test_nur_fehler_schliessen_aus(self):
        """Eine Warnung ist eine Auskunft, kein Urteil.

        Fehlende Tage, gekappte Abdeckung, ein Bar ohne Volumen — das sind die
        Normalfälle eines Bulk-Lakes. Wer daran Symbole verwirft, verwirft das
        halbe Universum.
        """
        r = _bericht(
            ("LEH", "high_lt_low", "1 Bars mit high < low", "error"),
            ("SPY", "missing_days", "2 von 251 fehlen", "warning"),
            ("QQQ", "coverage_truncated", "beginnt erst 2007-02-01", "warning"),
        )
        assert set(_unbrauchbare_symbole(r)) == {"LEH"}

    def test_der_grund_steht_dabei(self):
        """„Ausgeschlossen" ohne Grund ist keine Aussage, sondern ein Loch."""
        r = _bericht(("PNLYY", "high_lt_low", "1 Bars mit high < low", "error"))
        grund = _unbrauchbare_symbole(r)["PNLYY"]
        assert "high_lt_low" in grund
        assert "1 Bars" in grund

    def test_mehrere_gruende_gehen_nicht_verloren(self):
        """Ein Symbol kann auf mehr als eine Art kaputt sein, und das zweite
        Merkmal ist oft das, an dem man die Ursache erkennt."""
        r = _bericht(
            ("X", "high_lt_low", "3 Bars", "error"),
            ("X", "nonpositive_price", "close: 2 Werte <= 0", "error"),
        )
        grund = _unbrauchbare_symbole(r)["X"]
        assert "high_lt_low" in grund and "nonpositive_price" in grund

    def test_ein_sauberer_bericht_schliesst_nichts_aus(self):
        assert _unbrauchbare_symbole(QualityReport()) == {}
        assert _unbrauchbare_symbole(_bericht(("SPY", "zero_volume", "1 Bar", "warning"))) == {}


def test_der_ausschluss_ueberlebt_das_beschneiden():
    """`membership.apply` baut ein neues MarketData — und ein Ausschluss, den
    das Beschneiden verschluckt, ist beim Lesen der Kennzahlen nicht mehr da."""
    from quantrace.membership import Membership, MembershipPeriod

    md = _md({"A": np.full(20, 100.0), "B": np.full(20, 50.0)})
    md = md.model_copy(update={"unusable_symbols": {"C": "high_lt_low: 1 Bars"}})
    m = Membership(
        periods=(MembershipPeriod(start=md.start, end=None, symbols=frozenset({"A", "B"})),)
    )

    assert m.apply(md).unusable_symbols == {"C": "high_lt_low: 1 Bars"}


def test_die_vorgabe_ist_leer_und_nicht_none():
    """Leer heisst „nichts verworfen". `None` hiesse „unbekannt", und das wäre
    für Altergebnisse richtig — aber dieses Feld gibt es erst seit #322, also
    ist jeder neue Lauf eine Aussage."""
    md = _md({"A": np.full(5, 1.0)})
    assert md.unusable_symbols == {}


def test_ein_ausschluss_steht_im_ergebnis_json():
    """Dieselbe Regel wie bei `delisting_return` und `cost_model`: eine
    Annahme, die man später nicht mehr ablesen kann, ist keine."""
    md = _md({"A": np.full(5, 1.0)})
    md = md.model_copy(update={"unusable_symbols": {"PNLYY": "high_lt_low: 1 Bars"}})
    assert "PNLYY" in md.model_dump_json()


@pytest.mark.parametrize("feld", ["missing_symbols", "unusable_symbols"])
def test_die_beiden_listen_bleiben_getrennt(feld):
    """„Nie geladen" und „geladen und verworfen" sehen in einer Symbolliste
    gleich aus und bedeuten Verschiedenes: das eine ist eine Lücke im Lake, das
    andere ein Datenfehler. In einen Topf geworfen wäre keine der beiden
    Auskünfte mehr etwas wert."""
    from quantrace.models import MarketData

    assert feld in MarketData.model_fields
