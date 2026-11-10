"""Die Box (#356, ADR-018) — die vier Abnahmekriterien, soweit sie ohne Broker
prüfbar sind:

1. keine Folge von Agent-Aktionen überschreitet die Limits (Zufallsfolgen)
2. der Risk-Agent kann strukturell nicht erhöhen
3. der Kill-Switch lässt nichts Neues zu und stellt glatt
4. die Verlustleiter feuert an allen drei Stufen

Offen bleibt aus dem Ticket nur „< 60 s im Paper-Konto" — das braucht ein
Konto.
"""

from __future__ import annotations

import ast
import random
from datetime import date, timedelta
from pathlib import Path

import pytest

from quantrace.brokers.base import Broker, OrderSide
from quantrace.hub import executor, limits, risk
from quantrace.hub.box import AgentZustand, Allow, Auftrag, Instrument, Reduce, Reject, check
from quantrace.hub.ladder import Folge, Stand, Stufe, schritt
from quantrace.hub.limits import HubLimits, Mandat, MandatFehlerError, Verlustleiter
from quantrace.paper.executor import LiveExecutionBlocked

REPO = Path(__file__).resolve().parents[2]
BUY, SELL = OrderSide.BUY, OrderSide.SELL


def _mandat(**kw) -> Mandat:
    basis = dict(
        agent_id="bot",
        capital_usd=10_000.0,
        instruments=frozenset({"AAA", "BBB", "CCC"}),
        min_adv_usd=1_000_000.0,
        min_price=5.0,
        min_market_cap_usd=100_000_000.0,
    )
    return Mandat(**{**basis, **kw})


def _papier(price=100.0, adv=5_000_000.0, cap=1_000_000_000.0) -> Instrument:
    return Instrument(price=price, adv_usd=adv, market_cap_usd=cap)


MARKT = {"AAA": _papier(), "BBB": _papier(50.0), "CCC": _papier(20.0)}


def _pruefe(side, symbol="AAA", menge=1.0, *, zustand=None, mandat=..., markt=MARKT, kill=False):
    m = _mandat() if mandat is ... else mandat
    return check(
        Auftrag("bot", symbol, side, menge),
        zustand or AgentZustand(equity_usd=10_000.0),
        m,
        markt,
        kill_switch=kill,
    )


# ---------------------------------------------------------------------------
# check: was fehlt, ist nicht erlaubt


def test_kauf_innerhalb_der_limits_geht_durch():
    assert _pruefe(BUY, menge=5) == Allow(5)


def test_ohne_mandat_keine_order():
    assert isinstance(_pruefe(BUY, mandat=None), Reject)


def test_fremdes_mandat_gilt_nicht():
    assert isinstance(_pruefe(BUY, mandat=_mandat(agent_id="anderer")), Reject)


@pytest.mark.parametrize("menge", [0, -1, float("nan"), float("inf")])
def test_unsinnige_stueckzahl_wird_abgelehnt(menge):
    assert isinstance(_pruefe(BUY, menge=menge), Reject)


def test_nicht_auf_der_instrumentliste():
    r = _pruefe(BUY, symbol="ZZZ", markt={**MARKT, "ZZZ": _papier()})
    assert isinstance(r, Reject) and "Instrumentliste" in r.gruende[0]


def test_leeres_mandat_handelt_nicht():
    assert isinstance(_pruefe(BUY, mandat=_mandat(capital_usd=0)), Reject)
    assert isinstance(_pruefe(BUY, mandat=_mandat(instruments=frozenset())), Reject)


@pytest.mark.parametrize("preis", [None, 0, -5, float("nan")])
def test_ohne_gueltigen_kurs_kein_kauf(preis):
    assert isinstance(_pruefe(BUY, markt={"AAA": _papier(price=preis)}), Reject)


def test_unbekannte_liquiditaet_ist_nicht_genug_liquiditaet():
    """Der Pump-and-Dump-Schutz darf nicht an einer fehlenden Zahl scheitern."""
    r = _pruefe(BUY, markt={"AAA": Instrument(price=100.0)})
    assert isinstance(r, Reject)
    assert any("ADV" in g and "unbekannt" in g for g in r.gruende)
    assert any("Marktkapitalisierung" in g for g in r.gruende)


def test_zu_illiquide_zu_billig_zu_klein():
    assert isinstance(_pruefe(BUY, markt={"AAA": _papier(adv=10_000)}), Reject)
    assert isinstance(_pruefe(BUY, markt={"AAA": _papier(price=1.0)}), Reject)
    assert isinstance(_pruefe(BUY, markt={"AAA": _papier(cap=1_000_000)}), Reject)


def test_positionsgrenze_kuerzt_statt_abzulehnen():
    # 10 % von 10.000 $ = 1.000 $ = 10 Stück zu 100 $
    r = _pruefe(BUY, menge=50)
    assert isinstance(r, Reduce) and r.quantity == pytest.approx(10)


def test_volle_position_wird_abgelehnt():
    z = AgentZustand(equity_usd=10_000.0, positions={"AAA": 10})
    assert isinstance(_pruefe(BUY, zustand=z), Reject)


def test_brutto_grenze_zaehlt_alle_positionen():
    m = _mandat(max_position_pct=1.0, max_gross_exposure=0.5)
    z = AgentZustand(equity_usd=10_000.0, positions={"BBB": 90})  # 4.500 $ von 5.000 $
    r = _pruefe(BUY, menge=20, zustand=z, mandat=m)
    assert isinstance(r, Reduce) and r.quantity == pytest.approx(5)


def test_gehaltene_position_ohne_kurs_sperrt_jeden_kauf():
    """Ein Exposure, das sich nicht rechnen lässt, ist nicht null."""
    z = AgentZustand(equity_usd=10_000.0, positions={"XXX": 100})
    r = _pruefe(BUY, zustand=z)
    assert isinstance(r, Reject) and "XXX" in r.gruende[0]


def test_verluste_verkleinern_die_basis_gewinne_vergroessern_sie_nicht():
    arm = _pruefe(BUY, menge=50, zustand=AgentZustand(equity_usd=5_000.0))
    reich = _pruefe(BUY, menge=50, zustand=AgentZustand(equity_usd=50_000.0))
    assert arm.quantity == pytest.approx(5)
    assert reich.quantity == pytest.approx(10)  # Mandat bleibt 10.000 $


def test_negatives_eigenkapital_kauft_nicht():
    assert isinstance(_pruefe(BUY, zustand=AgentZustand(equity_usd=-1.0)), Reject)


# ---------------------------------------------------------------------------
# Verkäufe: schliessen geht immer, leerverkaufen nie


def test_verkauf_ohne_bestand_ist_kein_leerverkauf():
    assert isinstance(_pruefe(SELL), Reject)


def test_verkauf_wird_auf_den_bestand_gekuerzt():
    z = AgentZustand(equity_usd=10_000.0, positions={"AAA": 3})
    assert _pruefe(SELL, menge=10, zustand=z) == Reduce(3, ("Verkauf auf den Bestand von 3 gekürzt",))


@pytest.mark.parametrize("stufe", [Stufe.PAUSED_TODAY, Stufe.FROZEN, Stufe.RETIRED])
def test_schliessen_geht_in_jedem_zustand(stufe):
    """Eine Box, die das Schliessen verhindert, hält das Risiko fest."""
    z = AgentZustand(equity_usd=1.0, positions={"ZZZ": 3}, leiter=Stand(stufe))
    assert _pruefe(SELL, symbol="ZZZ", menge=3, zustand=z, markt={}, kill=True) == Allow(3)


@pytest.mark.parametrize("stufe", [Stufe.PAUSED_TODAY, Stufe.FROZEN, Stufe.RETIRED])
def test_kein_kauf_ausserhalb_von_active(stufe):
    z = AgentZustand(equity_usd=10_000.0, leiter=Stand(stufe))
    assert isinstance(_pruefe(BUY, zustand=z), Reject)


# ---------------------------------------------------------------------------
# Abnahme 3: Kill-Switch


def test_kill_switch_laesst_keinen_kauf_durch():
    r = _pruefe(BUY, kill=True)
    assert isinstance(r, Reject) and "Kill-Switch" in r.gruende[0]


def test_kill_switch_aus_der_umgebung_schaltet_nur_ein(monkeypatch):
    aus, an = HubLimits(False, {}), HubLimits(True, {})
    assert not limits.kill_switch_aktiv(aus)
    monkeypatch.setenv("QUANTRACE_HUB_KILL", "1")
    assert limits.kill_switch_aktiv(aus)
    monkeypatch.setenv("QUANTRACE_HUB_KILL", "0")
    assert limits.kill_switch_aktiv(an)  # die Umgebung hebt die Datei nicht auf


def test_glattstellen_verkauft_alles_und_nur_bestaende():
    pos = {"AAA": 5, "BBB": 0, "CCC": 2.5, "DDD": float("nan")}
    auftraege = risk.glattstellen("bot", pos)
    assert [(a.symbol, a.side, a.quantity) for a in auftraege] == [("AAA", SELL, 5), ("CCC", SELL, 2.5)]


# ---------------------------------------------------------------------------
# Abnahme 2: der Risk-Agent kann strukturell nicht erhöhen


@pytest.mark.parametrize("anteil", [0, -0.5, 1.01, 2, float("nan"), float("inf"), True])
def test_reduzieren_nimmt_nur_einen_anteil_zwischen_null_und_eins(anteil):
    with pytest.raises(ValueError):
        risk.Reduzieren("bot", "AAA", anteil)


def test_risk_aktionen_ergeben_nur_verkaeufe_bis_zum_bestand():
    rng = random.Random(7)
    for _ in range(2_000):
        bestand = rng.choice([0.0, rng.uniform(0.001, 500)])
        aktion = rng.choice(
            [
                risk.Reduzieren("bot", "AAA", rng.uniform(1e-9, 1.0)),
                risk.Schliessen("bot", "AAA"),
                risk.Einfrieren("bot"),
            ]
        )
        for a in risk.auftraege(aktion, {"AAA": bestand}):
            assert a.side is SELL
            assert 0 < a.quantity <= bestand


def test_risk_modul_kennt_keinen_kauf():
    """Auf dem AST: das Wort BUY kommt im Risk-Modul nicht vor, und die drei
    Aktionen tragen kein Feld, das eine Seite oder ein Ziel ausdrücken könnte."""
    tree = ast.parse((REPO / "quantrace/hub/risk.py").read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "BUY" not in attrs
    felder = {
        cls.__name__: set(cls.__dataclass_fields__)
        for cls in (risk.Reduzieren, risk.Schliessen, risk.Einfrieren)
    }
    assert felder == {
        "Reduzieren": {"agent_id", "symbol", "anteil"},
        "Schliessen": {"agent_id", "symbol"},
        "Einfrieren": {"agent_id"},
    }


# ---------------------------------------------------------------------------
# Abnahme 4: die Verlustleiter feuert an allen drei Stufen

L = Verlustleiter()
T0 = date(2026, 11, 10)


def _s(stand, *, tag=T0, verlust=0.0, dd=0.0, bruch=False):
    return schritt(stand, L, heute=tag, tagesverlust=verlust, drawdown=dd, regelbruch=bruch)


def test_stufe_1_tagesverlust_pausiert_bis_zum_naechsten_tag():
    stand, folgen = _s(Stand(), verlust=0.02)
    assert stand.stufe is Stufe.PAUSED_TODAY and folgen == (Folge.KEINE_NEUEN_POSITIONEN,)
    # Am selben Tag bleibt es dabei, auch wenn der Verlust zurückgeht.
    assert _s(stand, verlust=0.0) == (stand, ())
    # Am nächsten Tag von selbst wieder aktiv.
    morgen, folgen = _s(stand, tag=T0 + timedelta(days=1))
    assert morgen.stufe is Stufe.ACTIVE and folgen == ()


def test_neuer_tag_mit_neuem_verlust_pausiert_sofort_wieder():
    stand, _ = _s(Stand(), verlust=0.03)
    stand, folgen = _s(stand, tag=T0 + timedelta(days=1), verlust=0.05)
    assert stand == Stand(Stufe.PAUSED_TODAY, T0 + timedelta(days=1))
    assert folgen == (Folge.KEINE_NEUEN_POSITIONEN,)


def test_stufe_2_drawdown_friert_ein():
    stand, folgen = _s(Stand(), dd=0.10)
    assert stand.stufe is Stufe.FROZEN
    assert folgen == (Folge.POSITIONEN_SCHLIESSEN, Folge.KAPITAL_EINFRIEREN, Folge.POST_MORTEM)


def test_stufe_3_drawdown_oder_regelbruch_legt_still():
    stand, folgen = _s(Stand(Stufe.FROZEN, T0), dd=0.20)
    assert stand.stufe is Stufe.RETIRED and folgen == (Folge.STILLLEGEN,)
    stand, folgen = _s(Stand(), bruch=True)
    assert stand.stufe is Stufe.RETIRED
    # Direkt von oben: was das Einfrieren angeordnet hätte, gilt mit.
    assert Folge.POSITIONEN_SCHLIESSEN in folgen and folgen[-1] is Folge.STILLLEGEN


def test_die_ganze_leiter_in_einer_simulation():
    """Ein Agent verliert Tag für Tag — alle drei Stufen, in dieser Reihenfolge."""
    stand, gesehen = Stand(), []
    for i, (verlust, dd) in enumerate([(0.0, 0.0), (0.025, 0.03), (0.01, 0.04), (0.07, 0.11), (0.1, 0.21)]):
        stand, folgen = _s(stand, tag=T0 + timedelta(days=i), verlust=verlust, dd=dd)
        gesehen.append(stand.stufe)
        assert bool(folgen) == (i in (1, 3, 4))
    assert gesehen == [Stufe.ACTIVE, Stufe.PAUSED_TODAY, Stufe.ACTIVE, Stufe.FROZEN, Stufe.RETIRED]


def test_kein_weg_zurueck_aus_frozen_und_retired():
    """Egal, welche Messwerte kommen: nach oben geht es nicht."""
    rng = random.Random(3)
    for start in (Stufe.FROZEN, Stufe.RETIRED):
        stand = Stand(start, T0)
        for i in range(500):
            stand, _ = _s(
                stand,
                tag=T0 + timedelta(days=i),
                verlust=rng.uniform(-0.5, 0.5),
                dd=rng.uniform(-0.5, 0.19 if start is Stufe.FROZEN else 0.9),
            )
            assert stand.stufe in ((Stufe.FROZEN,) if start is Stufe.FROZEN else (Stufe.RETIRED,))


def test_folgen_nur_beim_uebergang_nicht_bei_jedem_messpunkt():
    stand, _ = _s(Stand(), dd=0.12)
    assert _s(stand, dd=0.15) == (stand, ())


@pytest.mark.parametrize("wert", [float("nan"), float("inf"), None])
def test_ein_fehlendes_risikomass_ist_kein_null_verlust(wert):
    with pytest.raises(ValueError):
        _s(Stand(), verlust=wert)
    with pytest.raises(ValueError):
        _s(Stand(), dd=wert)


def test_das_ladder_modul_hat_keine_funktion_nach_oben():
    import quantrace.hub.ladder as ladder

    oeffentlich = {n for n in vars(ladder) if not n.startswith("_") and callable(getattr(ladder, n))}
    assert oeffentlich == {"schritt", "Stand", "Stufe", "Folge", "Verlustleiter", "Enum", "dataclass", "date"}


# ---------------------------------------------------------------------------
# Limits lesen: streng


def _yaml(tmp_path, body):
    p = tmp_path / "hub.yaml"
    p.write_text(body)
    return p


_GUT = """
mandate_defaults: {min_adv_usd: 1000000, min_price: 5, min_market_cap_usd: 100000000}
mandates:
  bot: {capital_usd: 10000, instruments: [AAA, BBB]}
"""


def test_mandat_wird_mit_vorgaben_gelesen(tmp_path):
    hl = limits.lade(_yaml(tmp_path, _GUT))
    m = hl.mandat("bot")
    assert m.capital_usd == 10_000 and m.instruments == {"AAA", "BBB"}
    assert m.stage == "paper" and m.max_leverage == 1.0 and m.ladder == Verlustleiter()
    assert hl.mandat("unbekannt") is None and hl.kill_switch is False


@pytest.mark.parametrize(
    "eintrag, wort",
    [
        ("{instruments: [AAA]}", "Pflichtfelder"),
        ("{capital_usd: 1000}", "Pflichtfelder"),
        ("{capital_usd: 1000, instruments: [AAA], max_postion_pct: 0.5}", "unbekannte Felder"),
        ("{capital_usd: 1000, instruments: AAA}", "Liste"),
        ("{capital_usd: -1, instruments: [AAA]}", "capital_usd"),
        ("{capital_usd: 1000, instruments: [AAA], stage: live}", "stage"),
        ("{capital_usd: 1000, instruments: [AAA], max_position_pct: 1.5}", "max_position_pct"),
        ("{capital_usd: 1000, instruments: [AAA], max_leverage: 1.5}", "levered"),
        ("{capital_usd: 1000, instruments: [AAA], max_leverage: 3, stage: levered}", "max_leverage"),
        ("{capital_usd: 1000, instruments: [AAA], max_gross_exposure: 1.5}", "max_gross_exposure"),
        ("{capital_usd: 1000, instruments: [AAA], loss_ladder: {drawdown_freeze: 0.3}}", "Verlustleiter"),
        ("{capital_usd: 1000, instruments: [AAA], loss_ladder: {tippfehler: 0.3}}", "tippfehler"),
    ],
)
def test_ein_fehlerhaftes_mandat_laedt_nicht(tmp_path, eintrag, wort):
    body = _GUT.replace("{capital_usd: 10000, instruments: [AAA, BBB]}", eintrag)
    with pytest.raises(MandatFehlerError, match=wort):
        limits.lade(_yaml(tmp_path, body))


def test_hebel_nur_auf_der_obersten_stufe_und_nie_ueber_dem_deckel(tmp_path):
    ok = "{capital_usd: 1000, instruments: [AAA], stage: levered, max_leverage: 2.0, max_gross_exposure: 2.0}"
    hl = limits.lade(_yaml(tmp_path, _GUT.replace("{capital_usd: 10000, instruments: [AAA, BBB]}", ok)))
    assert hl.mandat("bot").max_leverage == limits.MAX_HEBEL == 2.0


def test_ohne_datei_kein_handel(tmp_path):
    with pytest.raises(MandatFehlerError, match="fehlt"):
        limits.lade(tmp_path / "gibt_es_nicht.yaml")


def test_kill_switch_muss_ein_bool_sein(tmp_path):
    with pytest.raises(MandatFehlerError, match="kill_switch"):
        limits.lade(_yaml(tmp_path, 'kill_switch: "nein"\n'))


def test_die_eingecheckte_hub_yaml_laedt_und_gibt_niemandem_ein_mandat():
    """Stand heute: niemand handelt. Wer das ändert, ändert diesen Test mit —
    im selben PR, sichtbar im Review."""
    hl = limits.lade()
    assert hl.kill_switch is False
    assert hl.mandate == {}


# ---------------------------------------------------------------------------
# Der Executor: der einzige Weg zum Broker


class _Paper(Broker):
    """Füllt sofort zum Kurs. Hält Buch, damit die Folgen-Tests rechnen können."""

    def __init__(self, cash: float, markt):
        self.cash, self.markt = cash, markt
        self.pos: dict[str, float] = {}
        self.orders: list = []
        self.kaputt = False

    def connect(self): ...
    def disconnect(self): ...
    def is_connected(self):
        return True

    def account_value(self):
        return self.cash + sum(q * self.markt[s].price for s, q in self.pos.items())

    def positions(self):
        return []

    def cancel(self, order_id): ...

    def submit(self, order):
        if self.kaputt:
            raise RuntimeError("broker weg")
        self.orders.append(order)
        vz = 1 if order.side is BUY else -1
        self.pos[order.symbol] = self.pos.get(order.symbol, 0.0) + vz * order.quantity
        self.cash -= vz * order.quantity * self.markt[order.symbol].price
        return f"o{len(self.orders)}"


def _hl(mandat=None, kill=False) -> HubLimits:
    m = mandat or _mandat()
    return HubLimits(kill, {m.agent_id: m})


def _ein(auftrag, broker, *, hl=None, zustand=None, **kw):
    return executor.reiche_ein(
        auftrag,
        zustand=zustand or AgentZustand(equity_usd=10_000.0),
        markt=MARKT,
        broker=broker,
        hub_limits=hl or _hl(),
        client_tag="2026-11-10",
        **kw,
    )


def test_abgelehnt_heisst_der_broker_hoert_nichts():
    b = _Paper(10_000, MARKT)
    r = _ein(Auftrag("bot", "ZZZ", BUY, 1), b)
    assert isinstance(r.entscheidung, Reject) and not r.eingereicht and b.orders == []


def test_gekuerzt_heisst_die_gekuerzte_menge_geht_raus():
    b = _Paper(10_000, MARKT)
    r = _ein(Auftrag("bot", "AAA", BUY, 500), b)
    assert r.eingereicht and b.orders[0].quantity == pytest.approx(10)
    assert b.orders[0].client_id == "hub:bot:2026-11-10:AAA:BUY"


def test_broker_fehler_ist_ein_ergebnis_kein_absturz():
    b = _Paper(10_000, MARKT)
    b.kaputt = True
    r = _ein(Auftrag("bot", "AAA", BUY, 1), b)
    assert not r.eingereicht and "broker weg" in r.error


@pytest.mark.parametrize(
    "kw, stage",
    [
        (dict(), "mini_live"),
        (dict(allow_live=True), "mini_live"),
        (dict(live_enabled=True), "mini_live"),
        (dict(allow_live=True, live_enabled=True), "paper"),
    ],
)
def test_live_braucht_alle_drei_gates(kw, stage):
    b = _Paper(10_000, MARKT)
    with pytest.raises(LiveExecutionBlocked):
        _ein(Auftrag("bot", "AAA", BUY, 1), b, hl=_hl(_mandat(stage=stage)), live=True, **kw)
    assert b.orders == []


def test_live_ohne_mandat_ist_gesperrt():
    with pytest.raises(LiveExecutionBlocked):
        _ein(Auftrag("x", "AAA", BUY, 1), _Paper(1, MARKT), live=True, allow_live=True, live_enabled=True)


def test_kill_switch_aus_der_umgebung_wirkt_im_executor(monkeypatch):
    monkeypatch.setenv("QUANTRACE_HUB_KILL", "1")
    b = _Paper(10_000, MARKT)
    assert not _ein(Auftrag("bot", "AAA", BUY, 1), b).eingereicht
    # … und glattstellen geht trotzdem.
    z = AgentZustand(equity_usd=10_000.0, positions={"AAA": 4})
    for a in risk.glattstellen("bot", z.positions):
        assert _ein(a, b, zustand=z).eingereicht
    assert [(o.symbol, o.side, o.quantity) for o in b.orders] == [("AAA", SELL, 4)]


def test_ohne_uebergebene_limits_wird_die_datei_gelesen_und_die_ist_leer():
    b = _Paper(10_000, MARKT)
    r = executor.reiche_ein(
        Auftrag("bot", "AAA", BUY, 1), zustand=AgentZustand(10_000.0), markt=MARKT, broker=b
    )
    assert isinstance(r.entscheidung, Reject) and "kein Mandat" in r.entscheidung.gruende[0]


# ---------------------------------------------------------------------------
# Abnahme 1: keine Folge von Aktionen überschreitet die Limits


@pytest.mark.parametrize("seed", range(25))
def test_keine_folge_von_auftraegen_ueberschreitet_die_limits(seed):
    """Zufällige Aufträge, zufällig bewegte Kurse, wechselnde Zustände.

    Geprüft wird **nach jedem Kauf**, zu den Kursen des Kaufs: keine Position
    über ihrem Anteil, kein Brutto über der Grenze, nie ein negativer Bestand,
    nie ein Papier ausserhalb der Liste. Kursbewegungen *zwischen* den Orders
    können ein Limit reissen — das fängt die Verlustleiter, nicht die Box —,
    aber dann kauft die Box nichts dazu.
    """
    rng = random.Random(seed)
    mandat = _mandat(
        max_position_pct=rng.choice([0.05, 0.1, 0.25, 1.0]),
        max_gross_exposure=rng.choice([0.3, 0.6, 1.0]),
    )
    markt = {s: _papier(price=rng.uniform(6, 300)) for s in ("AAA", "BBB", "CCC")}
    markt["ZZZ"] = _papier(price=50.0)  # liquide, aber nicht auf der Liste
    broker = _Paper(mandat.capital_usd, markt)
    stufen = [Stufe.ACTIVE] * 6 + [Stufe.PAUSED_TODAY, Stufe.FROZEN]
    eps = 1e-6

    for _ in range(400):
        for s in ("AAA", "BBB", "CCC"):
            if rng.random() < 0.3:
                markt[s] = _papier(price=max(5.01, markt[s].price * rng.uniform(0.7, 1.4)))
        zustand = AgentZustand(
            equity_usd=broker.account_value(),
            positions=dict(broker.pos),
            leiter=Stand(rng.choice(stufen)),
        )
        auftrag = Auftrag(
            "bot",
            rng.choice(["AAA", "BBB", "CCC", "ZZZ"]),
            rng.choice([BUY, BUY, SELL]),
            rng.choice([rng.uniform(0.01, 5), rng.uniform(5, 500), 1e9]),
        )
        kill = rng.random() < 0.05
        vorher = len(broker.orders)
        erg = executor.reiche_ein(
            auftrag,
            zustand=zustand,
            markt=markt,
            broker=broker,
            hub_limits=HubLimits(kill, {"bot": mandat}),
            client_tag="t",
        )

        assert all(q >= -eps for q in broker.pos.values()), "Leerverkauf"
        assert "ZZZ" not in broker.pos, "Papier ausserhalb der Liste"
        if len(broker.orders) == vorher:
            continue
        order = broker.orders[-1]
        assert order.quantity <= auftrag.quantity + eps, "die Box hat erhöht"
        if order.side is BUY:
            assert erg.eingereicht and not kill and zustand.leiter.stufe is Stufe.ACTIVE
            basis = min(mandat.capital_usd, zustand.equity_usd)
            brutto = sum(q * markt[s].price for s, q in broker.pos.items())
            position = broker.pos[order.symbol] * markt[order.symbol].price
            assert position <= mandat.max_position_pct * basis + 1e-3
            assert brutto <= mandat.max_gross_exposure * basis + 1e-3


# ---------------------------------------------------------------------------
# Kein Weg am Executor vorbei


def test_nur_der_hub_executor_fasst_einen_broker_an():
    """Auf dem AST, wie in test_daily_plan: kein anderes Hub-Modul ruft
    `.submit()` oder importiert einen Broker-Adapter, den Paper-Executor oder
    setzt allow_live=True."""
    for pfad in sorted((REPO / "quantrace/hub").glob("*.py")):
        if pfad.name == "executor.py":
            continue
        tree = ast.parse(pfad.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert mod in ("quantrace.brokers.base",) or not mod.startswith(
                    ("quantrace.brokers", "quantrace.paper")
                ), f"{pfad.name} importiert {mod}"
                assert "execute_plan" not in [a.name for a in node.names]
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("quantrace.brokers") for a in node.names)
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                assert name not in ("submit", "execute_plan", "get_broker"), f"{pfad.name}:{node.lineno}"
                for kw in node.keywords:
                    assert not (
                        kw.arg in ("allow_live", "live_enabled", "live")
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ), f"{pfad.name}:{node.lineno} setzt {kw.arg}=True"


def test_der_hub_executor_selbst_schaltet_nie_live():
    tree = ast.parse((REPO / "quantrace/hub/executor.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                assert not (
                    kw.arg in ("allow_live", "live_enabled", "live")
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                )
        if isinstance(node, ast.FunctionDef) and node.name == "reiche_ein":
            vorgaben = {a.arg: d for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)}
            for name in ("live", "allow_live", "live_enabled"):
                assert isinstance(vorgaben[name], ast.Constant) and vorgaben[name].value is False


def test_kein_llm_agent_importiert_einen_broker_oder_einen_executor():
    """ADR-018, Prinzip 1: das LLM schreibt den Bot, der Bot handelt."""
    verboten = ("quantrace.brokers", "quantrace.paper.executor", "quantrace.hub.executor")
    for pfad in sorted((REPO / "agents").rglob("*.py")):
        tree = ast.parse(pfad.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            namen = []
            if isinstance(node, ast.ImportFrom):
                namen = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                namen = [a.name for a in node.names]
            for n in namen:
                assert not n.startswith(verboten), f"{pfad.relative_to(REPO)} importiert {n}"
