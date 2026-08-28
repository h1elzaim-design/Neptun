"""Die aufrufbare Grenze — was der Server anbietet und was er niemals anbietet.

Diese Tests prüfen nicht, ob der Server *funktioniert*, sondern ob er **dicht**
ist. Der Unterschied ist der Punkt: ein Leck auf dieser Oberfläche sieht im
Betrieb aus wie eine Antwort. Es gibt keine Fehlermeldung, wenn ein Tool zu
viel verrät.

Zwei Eigenschaften tragen alles Weitere und stehen deshalb als eigene Tests da:

* **Die Namensmenge ist eingefroren.** Ein neues Tool auf dieser Oberfläche
  bleibt eine Entscheidung und rutscht nicht in einem großen Diff mit — genau
  wie `tests/test_agent_tools.py` es für die interne Registry hält.
* **Das Paket importiert nichts aus `agents/`.** Das ist die Bedingung dafür,
  dass der Server selbst öffentlich sein darf: er beschreibt die Maschine, nicht
  das Labor.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from quantrace.mcp_server import tools

REPO_ROOT = Path(__file__).resolve().parent.parent
PAKET = REPO_ROOT / "quantrace" / "mcp_server"

#: Die v1-Oberfläche, wörtlich. Wer sie ändert, ändert diesen Test mit — und
#: beantwortet dabei die Frage aus `docs/MCP_BOUNDARY.md`: was rekonstruiert
#: ein Aufrufer, der das neue Tool tausendmal aufruft?
V1 = {
    "platform_capabilities",
    "assess_feasibility",
    "list_graph_nodes",
    "validate_graph_strategy",
}


def _manifest(tmp_path: Path, callable_: list[dict], denied: list[str] | None = None) -> Path:
    p = tmp_path / "callable_manifest.yaml"
    p.write_text(
        yaml.safe_dump({"callable": callable_, "denied": denied or []}), encoding="utf-8"
    )
    return p


class TestOberflaeche:
    def test_namensmenge_ist_eingefroren(self):
        assert set(tools.registry()) == V1

    def test_jedes_tool_hat_ein_schema_und_eine_beschreibung(self):
        for t in tools.registry().values():
            assert t.parameters.get("type") == "object"
            assert "properties" in t.parameters
            assert len(t.description) > 40, f"{t.name} beschreibt sich nicht"

    def test_deckel_kommt_aus_dem_manifest(self):
        for t in tools.registry().values():
            assert t.max_bytes > 0


class TestWasNichtGeht:
    """Die Verbotsliste ist Dokumentation *und* Testgegenstand."""

    def test_kein_verbotener_name_ist_aufrufbar(self):
        verboten = tools.load_manifest().denied_names
        assert verboten, "eine leere Verbotsliste prüft nichts"
        for name in verboten:
            with pytest.raises(tools.ToolNotAllowedError):
                tools.call(name)

    @pytest.mark.parametrize(
        "name",
        ["read_vault_note", "get_candles", "run_evaluation", "start_backtest", "list_workspaces"],
    )
    def test_die_teuren_faelle_namentlich(self, name):
        """Fünf, die einzeln benannt gehören — je einer je Leckklasse.

        Ein `for` über die Manifest-Liste bestünde auch, wenn jemand die Liste
        leert. Diese fünf stehen im Test, nicht in der Datei.
        """
        assert name not in tools.registry()
        with pytest.raises(tools.ToolNotAllowedError):
            tools.call(name)

    def test_unbekannter_name_ist_kein_key_error(self):
        with pytest.raises(tools.ToolNotAllowedError):
            tools.call("gibt_es_nicht")


class TestManifestIstDasGate:
    def test_engeres_manifest_verengt(self, tmp_path, monkeypatch):
        p = _manifest(tmp_path, [{"name": "list_graph_nodes", "max_bytes": 262144}])
        monkeypatch.setenv(tools._MANIFEST_ENV, str(p))
        assert set(tools.registry()) == {"list_graph_nodes"}
        with pytest.raises(tools.ToolNotAllowedError):
            tools.call("platform_capabilities")

    def test_env_auf_nichts_faellt_nicht_zurueck(self, tmp_path, monkeypatch):
        """Der Tippfehler-Fall, und der teuerste.

        Wer eine engere Liste setzt und sich im Pfad vertippt, bekäme bei einem
        Rückfall still die volle Oberfläche — und im Log wäre das von einer
        korrekten Konfiguration nicht zu unterscheiden.
        """
        monkeypatch.setenv(tools._MANIFEST_ENV, str(tmp_path / "gibt-es-nicht.yaml"))
        with pytest.raises(tools.ManifestMissingError):
            tools.registry()

    def test_name_in_beiden_listen_ist_ein_fehler(self, tmp_path, monkeypatch):
        p = _manifest(tmp_path, [{"name": "list_graph_nodes"}], denied=["list_graph_nodes"])
        monkeypatch.setenv(tools._MANIFEST_ENV, str(p))
        with pytest.raises(ValueError, match="widerspricht"):
            tools.registry()

    def test_versprechen_ohne_handler_faellt_auf(self, tmp_path, monkeypatch):
        p = _manifest(tmp_path, [{"name": "erfundenes_tool"}])
        monkeypatch.setenv(tools._MANIFEST_ENV, str(p))
        with pytest.raises(ValueError, match="Handler"):
            tools.registry()


class TestDeckel:
    def test_zu_grosse_antwort_wird_abgelehnt_nicht_gekuerzt(self, tmp_path, monkeypatch):
        p = _manifest(tmp_path, [{"name": "list_graph_nodes", "max_bytes": 64}])
        monkeypatch.setenv(tools._MANIFEST_ENV, str(p))
        with pytest.raises(tools.ResponseTooLargeError) as exc:
            tools.call("list_graph_nodes")
        # Beide Zahlen in der Meldung: „zu groß" allein sagt dem Aufrufer nicht,
        # ob er das Fenster halbieren oder hundertteln muss.
        assert "64" in str(exc.value)

    def test_der_echte_katalog_passt_unter_seinen_deckel(self):
        antwort = tools.call("list_graph_nodes")
        daten = json.loads(antwort)
        assert daten["nodes"] and daten["presets"]
        assert len(antwort.encode("utf-8")) <= tools.registry()["list_graph_nodes"].max_bytes


class TestStruktur:
    """Eigenschaften, die den Server öffentlich-fähig halten."""

    def _top_level_importe(self, datei: Path) -> set[str]:
        baum = ast.parse(datei.read_text(encoding="utf-8"))
        namen: set[str] = set()
        for knoten in baum.body:  # nur Modulebene — Lazy-Importe sind erlaubt
            if isinstance(knoten, ast.Import):
                namen.update(a.name.split(".")[0] for a in knoten.names)
            elif isinstance(knoten, ast.ImportFrom) and knoten.module:
                namen.add(knoten.module.split(".")[0])
        return namen

    @pytest.mark.parametrize("datei", sorted(PAKET.glob("*.py")), ids=lambda p: p.name)
    def test_kein_import_aus_agents(self, datei):
        """Die Bedingung dafür, dass dieser Server selbst public sein darf.

        Ein Import aus `agents/` zöge Prompts und Vault-Orchestrierung in ein
        Paket, das nach Neptun exportiert werden können soll — und die Grenze
        aus `OPEN_SOURCE_BOUNDARY.md` wäre gebrochen, ohne dass jemand eine
        Datei verschoben hat.
        """
        quelle = datei.read_text(encoding="utf-8")
        baum = ast.parse(quelle)
        for knoten in ast.walk(baum):  # hier auch Lazy-Importe: sie zählen
            if isinstance(knoten, ast.ImportFrom) and (knoten.module or "").startswith("agents"):
                pytest.fail(f"{datei.name} importiert aus agents: {knoten.module}")
            if isinstance(knoten, ast.Import):
                for a in knoten.names:
                    assert not a.name.startswith("agents"), f"{datei.name}: {a.name}"

    def test_die_grenze_laeuft_ohne_das_sdk(self):
        """`tools.py` darf das MCP-SDK nicht auf Modulebene brauchen.

        Sonst prüfte CI die Grenze nur mit installiertem Extra — und ohne es
        gar nicht, also genau dort nicht, wo jemand den Server schlank
        betreiben will.
        """
        assert "mcp" not in self._top_level_importe(PAKET / "tools.py")


@pytest.fixture(scope="module")
def server():
    pytest.importorskip("mcp", reason='Extra nicht installiert: pip install -e ".[mcp]"')
    from quantrace.mcp_server.server import build_server

    return build_server()


@pytest.fixture(scope="module")
def angeboten(server):
    import asyncio

    return {t.name: t for t in asyncio.run(server.list_tools())}


class TestTransport:
    """Der Server selbst — nur wenn das Extra installiert ist.

    Ohne SDK übersprungen statt rot: die Grenze oben ist die Zusage, die immer
    gelten muss, und sie prüft sich ohne Abhängigkeit. Hier geht es um das
    Stück, das ohne `pip install -e ".[mcp]"` gar nicht existiert.
    """

    def test_der_server_bietet_genau_das_manifest_an(self, angeboten):
        assert set(angeboten) == V1

    def test_wire_schema_und_dokumentiertes_schema_nennen_dieselben_parameter(self, angeboten):
        """Die eine Doppelung im Entwurf — hier wird sie zur geprüften Zusage.

        Das SDK leitet sein Schema aus der Handler-Signatur ab, `tools.py` führt
        eine lesbare Fassung derselben Zusage. Zwei Fassungen ohne
        Gleichheitsbeweis wären zwei Wahrheiten, und die falsche stünde in der
        Beschreibung, die ein fremdes Modell liest.
        """
        for name, t in tools.registry().items():
            dokumentiert = set(t.parameters.get("properties", {}))
            verdrahtet = set((angeboten[name].input_schema or {}).get("properties", {}))
            assert dokumentiert == verdrahtet, f"{name}: {dokumentiert ^ verdrahtet}"

    def test_ein_aufruf_geht_durch_das_gate(self, server):
        import asyncio

        antwort = asyncio.run(server.call_tool("list_graph_nodes", {}))
        text = antwort.content[0].text
        assert json.loads(text)["nodes"]

    def test_ein_verbotenes_tool_existiert_gar_nicht(self, server):
        import asyncio

        with pytest.raises(Exception, match="read_vault_note"):
            asyncio.run(server.call_tool("read_vault_note", {"path": "x"}))


class TestKeineAuskunftUeberDenBestand:
    """Das Abo deckt keine Weitergabe an Dritte — also darf keine Antwort von
    unserem Datenbestand handeln.

    Entschieden am 2026-08-28. Die Folge ist schärfer, als sie klingt: es reicht
    nicht, keine Kurse auszuliefern. Auch „Schicht 2 ist nicht gebaut" oder
    „42.708 Instrumente, 2000-01-03 … 2015-02-04" sind Auskünfte über diesen
    Bestand — und sie kämen als *Begründung* in eine Antwort, ohne dass jemand
    eine Zeile Kursdaten angefragt hätte.
    """

    #: Spuren, die nur aus dieser Installation stammen können.
    VERRAETERISCH = ("Schicht 2", "scripts/", "Lake", "lake", "Instrumente", "#245", "#265")

    def test_capabilities_nennt_den_eigenen_bestand_nicht(self):
        cap = json.loads(tools.call("platform_capabilities"))
        assert "lake" not in cap
        for eintrag in cap["data"]:
            assert set(eintrag) == {"key", "label"}, eintrag

    def test_capabilities_bleibt_trotzdem_nuetzlich(self):
        """Weglassen ist nicht Ausdünnen: was die Engine kann, steht weiter da."""
        cap = json.loads(tools.call("platform_capabilities"))
        assert cap["n_nodes"] > 10
        assert cap["nodes"] and cap["universes"] and cap["data"]

    def test_kein_urteil_verraet_den_ladestand(self):
        for key in (e["key"] for e in json.loads(tools.call("platform_capabilities"))["data"]):
            antwort = tools.call(
                "assess_feasibility",
                {"needs_data": [key], "start": "2012-01-01", "end": "2014-12-31",
                 "data_from": "2010-01-01", "data_to": "2015-12-31"},
            )
            for spur in self.VERRAETERISCH:
                assert spur not in antwort, f"{key}: '{spur}' in der Antwort"

    def test_ohne_eigenes_fenster_wird_es_nicht_geprueft_aber_gesagt(self):
        """Lieber ausdrücklich unvollständig als still auf unseren Stand bezogen."""
        antwort = json.loads(
            tools.call("assess_feasibility", {"start": "1990-01-01", "end": "2030-12-31"})
        )
        assert antwort["feasible"] is True
        assert any("NICHT geprüft" in c for c in antwort["caveats"])

    def test_halbes_fenster_ist_ein_fehler_keine_annahme(self):
        antwort = json.loads(
            tools.call("assess_feasibility", {"start": "2012-01-01", "data_from": "2010-01-01"})
        )
        assert antwort["feasible"] is False
        assert any("gehören zusammen" in b for b in antwort["blockers"])

    def test_erfundene_datenklasse_bleibt_ein_blocker(self):
        """Die Existenzprüfung überlebt — sie ist eine Aussage über die Engine."""
        antwort = json.loads(tools.call("assess_feasibility", {"needs_data": ["optionsflow"]}))
        assert antwort["feasible"] is False
        assert antwort["missing_data"] == ["optionsflow"]
