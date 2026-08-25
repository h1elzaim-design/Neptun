"""Unit tests for quantrace.stats.survivorship."""

from __future__ import annotations

from datetime import date, timedelta

from quantrace.stats.survivorship import audit_universe


class TestSurvivorshipAudit:
    def test_missing_metadata_is_unknown(self):
        out = audit_universe("legacy_universe", {})
        assert out.risk in {"UNKNOWN", "MEDIUM"}  # MEDIUM if stale-bump applies
        assert any("delisted_included" in r for r in out.reasons)
        assert any("delisted_included" in r for r in out.recommendations)

    def test_delisted_false_is_high(self):
        out = audit_universe(
            "us_core_etfs",
            {
                "delisted_included": False,
                "provider": "openbb_yfinance",
                "last_audit_date": date.today().isoformat(),
            },
        )
        assert out.risk == "HIGH"
        assert any("survivors only" in r for r in out.reasons)

    def test_delisted_true_with_pit_and_trusted_provider_is_low(self):
        out = audit_universe(
            "russell_1000_pit",
            {
                "delisted_included": True,
                "point_in_time": True,
                "provider": "openbb_polygon",
                "last_audit_date": date.today().isoformat(),
            },
        )
        assert out.risk == "LOW"

    def test_trusted_data_but_untrusted_pit_provider_downgraded_to_medium(self):
        out = audit_universe(
            "ad_hoc",
            {
                "delisted_included": True,
                "point_in_time": True,
                "provider": "openbb_yfinance",  # not trusted for PIT
                "last_audit_date": date.today().isoformat(),
            },
        )
        assert out.risk == "MEDIUM"
        assert any("not generally trusted" in r for r in out.reasons)

    def test_stale_audit_downgrades_low_to_medium(self):
        stale = (date.today() - timedelta(days=365)).isoformat()
        out = audit_universe(
            "old",
            {
                "delisted_included": True,
                "point_in_time": True,
                "provider": "openbb_polygon",
                "last_audit_date": stale,
            },
        )
        assert out.audit_stale is True
        assert out.risk == "MEDIUM"

    def test_unknown_provider_recorded(self):
        out = audit_universe(
            "made_up",
            {
                "delisted_included": True,
                "provider": "exotic_provider",
                "last_audit_date": date.today().isoformat(),
            },
        )
        assert any("unknown to the survivorship audit table" in r for r in out.reasons)

    def test_tiingo_is_a_known_provider(self):
        # tiingo is the production provider — it must be recognised so the audit
        # refines instead of bailing out with "unknown provider".
        out = audit_universe(
            "us_core_etfs",
            {
                "delisted_included": True,
                "provider": "tiingo",
                "last_audit_date": date.today().isoformat(),
            },
        )
        assert not any("unknown to the survivorship audit table" in r for r in out.reasons)
        # Not PIT-trustworthy → a declared-LOW universe is refined down to MEDIUM.
        assert out.risk == "MEDIUM"
        assert any("tiingo" in r for r in out.reasons)

    def test_deterministic(self):
        meta = {
            "delisted_included": False,
            "provider": "openbb_yfinance",
            "last_audit_date": "2026-01-01",
        }
        a = audit_universe("u", meta)
        b = audit_universe("u", meta)
        assert a == b


# ---------------------------------------------------------------------------
# EODHD — der Bulk ist survivorship-frei, die Auswahl deshalb noch nicht


def test_eodhd_ist_der_audit_tabelle_bekannt():
    """Ohne Eintrag wäre jedes Universum nach dem Provider-Wechsel „unbekannt",
    und die Begründung im Audit fiele auf „Risiko nicht verfeinerbar" zurück —
    ausgerechnet bei der Quelle, deren ganzer Zweck die Toten sind."""
    from quantrace.stats.survivorship import PROVIDER_PROFILE

    profil = PROVIDER_PROFILE["eodhd"]
    assert profil["delivers_delisted_by_default"] is True
    assert profil["point_in_time_capable"] is True


def test_eodhd_heilt_eine_ueberlebenden_liste_nicht():
    """Der Kern der Unterscheidung: survivorship-freie *Kurse* sind nicht
    dasselbe wie ein survivorship-freies *Universum*. Eine handverlesene
    Symbolliste von heute bleibt eine Liste der Überlebenden — der Provider
    darf das nicht wegdefinieren."""
    from quantrace.stats.survivorship import audit_universe

    meta = {
        "provider": "eodhd",
        "symbols": ["SPY", "QQQ"],
        "delisted_included": False,
        "last_audit_date": date.today().isoformat(),
    }
    assert audit_universe("handverlesen", meta).risk == "HIGH"
