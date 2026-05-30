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

    def test_deterministic(self):
        meta = {
            "delisted_included": False,
            "provider": "openbb_yfinance",
            "last_audit_date": "2026-01-01",
        }
        a = audit_universe("u", meta)
        b = audit_universe("u", meta)
        assert a == b
