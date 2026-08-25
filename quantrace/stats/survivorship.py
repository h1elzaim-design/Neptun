"""Survivorship-bias risk audit for a universe definition.

A universe of *currently-listed* tickers systematically excludes companies
that went bankrupt, were acquired, or delisted. Backtesting on such a
universe inflates returns by an amount that is hard to quantify ex-post but
can be material (5–10% annualised in extreme cases — see Brown & Goetzmann
1995). This module classifies the survivorship risk of a universe definition
into HIGH / MEDIUM / LOW, with explicit reasoning, so the webapp can surface
a warning at the point of strategy selection.

Inputs
------
We do not have a survivorship-clean reference dataset to compare against;
instead we rely on metadata declared in `data/universes/*.yaml`:

    symbols: [SPY, QQQ, ...]
    provider: openbb_yfinance        # required
    delisted_included: false         # required
    point_in_time: false             # optional
    last_audit_date: 2026-04-15      # optional
    source_notes: "..."              # optional

The function is pure: same metadata → same audit result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

RiskLevel = Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]


# Static knowledge base: how each provider treats delisted symbols by default.
# Conservative defaults; please update when a new provider is added.
PROVIDER_PROFILE: dict[str, dict[str, Any]] = {
    "openbb_yfinance": {
        "delivers_delisted_by_default": False,
        "point_in_time_capable": False,
        "trustworthy_for_pit": False,
    },
    "openbb_fmp": {
        "delivers_delisted_by_default": False,
        "point_in_time_capable": True,
        "trustworthy_for_pit": True,
    },
    "openbb_polygon": {
        "delivers_delisted_by_default": True,
        "point_in_time_capable": True,
        "trustworthy_for_pit": True,
    },
    "openbb_intrinio": {
        "delivers_delisted_by_default": True,
        "point_in_time_capable": True,
        "trustworthy_for_pit": True,
    },
    "openbb_alphavantage": {
        "delivers_delisted_by_default": False,
        "point_in_time_capable": False,
        "trustworthy_for_pit": False,
    },
    "eodhd": {
        # Der US-Bulk liefert **Tagesquerschnitte**: was am 2008-09-12 handelte,
        # steht im Querschnitt vom 2008-09-12 — LEH, BSC, WM und AIG inklusive.
        # Die Toten sind nicht nachträglich ergänzt, sie sind nie verschwunden.
        #
        # **Was das nicht heißt.** Survivorship-freie *Kurse* sind nicht
        # dasselbe wie ein survivorship-freies *Universum*. Eine handverlesene
        # `symbols:`-Liste von heute bleibt eine Liste der Überlebenden, egal
        # wie ehrlich die Quelle ist — der Provider kann die Auswahl nicht
        # heilen. Deshalb bleibt `delisted_included` die Angabe, die zählt, und
        # dieses Profil verfeinert nur.
        "delivers_delisted_by_default": True,
        "point_in_time_capable": True,
        "trustworthy_for_pit": True,
    },
    "tiingo": {
        # Bleibt in der Tabelle, obwohl der Ladepfad am 2026-08-13 entfernt
        # wurde: alte Universe-YAMLs und archivierte Vault-Notes nennen den
        # Provider weiterhin, und für die ist die Einordnung genau dann
        # relevant, wenn jemand eine alte Zahl nachliest.
        #
        # Tiingo EOD covers currently-listed tickers (and some delisted ones
        # only if queried by symbol); it carries no point-in-time index
        # membership, so a static `symbols:` list is survivor-only by default.
        "delivers_delisted_by_default": False,
        "point_in_time_capable": False,
        "trustworthy_for_pit": False,
    },
    "manual": {
        "delivers_delisted_by_default": False,
        "point_in_time_capable": False,
        "trustworthy_for_pit": False,
    },
}


@dataclass(frozen=True, slots=True)
class SurvivorshipAudit:
    universe: str
    risk: RiskLevel
    reasons: list[str]
    recommendations: list[str]
    provider: str | None
    delisted_included: bool | None
    point_in_time: bool | None
    last_audit_date: date | None
    audit_stale: bool


_AUDIT_STALE_DAYS = 180


def audit_universe(name: str, metadata: dict[str, Any]) -> SurvivorshipAudit:
    """Classify the survivorship-bias risk of a universe definition.

    Parameters
    ----------
    name:
        Universe identifier (file stem).
    metadata:
        Raw dict loaded from the universe YAML.

    Returns
    -------
    SurvivorshipAudit with `risk` ∈ {HIGH, MEDIUM, LOW, UNKNOWN} and a list of
    plain-language `reasons` and `recommendations` suitable for direct UI
    display.
    """
    provider = _safe_str(metadata.get("provider"))
    delisted_included = _safe_bool(metadata.get("delisted_included"))
    point_in_time = _safe_bool(metadata.get("point_in_time"))
    last_audit_date = _safe_date(metadata.get("last_audit_date"))

    reasons: list[str] = []
    recommendations: list[str] = []
    audit_stale = (
        last_audit_date is None
        or (date.today() - last_audit_date) > timedelta(days=_AUDIT_STALE_DAYS)
    )

    profile = PROVIDER_PROFILE.get(provider or "")

    # --- Risk classification ------------------------------------------------
    if delisted_included is True:
        risk: RiskLevel = "LOW"
        reasons.append(
            "Universe metadata declares delisted_included=true — "
            "survivorship bias is structurally avoided."
        )
        if point_in_time is not True:
            reasons.append(
                "point_in_time is not explicitly true — index-membership "
                "drift may still cause subtler look-ahead. Verify before live use."
            )
            recommendations.append(
                "Add point_in_time:true once you have verified the "
                "as-of membership reconstruction."
            )
    elif delisted_included is False:
        risk = "HIGH"
        reasons.append(
            "Universe metadata declares delisted_included=false — "
            "all returns assume the survivors only. Inflation of CAGR / Sharpe "
            "is expected; do not promote to live trading from this universe alone."
        )
        recommendations.append(
            "Re-run the backtest on a survivorship-free universe before approval."
        )
    else:
        risk = "UNKNOWN"
        reasons.append(
            "Universe metadata is missing `delisted_included` — we cannot tell "
            "whether the data is survivorship-free. Treat as HIGH risk until proven."
        )
        recommendations.append(
            "Add `delisted_included: true|false` to the universe YAML."
        )

    # Refine by provider profile
    if profile is not None:
        if not profile["trustworthy_for_pit"] and risk == "LOW":
            risk = "MEDIUM"
            reasons.append(
                f"Provider `{provider}` is not generally trusted for point-in-time data. "
                "The LOW classification has been downgraded to MEDIUM."
            )
        if not profile["delivers_delisted_by_default"]:
            reasons.append(
                f"Provider `{provider}` does not include delisted tickers by default — "
                "verify the data-loading pipeline actually requested them."
            )
    elif provider:
        reasons.append(
            f"Provider `{provider}` is unknown to the survivorship audit table. "
            "Risk cannot be refined from provider characteristics."
        )

    if audit_stale:
        reasons.append(
            "Universe audit is missing or older than 180 days — re-verify before "
            "treating the audit as authoritative."
        )
        recommendations.append("Update `last_audit_date` in the universe YAML.")
        if risk == "LOW":
            risk = "MEDIUM"

    return SurvivorshipAudit(
        universe=name,
        risk=risk,
        reasons=reasons,
        recommendations=recommendations,
        provider=provider,
        delisted_included=delisted_included,
        point_in_time=point_in_time,
        last_audit_date=last_audit_date,
        audit_stale=audit_stale,
    )


# -----------------------------------------------------------------------------
# Defensive parsing helpers
# -----------------------------------------------------------------------------

def _safe_str(v: Any) -> str | None:
    return str(v) if isinstance(v, str) and v else None


def _safe_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    return None


def _safe_date(v: Any) -> date | None:
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        try:
            return date.fromisoformat(v)
        except ValueError:
            return None
    return None
