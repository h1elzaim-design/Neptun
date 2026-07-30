"""Provider-Bootstrap für OpenBB.

OpenBB liest Credentials aus dem User-Objekt, nicht aus Env-Vars. Damit
`provider="tiingo"` etc. in `data_agent.load_universe` direkt funktioniert,
registrieren wir die Keys einmal pro Prozess beim ersten Fetch.

Unterstützte Provider (alle kostenlos verfügbar):
    yfinance      kein Key, Default für lokale Iteration
    tiingo        TIINGO_TOKEN — empfohlen für Research (sauberer als yfinance)
    fmp           FMP_API_KEY — Financial Modeling Prep
    polygon       POLYGON_API_KEY — Polygon.io

Default-Provider via QUANTRACE_DATA_PROVIDER, sonst yfinance.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_CREDENTIAL_ENV_MAP: dict[str, tuple[str, str]] = {
    "tiingo": ("TIINGO_TOKEN", "tiingo_token"),
    "fmp": ("FMP_API_KEY", "fmp_api_key"),
    "polygon": ("POLYGON_API_KEY", "polygon_api_key"),
    "intrinio": ("INTRINIO_API_KEY", "intrinio_api_key"),
    "alpha_vantage": ("ALPHA_VANTAGE_API_KEY", "alpha_vantage_api_key"),
}

_DEFAULT_PROVIDER = "yfinance"
_bootstrapped = False


def default_provider() -> str:
    """Default-Provider für Backtest-Fetches.

    Priorität: `QUANTRACE_DATA_PROVIDER` Env > Tiingo (wenn Token gesetzt) > yfinance.
    """
    env = os.environ.get("QUANTRACE_DATA_PROVIDER", "").strip().lower()
    if env:
        return env
    if os.environ.get("TIINGO_TOKEN"):
        return "tiingo"
    return _DEFAULT_PROVIDER


def bootstrap_credentials(force: bool = False) -> list[str]:
    """Registriert alle gesetzten Provider-Keys einmalig mit OpenBB.

    Returns: Liste der erfolgreich registrierten Provider-Namen.
    Idempotent — mehrfaches Aufrufen ist gefahrlos. Wenn `openbb` nicht
    installiert ist, wird nichts geloggt und eine leere Liste zurückgegeben
    (der eigentliche ImportError fällt erst in `data_agent` an, wo er gehört).
    """
    global _bootstrapped
    if _bootstrapped and not force:
        return []

    try:
        from openbb import obb
    except ImportError:
        return []

    registered: list[str] = []
    for provider, (env_var, cred_attr) in _CREDENTIAL_ENV_MAP.items():
        token = os.environ.get(env_var, "").strip()
        if not token:
            continue
        try:
            setattr(obb.user.credentials, cred_attr, token)
            registered.append(provider)
        except Exception as exc:
            log.warning("Provider %s konnte nicht registriert werden: %s", provider, exc)

    if registered:
        log.info("OpenBB-Credentials registriert: %s", ", ".join(registered))
    _bootstrapped = True
    return registered


def reset_bootstrap_for_tests() -> None:
    """Setzt das Bootstrap-Flag zurück. Nur für Tests."""
    global _bootstrapped
    _bootstrapped = False
