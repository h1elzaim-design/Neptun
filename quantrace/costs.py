"""Per-Asset-Class-Transaktionskosten — Auflösung Symbol → Kostenprofil.

Quelle ist `config/costs.yaml`: Kostenprofile pro Asset-Klasse (Fees, Impact-
Slippage, Quoted Spread — alles pro Order-Seite in bps), eine Symbol→Klasse-
Zuordnung und optionale Voll-Overrides pro Symbol. Dieses Modul ist rein
(nur Filesystem-Read, gecacht) — die Anwendung der Kosten passiert im
Backtest-Runner über per-Spalte-Arrays.

Warum überhaupt: ein flacher `slippage_bps=5` behandelt SPY (Spread < 1 bp)
und DBC (Spread ~5 bp) identisch — er bestraft Index-ETF-Strategien zu hart
und subventioniert Rohstoff-/Small-ETF-Strategien. Per-Klasse-Kosten sind die
Voraussetzung dafür, dass Cross-Universe-Vergleiche (Score, DSR) fair sind.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

from quantrace.models import SymbolCosts

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COSTS_PATH = _REPO_ROOT / "config" / "costs.yaml"

_PROFILE_KEYS = ("fees_bps", "slippage_bps", "spread_bps")


def _profile(raw: dict, *, asset_class: str, source: str) -> SymbolCosts:
    missing = [k for k in _PROFILE_KEYS if k not in raw]
    if missing:
        raise ValueError(f"costs.yaml: {source} fehlt {missing}")
    return SymbolCosts(
        asset_class=asset_class,
        fees_bps=float(raw["fees_bps"]),
        slippage_bps=float(raw["slippage_bps"]),
        spread_bps=float(raw["spread_bps"]),
    )


@lru_cache(maxsize=4)
def _load_config(path: Path) -> tuple[dict[str, SymbolCosts], dict[str, str], str, dict[str, SymbolCosts]]:
    """(class_profiles, symbol→class, default_class, symbol_overrides) — gecacht."""
    raw = yaml.safe_load(path.read_text()) or {}
    classes_raw = raw.get("classes") or {}
    if not classes_raw:
        raise ValueError(f"costs.yaml ({path}): keine `classes` definiert")

    profiles = {
        name: _profile(cfg or {}, asset_class=name, source=f"classes.{name}")
        for name, cfg in classes_raw.items()
    }

    default_class = str(raw.get("default_class") or "")
    if default_class not in profiles:
        raise ValueError(
            f"costs.yaml ({path}): default_class '{default_class}' ist keine definierte Klasse"
        )

    symbol_map = {str(k).upper(): str(v) for k, v in (raw.get("symbols") or {}).items()}
    unknown_classes = sorted({c for c in symbol_map.values() if c not in profiles})
    if unknown_classes:
        raise ValueError(f"costs.yaml ({path}): symbols verweisen auf unbekannte Klassen {unknown_classes}")

    # Overrides behalten das Klassen-Label aus der Symbol-Zuordnung (falls
    # vorhanden), damit die persistierte Kosten-Tabelle lesbar bleibt.
    overrides = {
        str(sym).upper(): _profile(
            cfg or {},
            asset_class=symbol_map.get(str(sym).upper(), "symbol_override"),
            source=f"symbol_overrides.{sym}",
        )
        for sym, cfg in (raw.get("symbol_overrides") or {}).items()
    }

    return profiles, symbol_map, default_class, overrides


def resolve_symbol_costs(
    symbols: list[str] | tuple[str, ...],
    *,
    config_path: Path | None = None,
) -> dict[str, SymbolCosts]:
    """Kostenprofil pro Symbol: Override > Klassen-Zuordnung > default_class.

    Symbole ohne Zuordnung fallen auf ``default_class`` zurück und werden als
    Warnung geloggt — bei Universe-Erweiterungen gehört die Klassifikation in
    `config/costs.yaml` nachgezogen (siehe Issue #35).
    """
    profiles, symbol_map, default_class, overrides = _load_config(
        config_path or DEFAULT_COSTS_PATH
    )

    resolved: dict[str, SymbolCosts] = {}
    unmapped: list[str] = []
    for sym in symbols:
        key = sym.upper()
        if key in overrides:
            resolved[sym] = overrides[key]
        elif key in symbol_map:
            resolved[sym] = profiles[symbol_map[key]]
        else:
            resolved[sym] = profiles[default_class]
            unmapped.append(sym)

    if unmapped:
        log.warning(
            "costs: %s nicht in config/costs.yaml klassifiziert — default_class '%s' angenommen",
            unmapped,
            default_class,
        )
    return resolved


__all__ = ["DEFAULT_COSTS_PATH", "resolve_symbol_costs"]
