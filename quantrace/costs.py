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


#: Median-Tagesvolumen (in $) → Kostenklasse, absteigend geprüft.
#:
#: Die Grenzen sind Schätzungen, keine Messungen — wie der Rest dieser Datei.
#: Was sie belastbar macht, ist die Richtung: unterschätztes Volumen führt zu
#: *höheren* angesetzten Kosten, nie zu niedrigeren.
_LIQUIDITY_CLASSES: tuple[tuple[float, str], ...] = (
    (25_000_000.0, "us_equity_single"),
    (5_000_000.0, "us_equity_liquid"),
    (1_000_000.0, "us_equity_smallcap"),
)


class UnpriceableError(ValueError):
    """Für diese Liquidität gibt es keine ehrliche bps-Zahl."""


def class_for_liquidity(median_dollar_volume: float) -> str:
    """Kostenklasse aus dem **gemessenen** Liquiditätsboden einer Auswahl.

    Gedacht für konstruierte Universen (`quantrace.screen`): dort ist die
    Liquidität nicht geraten, sondern das Auswahlkriterium selbst. Übergeben
    gehört der Boden — das kleinste Dollarvolumen unter den Ausgewählten —,
    nicht der Durchschnitt: der ganze Korb wird nach seinem schwächsten
    Mitglied bepreist.

    Raises
    ------
    UnpriceableError
        Unter 1 Mio $ Median-Tagesvolumen. Eine Market-Order nahe dem Close
        bewegt dort den Kurs selbst; jede feste bps-Zahl wäre erfunden. Eine
        Fehlermeldung ist die ehrlichere Antwort als eine große Zahl.
    """
    for schwelle, klasse in _LIQUIDITY_CLASSES:
        if median_dollar_volume >= schwelle:
            return klasse
    raise UnpriceableError(
        f"Liquiditätsboden {median_dollar_volume:,.0f} $ liegt unter "
        f"{_LIQUIDITY_CLASSES[-1][0]:,.0f} $ Median-Tagesvolumen. Für so dünne "
        "Titel gibt es keine feste bps-Zahl, die stimmt — setz die Schwelle "
        "höher oder ein top_n, statt Kosten zu erfinden."
    )


def resolve_symbol_costs(
    symbols: list[str] | tuple[str, ...],
    *,
    config_path: Path | None = None,
    fallback_class: str | None = None,
) -> dict[str, SymbolCosts]:
    """Kostenprofil pro Symbol: Override > Klassen-Zuordnung > Fallback.

    Symbole ohne Zuordnung fallen auf ``default_class`` zurück und werden als
    Warnung geloggt — bei Universe-Erweiterungen gehört die Klassifikation in
    `config/costs.yaml` nachgezogen (siehe Issue #35).

    ``fallback_class`` ersetzt diesen Rückfall für einen Aufruf. Gedacht für
    **konstruierte** Universen, die ihre Klasse im YAML deklarieren
    (``cost_class:``): dort steht die Klasse nicht pro Symbol, weil die Regel
    sie für alle Mitglieder gemeinsam festlegt. Ohne diesen Parameter bekäme
    ein Korb aus 400 Small Caps die Kosten von SPY — der `default_class` ist
    das *günstigste* Profil der Datei, und stillschweigend anzuwenden wäre
    dieselbe Falle wie fehlende Kosten als 0,0 zu lesen.

    Es bleibt ein Fallback, kein Override: ein Symbol, das in ``symbols:``
    steht, behält seine eigene Klasse.
    """
    profiles, symbol_map, default_class, overrides = _load_config(
        config_path or DEFAULT_COSTS_PATH
    )

    if fallback_class is not None and fallback_class not in profiles:
        raise ValueError(
            f"fallback_class '{fallback_class}' ist keine Klasse in costs.yaml "
            f"(bekannt: {sorted(profiles)})"
        )
    rueckfall = fallback_class or default_class

    resolved: dict[str, SymbolCosts] = {}
    unmapped: list[str] = []
    for sym in symbols:
        key = sym.upper()
        if key in overrides:
            resolved[sym] = overrides[key]
        elif key in symbol_map:
            resolved[sym] = profiles[symbol_map[key]]
        else:
            resolved[sym] = profiles[rueckfall]
            unmapped.append(sym)

    # Nur der *ungewollte* Rückfall ist eine Warnung wert. Ein deklariertes
    # `cost_class` ist eine Angabe, kein Versehen — sonst stünde bei jedem
    # Regel-Universum eine 400-Symbol-Warnung im Log, und die nächste echte
    # ginge darin unter.
    if unmapped and fallback_class is None:
        log.warning(
            "costs: %s nicht in config/costs.yaml klassifiziert — default_class '%s' angenommen",
            unmapped,
            default_class,
        )
    return resolved


__all__ = [
    "DEFAULT_COSTS_PATH",
    "UnpriceableError",
    "class_for_liquidity",
    "resolve_symbol_costs",
]
