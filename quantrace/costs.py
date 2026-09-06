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


#: Mindestpreisschritte am US-Aktienmarkt — SEC Rule 612 („Sub-Penny Rule"):
#: ein Cent ab 1,00 $, ein Hundertstelcent darunter. **Eine Marktregel, keine
#: Annahme** — anders als jede bps-Zahl in `costs.yaml`.
TICK_AB_EIN_DOLLAR = 0.01
TICK_UNTER_EIN_DOLLAR = 0.0001


def spread_untergrenze_bps(preis):
    """Der kleinstmögliche Halbspread an diesem Kursniveau, in bps.

    **Warum das gebraucht wird** (#323). Die Kostenklassen in `costs.yaml`
    stehen in bps und gelten pro Symbol für den ganzen Backtest. Das ist
    richtig, solange ein Papier auf normalem Kursniveau handelt. Fällt es auf
    zwei Zehntelcent, stimmt die Zahl nicht mehr um Grössenordnungen: eine
    Bewegung von 0,0002 auf 0,0003 ist **+50 %** und dabei genau ein Tick.

    Gemessen am 2026-09-05 über `us_top500_liquid`: 0,5–0,7 % der Kurszellen
    tragen 99,7 % der Rendite, und es sind durchweg insolvente Papiere kurz
    vor dem Delisting. `buy_and_hold` kam über 2007–2012 auf **CAGR +2.333 %
    bei 99 % Drawdown** — beides zusammen gibt es nicht, also hat keine der
    beiden Zahlen gemessen, was sie behauptet.

    Die Untergrenze trifft die Ursache statt des Symptoms und braucht keinen
    gesetzten Grenzwert: der Tick ist vorgeschrieben, und wer über den Spread
    handelt, zahlt mindestens einen halben davon.

        Kurs      Tick       Halbspread   in bps
        100,00 $  0,01 $     0,005 $         0,5
          5,00 $  0,01 $     0,005 $        10
          0,02 $  0,0001 $   0,00005 $      25
          0,001 $ 0,0001 $   0,00005 $     500
          0,0002 $ 0,0001 $  0,00005 $   2.500

    **Sie ist eine Untergrenze und bleibt optimistisch.** Reale Spreads auf
    solchen Papieren liegen weit darüber, und Market Impact kommt obendrauf.
    Was sie belastbar macht, ist die Richtung: sie kann Kosten nur erhöhen,
    nie senken — dieselbe Konvention wie bei `dollar_volume`.
    """
    import numpy as np

    p = np.asarray(preis, dtype=float)
    tick = np.where(p < 1.0, TICK_UNTER_EIN_DOLLAR, TICK_AB_EIN_DOLLAR)
    # Nicht-positive Kurse gibt es im Lesepfad nicht mehr (#322/#324); käme
    # doch einer durch, wäre eine unendliche Kostenzahl schlimmer als keine.
    with np.errstate(divide="ignore", invalid="ignore"):
        bps = (tick / 2.0) / p * 10_000.0
    return np.where(np.isfinite(bps) & (p > 0), bps, 0.0)


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
    "TICK_AB_EIN_DOLLAR",
    "TICK_UNTER_EIN_DOLLAR",
    "UnpriceableError",
    "class_for_liquidity",
    "resolve_symbol_costs",
    "spread_untergrenze_bps",
]
