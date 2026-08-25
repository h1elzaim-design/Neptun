"""Geht das überhaupt? — Machbarkeit gegen Engine und Datenstand, gerechnet.

Die Frage, die vor jedem Spec steht und die bisher niemand gestellt hat. Ein
Research-Agent, der eine Strategie auf Intraday-Bars vorschlägt, ist nicht
kreativ, sondern falsch informiert: EODHD-Intraday gibt im aktuellen Abo 403,
und der Lake reicht heute bis ~2001. Wer das erst im Backtest merkt, hat die
Idee schon ausformuliert, das Spec geschrieben und den Lauf gestartet.

**Warum das Code ist und kein Prompt.** Ein LLM, das über Machbarkeit
*urteilt*, rät — es kennt den Ladestand nicht und liest keinen Knotenkatalog.
Die Fakten hier kommen deshalb aus den Quellen selbst: der Knotenkatalog aus
``quantrace.graph.nodes``, das Kursfenster aus dem Schicht-2-Manifest, die
Universen aus ihren YAMLs. Das LLM bekommt sie als Kontext und **formuliert**
daraus eine Alternative; entscheiden tut diese Datei.

Das ist dieselbe Arbeitsteilung wie in ADR-011: das Gate ist Code, nicht
Prompt.

**Was »machbar« hier heißt und was nicht.** Machbar = *rechenbar*: die
Bausteine existieren, die Daten liegen, das Fenster trägt. Über die *Güte*
einer Strategie sagt dieses Modul nichts — dafür gibt es den Backtest und den
Evaluation-Agenten. Ein Urteil »machbar« ist keine Empfehlung.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIVERSES_DIR = REPO_ROOT / "data" / "universes"


@dataclass(frozen=True)
class DataKind:
    """Eine Datenklasse, die eine Strategie brauchen kann.

    ``available`` ist die einzige Angabe, die zählt — ``reason`` erklärt sie
    und ist bei ``False`` **Pflicht**: »nicht verfügbar« ohne Grund ist eine
    Sackgasse, mit Grund ist es der Anfang einer Alternative.
    """

    key: str
    label: str
    available: bool
    reason: str = ""
    #: Ab wann tragbar. ``None`` heißt »unbekannt oder nicht datumsgebunden«.
    since: date | None = None
    until: date | None = None
    issue: str = ""


def _lake_window() -> tuple[date | None, date | None, int]:
    """Erster und letzter Tag mit Kursen im Lake, plus die Zahl der Instrumente.

    Fällt auf ``(None, None, 0)`` zurück, wenn Schicht 2 nicht gebaut oder der
    Lake nicht erreichbar ist. Das ist kein Fehler, sondern eine Antwort: dann
    ist *nichts* rechenbar, und genau das soll das Urteil sagen.
    """
    try:
        from quantrace import resolve

        manifest = resolve.read_manifest()
        if manifest.empty:
            return None, None, 0
        return (
            min(manifest["first"]),
            max(manifest["last"]),
            int(len(manifest)),
        )
    except Exception:  # noqa: BLE001 — ein unerreichbarer Lake ist »nichts da«
        return None, None, 0


def data_kinds() -> list[DataKind]:
    """Was die Plattform an Daten hergibt — heute, nicht im Prinzip.

    Die Liste ist bewusst kurz und vollständig statt lang und ungefähr. Jede
    Zeile hier ist eine Zusage, gegen die ein Spec geprüft wird; eine Klasse,
    die fehlt, führt zu »nicht machbar« statt zu einem stillen NaN.
    """
    first, last, n = _lake_window()

    import os

    return [
        DataKind(
            key="daily_prices",
            label="Tages-OHLCV, survivorship-frei",
            available=bool(first and last),
            reason=(
                f"{n:,} Instrumente aus Schicht 2, {first} … {last}".replace(",", ".")
                if first
                else "Schicht 2 nicht gebaut — scripts/build_resolved.py --manifest-only"
            ),
            since=first,
            until=last,
            issue="#245",
        ),
        DataKind(
            key="corporate_actions",
            label="Splits + Dividenden (Total Return)",
            available=bool(first),
            reason=(
                "Geladen, aber hinter den Kursen — außerhalb des Actions-Fensters "
                "gibt der Lesepfad rohe Reihen zurück (`Adjustment.status`). "
                "Kein stiller Fehler, aber kein Total Return."
            ),
            issue="#245",
        ),
        DataKind(
            key="fundamentals",
            label="Fundamentaldaten point-in-time (SEC EDGAR)",
            available=True,
            reason=(
                "30 Kennzahlen, nach `filed` gefiltert, Restatements aufgelöst. "
                "Nur US-Filer, in XBRL verlässlich ab ~2009. Braucht SEC_USER_AGENT."
            ),
            since=date(2009, 1, 1),
            issue="#257",
        ),
        DataKind(
            key="macro",
            label="Makro (FRED): Zinsstruktur, Credit-Spread",
            available=bool(os.environ.get("FRED_API_KEY")),
            reason=(
                "Als Regime-Feature erreichbar (`use_macro`), Default aus."
                if os.environ.get("FRED_API_KEY")
                else "FRED_API_KEY nicht gesetzt."
            ),
            issue="#231",
        ),
        DataKind(
            key="intraday",
            label="Intraday-Bars",
            available=False,
            reason="EODHD-Intraday gibt im aktuellen Abo HTTP 403. Keine zweite Quelle.",
        ),
        DataKind(
            key="pit_index_membership",
            label="Punkt-in-Zeit-Indexzugehörigkeit",
            available=False,
            reason=(
                "Es gibt keine Quelle für »wer war am 12.03.2007 im S&P 500«. "
                "Der Regel-Screen (ADR-015) ersetzt die *Aufzählung*, nicht den "
                "Index — »die 500 liquidesten Titel« ist nicht »der S&P 500«."
            ),
            issue="#254",
        ),
        DataKind(
            key="crypto",
            label="Crypto-Kurse",
            available=False,
            reason=(
                "Am 2026-08-13 mit Tiingo entfernt; EODHDs Bulk führt keine "
                "Crypto-Paare. Der `crypto_24_7`-Kalender existiert weiter."
            ),
        ),
        DataKind(
            key="news",
            label="News / Sentiment",
            available=os.environ.get("NEWS_PROVIDER", "off").strip().lower() != "off",
            reason=(
                f"NEWS_PROVIDER={os.environ.get('NEWS_PROVIDER', 'off')}. "
                "Kein Backtest-Feature — nicht historisch vollständig."
            ),
            issue="#181",
        ),
        DataKind(
            key="short_interest",
            label="Short Interest, Insider-Trades, Optionsdaten",
            available=False,
            reason="Nicht angebunden. Keine Quelle im aktuellen Stack.",
        ),
    ]


@dataclass
class Verdict:
    """Das Urteil. ``feasible`` ist die Antwort, der Rest ist die Begründung.

    ``blockers`` und ``caveats`` sind getrennt, weil sie Verschiedenes heißen:
    ein Blocker macht die Strategie **unrechenbar**, ein Caveat macht sie
    *eingeschränkt* rechenbar. Beides in eine Liste zu werfen hieße, dem
    Nutzer die Unterscheidung zu überlassen, die dieses Modul treffen soll.
    """

    feasible: bool
    blockers: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    missing_nodes: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "blockers": self.blockers,
            "caveats": self.caveats,
            "missing_nodes": self.missing_nodes,
            "missing_data": self.missing_data,
        }


def assess(
    *,
    needs_data: list[str] | None = None,
    needs_nodes: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    universe: str | None = None,
) -> Verdict:
    """Ist das mit dem heutigen Stand rechenbar?

    ``needs_data`` sind Schlüssel aus `data_kinds`, ``needs_nodes`` Typen aus
    dem Graph-Katalog. Beide kommen vom Aufrufer (im Agentenpfad: aus dem
    Spec extrahiert) — was daraus folgt, rechnet diese Funktion.

    **Ein unbekannter Schlüssel ist ein Blocker, kein ignoriertes Feld.** Wer
    ``needs_data=["optionsflow"]`` übergibt, meint etwas, das es nicht gibt;
    das stillschweigend durchzulassen wäre die teuerste Antwort — »machbar«
    für eine Strategie, deren halbe Idee fehlt.
    """
    from quantrace.graph.nodes import CATALOG

    v = Verdict(feasible=True)
    bekannt = {d.key: d for d in data_kinds()}

    for key in needs_data or []:
        d = bekannt.get(key)
        if d is None:
            v.missing_data.append(key)
            v.blockers.append(
                f"Datenklasse `{key}` ist der Plattform unbekannt. "
                f"Bekannt sind: {', '.join(sorted(bekannt))}."
            )
        elif not d.available:
            v.missing_data.append(key)
            v.blockers.append(
                f"{d.label}: {d.reason}" + (f" ({d.issue})" if d.issue else "")
            )
        elif d.reason:
            v.caveats.append(f"{d.label}: {d.reason}")

    for typ in needs_nodes or []:
        if typ not in CATALOG:
            v.missing_nodes.append(typ)
            v.blockers.append(
                f"Graph-Knoten `{typ}` gibt es nicht. Der Katalog hat "
                f"{len(CATALOG)} Typen — `catalog_payload()` listet sie."
            )

    # Das Fenster gegen den Lake. Diese Prüfung ist der Grund, warum das Modul
    # heute überhaupt etwas ablehnt: die Bausteine sind fast immer da, die
    # Jahre nicht.
    lake_first, lake_last, _ = _lake_window()
    if start or end:
        if lake_first is None:
            v.blockers.append(
                "Kein Kursfenster im Lake — Schicht 2 ist nicht gebaut. "
                "→ scripts/build_resolved.py --manifest-only"
            )
        else:
            if start and start < lake_first:
                v.blockers.append(
                    f"Fenster beginnt {start}, der Lake trägt erst ab {lake_first}. "
                    f"Fehlend: {(lake_first - start).days} Kalendertage. Der Ladelauf "
                    "läuft (#265)."
                )
            if end and end > lake_last:
                v.blockers.append(
                    f"Fenster endet {end}, der Lake reicht bis {lake_last}."
                )

    if universe:
        pfad = UNIVERSES_DIR / f"{universe}.yaml"
        if not pfad.exists():
            v.blockers.append(
                f"Universum `{universe}` gibt es nicht. Vorhanden: "
                f"{', '.join(sorted(p.stem for p in UNIVERSES_DIR.glob('*.yaml')))}."
            )
        else:
            import yaml

            meta = yaml.safe_load(pfad.read_text(encoding="utf-8")) or {}
            fenster = meta.get("usable_window") or {}
            uni_start = fenster.get("start")
            if start and uni_start and str(start) < str(uni_start):
                # Caveat, kein Blocker: das gemessene Fenster stammt aus der
                # Tiingo-Zeit und beschreibt Verfügbarkeit beim alten Anbieter,
                # nicht Lake-Inhalt (#265). Es als harte Grenze zu nehmen wäre
                # eine Zusage über Daten, die niemand gegen den Lake geprüft hat.
                v.caveats.append(
                    f"`{universe}` nennt ein nutzbares Fenster ab {uni_start}; "
                    f"angefragt ist {start}. Der Wert wurde gegen Tiingos "
                    "Survivor-Liste gemessen und ist seit dem Anbieterwechsel "
                    "unbestätigt (#265)."
                )

    v.feasible = not v.blockers
    return v


def capabilities() -> dict[str, Any]:
    """Der volle Fähigkeitsbericht — Kontext für das LLM, Anzeige für die UI.

    Das ist bewusst *eine* Funktion und nicht drei Endpunkte: das LLM soll den
    Stand in einem Stück bekommen. Wer ihn in Häppchen holt, baut ein Modell,
    das die Hälfte kennt und über die andere rät.
    """
    from quantrace.graph.nodes import CATALOG

    familien: dict[str, list[str]] = {}
    for typ in CATALOG:
        familien.setdefault(typ.split(".")[0], []).append(typ)

    first, last, n = _lake_window()
    return {
        "nodes": {k: sorted(v) for k, v in sorted(familien.items())},
        "n_nodes": len(CATALOG),
        "data": [
            {
                "key": d.key,
                "label": d.label,
                "available": d.available,
                "reason": d.reason,
                "since": d.since.isoformat() if d.since else None,
                "until": d.until.isoformat() if d.until else None,
                "issue": d.issue,
            }
            for d in data_kinds()
        ],
        "lake": {
            "first": first.isoformat() if first else None,
            "last": last.isoformat() if last else None,
            "n_instruments": n,
        },
        "universes": sorted(p.stem for p in UNIVERSES_DIR.glob("*.yaml")),
    }


__all__ = ["DataKind", "Verdict", "assess", "capabilities", "data_kinds"]
