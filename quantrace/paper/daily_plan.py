"""Daily-Rebalance-Plan + Reconciliation (Rechen-Kern).

Der tägliche Zyklus nach US-Close: Registry-Zielgewichte gegen die echten
Broker-Positionen halten, Drift messen, den Rebalance-Plan rechnen und alles
als Monitoring-Note dokumentieren. **Es wird nie submittet** — dieses Modul
importiert den Executor nicht einmal; Ausführung bleibt hinter dem gated
`POST /api/paper/execute` (Role + confirm + live-Switches).

Aufteilung:
- :func:`build_daily_plan` — pure Funktion: (Registry, Positionen, Preise,
  NAV, Limits) → Plan + Drift-Reconciliation + Equity-Snapshot.
- :func:`render_plan_note` — KnowledgeNote für `11 Live Monitoring/`,
  idempotent pro Tag (Auto-generated-Sektion, VAULT_CONVENTIONS §6.4).

Das IO (Broker-Read, Lake-Preise, git push) lebt im ACA-Job
`worker/daily_plan_job.py` — dieser Kern bleibt ohne Netz testbar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from quantrace.brokers.base import Position
from quantrace.models import KnowledgeNote
from quantrace.paper.rebalance import RebalancePlan, RiskLimits, plan_rebalance
from quantrace.paper.registry import WEIGHTING_BASIS_NEUTRAL, PortfolioRegistry

#: Ab wann ein Kursstand als veraltet gilt (Kalendertage).
#:
#: Großzügig, weil ein langes Wochenende plus Feiertag schon vier Tage frisst
#: und Tiingos EOD am Abend des Handelstags noch nachhängen kann. Ein zu enger
#: Wert erzeugt täglich falschen Alarm — und ein Alert, den man wegklickt, ist
#: schlimmer als keiner.
MAX_PRICE_AGE_DAYS = 5


#: Ab dieser absoluten Gewichts-Abweichung (Ist − Soll) pro Symbol wird ein
#: Drift-Alert in die Note geschrieben. 5 Prozentpunkte = deutlich mehr als
#: tägliches Markt-Rauschen bei ETF-Sleeves.
DRIFT_ALERT_THRESHOLD = 0.05


@dataclass(frozen=True, slots=True)
class DriftRow:
    """Ist-vs-Soll eines Symbols — die Reconciliation-Zeile."""

    symbol: str
    target_weight: float
    current_weight: float

    @property
    def drift(self) -> float:
        return self.current_weight - self.target_weight


@dataclass(slots=True)
class DailyPlanReport:
    """Alles, was die Monitoring-Note eines Tages braucht."""

    as_of: date
    account_value: float
    weighting: str
    weighting_basis: str
    n_candidates: int
    plan: RebalancePlan
    drift: list[DriftRow] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    registry_warnings: list[str] = field(default_factory=list)
    #: Jüngster Schlusskurs, den der Plan gesehen hat, und sein Alter in Tagen.
    #: Ohne das steht in der Note eine Zahl ohne Datum — und ein Plan auf drei
    #: Wochen alten Kursen sieht genauso aus wie einer auf heutigen (#192).
    price_as_of: date | None = None
    price_age_days: int | None = None

    @property
    def max_abs_drift(self) -> float:
        return max((abs(r.drift) for r in self.drift), default=0.0)


def build_daily_plan(
    registry: PortfolioRegistry,
    positions: list[Position],
    prices: dict[str, float],
    account_value: float,
    limits: RiskLimits,
    *,
    as_of: date,
    price_dates: dict[str, date] | None = None,
    max_price_age_days: int = MAX_PRICE_AGE_DAYS,
    drawdown: float = 0.0,
    drift_alert_threshold: float = DRIFT_ALERT_THRESHOLD,
) -> DailyPlanReport:
    """Rechnet Plan + Drift-Reconciliation. Submittet nichts.

    Drift = aktuelles Gewicht (Position × Preis / NAV) minus Registry-Ziel.
    Symbole ohne Preis fließen nicht in die Ist-Gewichte ein — der Planner
    überspringt sie ohnehin mit Warnung, hier wird zusätzlich ein Alert
    geschrieben, damit die Lücke in der Note sichtbar ist.
    """
    plan = plan_rebalance(
        target_weights=registry.target_weights,
        positions=positions,
        prices=prices,
        account_value=account_value,
        limits=limits,
        drawdown=drawdown,
    )

    current: dict[str, float] = {}
    if account_value > 0:
        for p in positions:
            price = prices.get(p.symbol)
            if price is not None and price > 0:
                current[p.symbol] = p.quantity * price / account_value

    symbols = sorted(set(registry.target_weights) | set(current))
    drift_rows = [
        DriftRow(
            symbol=s,
            target_weight=float(registry.target_weights.get(s, 0.0)),
            current_weight=float(current.get(s, 0.0)),
        )
        for s in symbols
    ]

    alerts: list[str] = []
    if registry.weighting_basis == WEIGHTING_BASIS_NEUTRAL:
        alerts.append(
            "Zielgewichte sind die NEUTRALE Sleeve-Gleichgewichtung (kein "
            "Strategie-Signal) — Plan dient Monitoring/Dry-Run, nicht als "
            "Handelsempfehlung."
        )
    if plan.blocked:
        alerts.append("Rebalance BLOCKIERT: " + "; ".join(plan.warnings))
    for row in drift_rows:
        if abs(row.drift) >= drift_alert_threshold:
            alerts.append(
                f"Drift-Alert {row.symbol}: Ist {row.current_weight:.1%} vs "
                f"Soll {row.target_weight:.1%} (Δ {row.drift:+.1%})."
            )
    priced = set(prices)
    for p in positions:
        if p.symbol not in priced:
            alerts.append(
                f"Kein Preis für gehaltene Position {p.symbol} — Ist-Gewicht "
                "unvollständig, Symbol im Plan übersprungen."
            )

    # Wie alt sind die Kurse, auf denen hier geplant wird? Ein Plan auf
    # veralteten Daten ist gefährlicher als gar keiner, weil er genauso
    # aussieht wie ein frischer.
    price_as_of = max(price_dates.values()) if price_dates else None
    price_age_days = (as_of - price_as_of).days if price_as_of else None
    if price_as_of is not None and price_age_days is not None:
        if price_age_days > max_price_age_days:
            alerts.append(
                f"VERALTETE KURSE: jüngster Close ist {price_as_of.isoformat()} "
                f"({price_age_days} Tage alt, Grenze {max_price_age_days}). "
                "Der Plan beruht auf altem Stand — vor einer Ausführung neu fetchen."
            )
        # Einzelne Nachzügler: ein Symbol, das nicht mehr aktualisiert wird,
        # verzerrt die Ist-Gewichte, ohne dass die Gesamtlage alt aussieht.
        for sym, d in sorted(price_dates.items()):
            lag = (price_as_of - d).days
            if lag > max_price_age_days:
                alerts.append(
                    f"Nachzügler {sym}: letzter Close {d.isoformat()} — {lag} Tage "
                    "hinter dem Rest. Ist-Gewicht für dieses Symbol unzuverlässig."
                )

    return DailyPlanReport(
        as_of=as_of,
        account_value=float(account_value),
        weighting=registry.weighting,
        weighting_basis=registry.weighting_basis,
        n_candidates=registry.n_deployable,
        plan=plan,
        drift=drift_rows,
        alerts=alerts,
        registry_warnings=list(registry.warnings),
        price_as_of=price_as_of,
        price_age_days=price_age_days,
    )


# -----------------------------------------------------------------------------
# Vault-Note (11 Live Monitoring/) — idempotent pro Tag
# -----------------------------------------------------------------------------

def plan_note_title(as_of: date) -> str:
    return f"{as_of.isoformat()}_daily_plan"


def render_plan_note(report: DailyPlanReport) -> KnowledgeNote:
    """Monitoring-Note nach VAULT_CONVENTIONS: Frontmatter + Auto-generated
    (nur H3-Subsections), manuelle Sektion bleibt beim Re-Save erhalten."""
    fm = {
        "type": "daily_plan",
        "date": report.as_of,
        "account_value": round(report.account_value, 2),
        "n_candidates": report.n_candidates,
        "weighting": report.weighting,
        "weighting_basis": report.weighting_basis,
        "n_orders": report.plan.n_orders,
        "turnover": round(report.plan.turnover, 6),
        "gross_after": round(report.plan.gross_after, 6),
        "blocked": report.plan.blocked,
        "max_abs_drift": round(report.max_abs_drift, 6),
        "n_alerts": len(report.alerts),
        "price_as_of": report.price_as_of,
        "price_age_days": report.price_age_days,
        "executed": False,  # wird nur durch den gated Execute-Flow wahr
        "status": "open",
    }

    lines: list[str] = [
        f"# 📅 Daily Rebalance Plan — {report.as_of.isoformat()}",
        "",
        "## Auto-generated",
        "",
        "### Snapshot",
        "",
        f"- **NAV (Paper):** ${report.account_value:,.2f}",
        f"- **Kandidaten im Sleeve:** {report.n_candidates} ({report.weighting}, {report.weighting_basis})",
        f"- **Geplante Orders:** {report.plan.n_orders} · Turnover {report.plan.turnover:.2%} · Gross danach {report.plan.gross_after:.2f}",
        f"- **Blocked:** {'JA — kein Order-Vorschlag' if report.plan.blocked else 'nein'}",
    ]
    if report.price_as_of is not None:
        lines.append(
            f"- **Kursstand:** {report.price_as_of.isoformat()} "
            f"({report.price_age_days} Tage alt)"
        )
    lines.append("")

    if report.alerts:
        lines += ["### ⚠️ Alerts", ""]
        lines += [f"- {a}" for a in report.alerts]
        lines.append("")

    lines += [
        "### Reconciliation (Ist vs. Soll)",
        "",
        "| Symbol | Soll | Ist | Drift |",
        "|---|---:|---:|---:|",
    ]
    for row in report.drift:
        lines.append(
            f"| {row.symbol} | {row.target_weight:.2%} | {row.current_weight:.2%} "
            f"| {row.drift:+.2%} |"
        )
    lines.append("")

    lines += ["### Geplante Orders (NICHT submittet)", ""]
    if report.plan.orders:
        lines += [
            "| Symbol | Seite | Stück | Kurs | Notional |",
            "|---|---|---:|---:|---:|",
        ]
        for o in report.plan.orders:
            lines.append(
                f"| {o.symbol} | {o.side.value} | {o.quantity:,.4f} "
                f"| ${o.price:,.2f} | ${o.notional:,.2f} |"
            )
    else:
        lines.append("_Keine Orders — Portfolio innerhalb der Toleranzen oder blockiert._")
    lines.append("")

    if report.plan.warnings:
        lines += ["### Planner-Warnungen", ""]
        lines += [f"- {w}" for w in report.plan.warnings]
        lines.append("")

    lines += [
        "### Governance",
        "",
        "- Dieser Plan wurde **nicht** submittet und wird es automatisch nie —",
        "  Ausführung nur über `POST /api/paper/execute` (Role + `confirm=true`).",
        "",
        "## 📝 Beobachtungen",
        "",
        "_(Manuell ergänzen: ausgeführt? abgewichen? warum?)_",
    ]

    return KnowledgeNote(
        folder="11 Live Monitoring",
        title=plan_note_title(report.as_of),
        frontmatter=fm,
        body="\n".join(lines) + "\n",
    )


__all__ = [
    "DRIFT_ALERT_THRESHOLD",
    "DailyPlanReport",
    "DriftRow",
    "build_daily_plan",
    "plan_note_title",
    "render_plan_note",
]
