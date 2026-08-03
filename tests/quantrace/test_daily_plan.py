"""Daily-Rebalance-Plan (PLAN P0 #6/#7): Drift-Reconciliation, Note, Governance.

Der Rechen-Kern ist pur — hier mit handkonstruierten Positionen/Preisen
gegen bekannte Zahlen geprüft. Der Governance-Test stellt strukturell sicher,
dass der Job-Code keinen Submit-Pfad importiert.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from quantrace.brokers.base import Position
from quantrace.paper.daily_plan import (
    build_daily_plan,
    plan_note_title,
    render_plan_note,
)
from quantrace.paper.rebalance import RiskLimits
from quantrace.paper.registry import WEIGHTING_BASIS_NEUTRAL, PortfolioRegistry

AS_OF = date(2026, 7, 3)


def _registry(weights: dict[str, float]) -> PortfolioRegistry:
    return PortfolioRegistry(
        candidates=[],
        target_weights=weights,
        weighting="equal",
        warnings=["synthetic"],
    )


def _pos(symbol: str, qty: float) -> Position:
    return Position(symbol=symbol, quantity=qty, avg_cost=0.0, market_value=0.0)


def _limits() -> RiskLimits:
    return RiskLimits(max_notional_pct=1.0, max_notional_abs=1e9)


# --- Drift-Reconciliation -----------------------------------------------------------


def test_drift_hand_computed():
    # NAV 100k; SPY: 100 Stück à 500 = 50% Ist vs 60% Soll → Drift −10 pp.
    # QQQ: 0 Ist vs 40% Soll → Drift −40 pp. GLD gehalten, aber kein Ziel → +10 pp.
    registry = _registry({"SPY": 0.6, "QQQ": 0.4})
    positions = [_pos("SPY", 100), _pos("GLD", 50)]
    prices = {"SPY": 500.0, "QQQ": 400.0, "GLD": 200.0}

    report = build_daily_plan(registry, positions, prices, 100_000.0, _limits(), as_of=AS_OF)

    by_sym = {r.symbol: r for r in report.drift}
    assert by_sym["SPY"].current_weight == pytest.approx(0.5)
    assert by_sym["SPY"].drift == pytest.approx(-0.1)
    assert by_sym["QQQ"].drift == pytest.approx(-0.4)
    assert by_sym["GLD"].current_weight == pytest.approx(0.1)
    assert by_sym["GLD"].drift == pytest.approx(0.1)
    assert report.max_abs_drift == pytest.approx(0.4)


def test_drift_alerts_fire_above_threshold_only():
    registry = _registry({"SPY": 0.5, "QQQ": 0.5})
    # SPY fast auf Ziel (Drift 2 pp), QQQ weit weg (Drift −50 pp → Alert).
    positions = [_pos("SPY", 104)]
    prices = {"SPY": 500.0, "QQQ": 400.0}

    report = build_daily_plan(registry, positions, prices, 100_000.0, _limits(), as_of=AS_OF)

    drift_alerts = [a for a in report.alerts if a.startswith("Drift-Alert")]
    assert len(drift_alerts) == 1
    assert "QQQ" in drift_alerts[0]


def test_neutral_placeholder_always_flagged():
    report = build_daily_plan(
        _registry({"SPY": 1.0}), [], {"SPY": 500.0}, 100_000.0, _limits(), as_of=AS_OF
    )
    assert report.weighting_basis == WEIGHTING_BASIS_NEUTRAL
    assert any("NEUTRALE" in a for a in report.alerts)


def test_kill_switch_blocks_and_alerts():
    report = build_daily_plan(
        _registry({"SPY": 1.0}),
        [],
        {"SPY": 500.0},
        100_000.0,
        RiskLimits(max_drawdown_kill_switch=-0.10),
        as_of=AS_OF,
        drawdown=-0.15,
    )
    assert report.plan.blocked
    assert report.plan.n_orders == 0
    assert any(a.startswith("Rebalance BLOCKIERT") for a in report.alerts)


def test_held_position_without_price_raises_alert():
    report = build_daily_plan(
        _registry({"SPY": 1.0}),
        [_pos("GLD", 10)],
        {"SPY": 500.0},  # kein GLD-Preis
        100_000.0,
        _limits(),
        as_of=AS_OF,
    )
    assert any("Kein Preis" in a and "GLD" in a for a in report.alerts)


# --- Note-Rendering ------------------------------------------------------------------


def test_note_schema_and_structure():
    registry = _registry({"SPY": 0.6, "QQQ": 0.4})
    report = build_daily_plan(
        registry, [_pos("SPY", 100)], {"SPY": 500.0, "QQQ": 400.0},
        100_000.0, _limits(), as_of=AS_OF,
    )
    note = render_plan_note(report)

    assert note.folder == "11 Live Monitoring"
    assert note.title == plan_note_title(AS_OF) == "2026-07-03_daily_plan"
    fm = note.frontmatter
    assert fm["type"] == "daily_plan"
    assert fm["executed"] is False
    assert fm["weighting_basis"] == WEIGHTING_BASIS_NEUTRAL
    assert fm["n_orders"] == report.plan.n_orders
    assert fm["max_abs_drift"] == pytest.approx(report.max_abs_drift, abs=1e-6)

    # §6.4: genau ein H2 "## Auto-generated", H3-Subsections, manuelle Sektion danach.
    body = note.body
    auto_idx = body.index("## Auto-generated")
    manual_idx = body.index("## 📝 Beobachtungen")
    auto_block = body[auto_idx + len("## Auto-generated"):manual_idx]
    assert "\n## " not in auto_block  # keine weiteren H2s in der Auto-Sektion
    assert "### Reconciliation (Ist vs. Soll)" in auto_block
    assert "NICHT submittet" in auto_block


# --- Governance: der Job kann strukturell nicht submitten ----------------------------


def test_daily_plan_job_has_no_submit_path():
    """Die Invariante 'forscht autonom, handelt nie autonom' strukturell:
    weder der Rechen-Kern noch der ACA-Job importieren execute_plan, rufen
    irgendein .submit()/execute_plan() auf oder setzen allow_live=True.
    Geprüft auf dem AST (Docstrings/Kommentare zählen nicht)."""
    import ast

    repo = Path(__file__).resolve().parents[2]
    for rel in ("quantrace/paper/daily_plan.py", "worker/daily_plan_job.py"):
        tree = ast.parse((repo / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names]
                assert "execute_plan" not in names, f"{rel} importiert execute_plan"
            if isinstance(node, ast.Call):
                fn = node.func
                called = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                assert called not in ("submit", "execute_plan"), (
                    f"{rel}:{node.lineno} ruft verbotenen Submit-Pfad auf: {called}"
                )
                for kw in node.keywords:
                    if kw.arg == "allow_live":
                        assert not (
                            isinstance(kw.value, ast.Constant) and kw.value.value is True
                        ), f"{rel}:{node.lineno} setzt allow_live=True"
    # Der Job konstruiert den Broker ausschließlich als Paper.
    job_src = (repo / "worker/daily_plan_job.py").read_text(encoding="utf-8")
    assert 'get_broker("alpaca", paper=True)' in job_src


def test_latest_prices_picks_last_close_per_symbol(monkeypatch):
    import pandas as pd

    from worker import daily_plan_job

    df = pd.DataFrame(
        {
            "date": ["2026-06-30", "2026-07-01", "2026-06-30"],
            "symbol": ["SPY", "SPY", "QQQ"],
            "close": [500.0, 505.0, 400.0],
        }
    )
    monkeypatch.setattr(
        "quantrace.storage.read_symbols", lambda symbols, start, end: df
    )
    prices, dates = daily_plan_job._latest_prices(["SPY", "QQQ", "GLD"], AS_OF)
    assert prices == {"SPY": 505.0, "QQQ": 400.0}  # GLD ohne Partition fehlt
    # Das Datum kommt mit, sonst ist der Preis nicht prüfbar (#192).
    assert dates == {"SPY": date(2026, 7, 1), "QQQ": date(2026, 6, 30)}


# --- Kursstand: Alter sichtbar machen (#192) ----------------------------------
#
# Der Job las bis 2026-08-03 nur aus dem Lake; nichts holte auf Zeitplan Daten
# nach. Ein Plan auf drei Wochen alten Kursen sah damit genauso aus wie einer
# auf heutigen. Diese Tests halten fest, dass das Alter jetzt im Plan steht und
# ab einer Grenze zum Alert wird.


def _plan_with_dates(price_dates, **kw):
    return build_daily_plan(
        _registry({"SPY": 0.6, "QQQ": 0.4}),
        [_pos("SPY", 100)],
        {"SPY": 500.0, "QQQ": 400.0},
        100_000.0,
        _limits(),
        as_of=AS_OF,
        price_dates=price_dates,
        **kw,
    )


def test_price_age_is_recorded():
    report = _plan_with_dates({"SPY": date(2026, 7, 2), "QQQ": date(2026, 7, 2)})
    assert report.price_as_of == date(2026, 7, 2)
    assert report.price_age_days == 1


def test_fresh_prices_raise_no_alert():
    report = _plan_with_dates({"SPY": date(2026, 7, 2), "QQQ": date(2026, 7, 2)})
    assert not any("VERALTETE KURSE" in a for a in report.alerts)


def test_stale_prices_are_flagged():
    """Drei Wochen alt — das darf nicht still durchgehen."""
    old = date(2026, 6, 12)
    report = _plan_with_dates({"SPY": old, "QQQ": old})
    assert report.price_age_days == 21
    assert any("VERALTETE KURSE" in a for a in report.alerts)


def test_a_single_lagging_symbol_is_named():
    """Ein Symbol, das nicht mehr aktualisiert wird, verzerrt die Ist-Gewichte,
    ohne dass die Gesamtlage alt aussieht — der häufigere und tückischere Fall."""
    report = _plan_with_dates({"SPY": date(2026, 7, 2), "QQQ": date(2026, 5, 1)})
    assert report.price_age_days == 1, "die Gesamtlage sieht frisch aus"
    assert any(a.startswith("Nachzügler QQQ") for a in report.alerts)


def test_threshold_is_generous_enough_for_a_long_weekend():
    """Ein Alert, den man täglich wegklickt, ist schlimmer als keiner. Freitag
    zu Dienstag nach Feiertag sind vier Tage — die dürfen nicht feuern."""
    from quantrace.paper.daily_plan import MAX_PRICE_AGE_DAYS

    assert MAX_PRICE_AGE_DAYS >= 4
    report = _plan_with_dates({"SPY": AS_OF - timedelta(days=4)})
    assert not any("VERALTETE KURSE" in a for a in report.alerts)


def test_without_price_dates_nothing_changes():
    """Alte Aufrufer (und Tests) sollen unverändert funktionieren."""
    report = _plan_with_dates(None)
    assert report.price_as_of is None
    assert report.price_age_days is None
    assert not any("VERALTETE KURSE" in a for a in report.alerts)


def test_price_age_lands_in_the_note():
    """Nur im Log wäre es wertlos — der Kursstand gehört in die Note, die man
    beim Review liest."""
    report = _plan_with_dates({"SPY": date(2026, 6, 12)})
    note = render_plan_note(report)
    assert note.frontmatter["price_as_of"] == date(2026, 6, 12)
    assert note.frontmatter["price_age_days"] == 21
    assert "Kursstand:" in note.body
