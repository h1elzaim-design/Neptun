"""Knowledge Agent — übersetzt Backtest- und Evaluation-Ergebnisse in Obsidian-Notes.

Trennt strikt: Maschine schreibt strukturiert (Frontmatter), Mensch liest und entscheidet.
"""

from __future__ import annotations

from datetime import UTC, datetime

from quantrace.models import (
    BacktestResult,
    EvaluationReport,
    KnowledgeNote,
    StrategySpec,
)
from quantrace.obsidian_client import ObsidianClient


def strategy_note(spec: StrategySpec) -> KnowledgeNote:
    body = (
        f"## Idee\n{spec.description or '(keine Beschreibung)'}\n\n"
        f"## Klasse\n`{spec.strategy_class}`\n\n"
        f"## Universum\n`{spec.universe}` @ `{spec.timeframe.value}`\n\n"
        f"## Parameter\n```json\n{spec.params}\n```\n\n"
        f"## Risiken\n" + "\n".join(f"- {r}" for r in spec.risks or ["(keine notiert)"]) + "\n\n"
        "## Verlinkte Backtests\n_Wird automatisch ergänzt._\n"
    )
    return KnowledgeNote(
        folder="02 Strategien",
        title=spec.name,
        frontmatter={
            "strategy_id": spec.strategy_id,
            "class": spec.strategy_class,
            "universe": spec.universe,
            "timeframe": spec.timeframe.value,
            "status": spec.status.value,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        tags=["strategy", spec.strategy_class, spec.status.value],
        body=body,
    )


def backtest_note(spec: StrategySpec, bt: BacktestResult) -> KnowledgeNote:
    title = f"{spec.name} — Backtest {bt.start}..{bt.end}"
    body = (
        f"## Kernmetriken\n"
        f"| Metrik | Wert |\n|---|---|\n"
        f"| CAGR | {bt.cagr:.2%} |\n"
        f"| Sharpe | {bt.sharpe:.2f} |\n"
        f"| Sortino | {bt.sortino:.2f} |\n"
        f"| Calmar | {bt.calmar:.2f} |\n"
        f"| Max Drawdown | {bt.max_drawdown:.2%} |\n"
        f"| Ulcer Index | {bt.ulcer_index:.4f} |\n"
        f"| Trades | {bt.trades.n_trades} |\n"
        f"| Win Rate | {bt.trades.win_rate:.2%} |\n"
        f"| Profit Factor | {bt.trades.profit_factor:.2f} |\n\n"
        f"## Konfiguration\n"
        f"- Cash: {bt.config.cash}\n"
        f"- Fees: {bt.config.fees_bps} bps\n"
        f"- Slippage: {bt.config.slippage_bps} bps\n"
        f"- Size: {bt.config.size} ({bt.config.size_type})\n\n"
        f"## Diagnose\n_(Hier kommen Charts und Anhänge.)_\n"
    )
    return KnowledgeNote(
        folder="03 Backtests",
        title=title,
        frontmatter={
            "strategy_id": spec.strategy_id,
            "data_hash": bt.data_hash,
            "start": str(bt.start),
            "end": str(bt.end),
            "sharpe": round(bt.sharpe, 3),
            "cagr": round(bt.cagr, 4),
            "max_drawdown": round(bt.max_drawdown, 4),
            "n_trades": bt.trades.n_trades,
        },
        tags=["backtest", spec.strategy_class],
        body=body,
    )


def evaluation_note(spec: StrategySpec, report: EvaluationReport) -> KnowledgeNote:
    verdict = "APPROVED" if report.passed_guardrails else "REJECTED"
    body = (
        f"## Verdikt: **{verdict}**\n\n"
        f"## Score-Komponenten\n"
        f"| Dimension | Score |\n|---|---|\n"
        f"| Performance | {report.score_performance:.2f} |\n"
        f"| Risk | {report.score_risk:.2f} |\n"
        f"| Stability | {report.score_stability:.2f} |\n"
        f"| Realism | {report.score_realism:.2f} |\n"
        f"| Generalization | {report.score_generalization:.2f} |\n"
        f"| Simplicity | {report.score_simplicity:.2f} |\n"
        f"| **Total** | **{report.score_total:.2f}** |\n\n"
        f"## Ablehnungsgründe\n"
        + ("\n".join(f"- {r}" for r in report.rejection_reasons) or "_keine_")
        + "\n\n## Nächster Versuch\n"
        + (report.next_variation_hint or "_keiner vorgeschlagen_")
    )
    folder = "05 Approved Candidates" if report.passed_guardrails else "06 Rejected Ideas"
    return KnowledgeNote(
        folder=folder,
        title=f"{spec.name} — Evaluation",
        frontmatter={
            "strategy_id": spec.strategy_id,
            "score_total": round(report.score_total, 3),
            "passed": report.passed_guardrails,
            "created": report.created_at.isoformat(timespec="seconds"),
        },
        tags=["evaluation", verdict.lower()],
        body=body,
    )


def publish_full(
    spec: StrategySpec,
    bts: list[BacktestResult],
    report: EvaluationReport,
    client: ObsidianClient,
) -> list[str]:
    paths = [client.publish(strategy_note(spec))]
    for bt in bts:
        paths.append(client.publish(backtest_note(spec, bt)))
    paths.append(client.publish(evaluation_note(spec, report)))
    return paths
