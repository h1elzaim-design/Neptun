"""QuantRace CLI — Glue zwischen Datenladen, Backtest, Evaluation und Obsidian.

Beispiele:
    quantrace fetch --universe us_core_etfs --start 2018-01-01 --end 2024-12-31
    quantrace backtest --strategy sma_crossover --universe us_core_etfs --fast 20 --slow 1
    quantrace backtest --strategy sma_crossover --universe us_core_etfs --fast 20 --slow 100
    quantrace evaluate --strategy-id sma_20_100
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from quantrace.data_agent import load_universe
from quantrace.models import BacktestConfig, StrategySpec, Timeframe, WalkForwardResult
from quantrace.sweep import SweepResult
from quantrace.sweep import sweep as run_sweep
from quantrace.walk_forward import walk_forward as run_walk_forward

app = typer.Typer(help="QuantRace — Trading Research CLI")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _load_universe_yaml(name: str) -> dict:
    path = Path(f"data/universes/{name}.yaml")
    if not path.exists():
        raise typer.BadParameter(f"Universum {name!r} nicht gefunden: {path}")
    return yaml.safe_load(path.read_text())


@app.command()
def fetch(
    universe: str = typer.Option(..., help="Name in data/universes/*.yaml"),
    start: datetime = typer.Option(..., formats=["%Y-%m-%d"]),
    end: datetime = typer.Option(..., formats=["%Y-%m-%d"]),
    provider: str = typer.Option("yfinance"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Lädt ein Universum über OpenBB und cached es."""
    cfg = _load_universe_yaml(universe)
    md = load_universe(
        universe=universe,
        symbols=cfg["symbols"],
        start=start.date(),
        end=end.date(),
        timeframe=Timeframe(cfg.get("timeframe", "1d")),
        provider=provider,
        force_refresh=force,
    )
    console.print(
        f"[green]OK[/green] {len(md.symbols)} Symbole, {len(md.frame)} Zeilen, "
        f"hash={md.content_hash}"
    )


@app.command()
def backtest(
    strategy: str = typer.Option(..., help="z.B. sma_crossover, mean_reversion"),
    universe: str = typer.Option(...),
    start: datetime = typer.Option(..., formats=["%Y-%m-%d"]),
    end: datetime = typer.Option(..., formats=["%Y-%m-%d"]),
    fast: int = typer.Option(20),
    slow: int = typer.Option(100),
    lookback: int = typer.Option(20),
    entry_z: float = typer.Option(2.0),
    exit_z: float = typer.Option(0.0),
    out: Path = typer.Option(Path("backtests/results"), help="Ergebnis-Verzeichnis"),
) -> None:
    """Führt einen Backtest aus und schreibt das Ergebnis als JSON."""
    from quantrace.backtest_runner import run_backtest

    cfg = _load_universe_yaml(universe)
    md = load_universe(
        universe=universe,
        symbols=cfg["symbols"],
        start=start.date(),
        end=end.date(),
        timeframe=Timeframe(cfg.get("timeframe", "1d")),
    )

    strategy_id, spec = _build_spec(strategy, universe, cfg, fast, slow, lookback, entry_z, exit_z)
    result = run_backtest(spec, md, BacktestConfig())

    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f"{strategy_id}__{start.date()}__{end.date()}.json"
    result_path.write_text(
        json.dumps(result.model_dump(mode="json", exclude={"equity_curve"}), indent=2, default=str)
    )
    _print_result(result)
    console.print(f"[blue]→[/blue] {result_path}")


@app.command()
def sweep(
    strategy: str = typer.Option(..., help="z.B. sma_crossover, mean_reversion"),
    universe: str = typer.Option(...),
    start: datetime = typer.Option(..., formats=["%Y-%m-%d"]),
    end: datetime = typer.Option(..., formats=["%Y-%m-%d"]),
    rank_by: str = typer.Option(
        "sharpe", help="Metrik zum Ranken: sharpe, cagr, sortino, calmar, max_drawdown"
    ),
    param_space_json: str = typer.Option(
        "",
        "--params",
        help='JSON param_space, z.B. \'{"fast": [10,20,50], "slow": [100,200]}\'',
    ),
    out: Path = typer.Option(Path("backtests/sweeps"), help="Ergebnis-Verzeichnis"),
) -> None:
    """Führt einen Parameter-Sweep aus: alle Kombinationen, sortiert nach Metrik."""
    import json as _json

    cfg = _load_universe_yaml(universe)
    md = load_universe(
        universe=universe,
        symbols=cfg["symbols"],
        start=start.date(),
        end=end.date(),
        timeframe=Timeframe(cfg.get("timeframe", "1d")),
    )

    # param_space entweder aus --params JSON oder vordefinierte Defaults
    if param_space_json:
        param_space = _json.loads(param_space_json)
    else:
        param_space = _default_param_space(strategy)

    _, spec = _build_spec(strategy, universe, cfg, 20, 100, 20, 2.0, 0.0)
    spec = spec.model_copy(update={"param_space": param_space})

    console.print(
        f"[bold]Sweep:[/bold] {strategy} × {sum(1 for _ in __import__('itertools').product(*param_space.values()))}"
        f" Kombinationen, rank_by={rank_by}"
    )

    result = run_sweep(spec, md, rank_by=rank_by)

    _print_sweep_result(result)

    # Ergebnis speichern
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f"sweep_{strategy}__{start.date()}__{end.date()}.json"
    result_path.write_text(
        _json.dumps(result.model_dump(mode="json", exclude_none=True), indent=2, default=str)
    )
    console.print(f"[blue]→[/blue] {result_path}")


def _default_param_space(strategy: str) -> dict:
    """Vordefinierte Sweep-Grids für bekannte Strategien."""
    if strategy == "sma_crossover":
        return {
            "fast": [5, 10, 15, 20, 30, 50],
            "slow": [50, 100, 150, 200],
        }
    elif strategy == "mean_reversion":
        return {
            "lookback": [10, 15, 20, 30, 50],
            "entry_z": [1.5, 2.0, 2.5, 3.0],
            "exit_z": [-0.5, 0.0, 0.5],
        }
    else:
        raise typer.BadParameter(
            f"Kein Default-param_space für '{strategy}'. Nutze --params mit JSON."
        )


def _print_sweep_result(sr: SweepResult) -> None:
    """Gibt die Sweep-Ergebnisse als Rich-Tabelle aus."""
    t = Table(
        title=f"Sweep {sr.strategy_id} — {sr.completed}/{sr.total_combinations} Runs, rank_by={sr.rank_by}"
    )
    t.add_column("#", justify="right")
    t.add_column("Parameter")
    t.add_column("Sharpe", justify="right")
    t.add_column("CAGR", justify="right")
    t.add_column("Sortino", justify="right")
    t.add_column("MaxDD", justify="right")
    t.add_column("Trades", justify="right")
    t.add_column("WinRate", justify="right")

    for i, run in enumerate(sr.runs, 1):
        r = run.result
        params_str = ", ".join(f"{k}={v}" for k, v in sorted(run.params.items()))
        highlight = "[bold green]" if i == 1 else ""
        end_h = "[/bold green]" if highlight else ""
        t.add_row(
            str(i),
            f"{highlight}{params_str}{end_h}",
            f"{highlight}{r.sharpe:.2f}{end_h}",
            f"{r.cagr:.2%}",
            f"{r.sortino:.2f}",
            f"{r.max_drawdown:.2%}",
            str(r.trades.n_trades),
            f"{r.trades.win_rate:.0%}",
        )
    console.print(t)

    if sr.best_run:
        console.print(
            f"\n[bold green]🏆 Bester Run:[/bold green] {sr.best_params} "
            f"→ {sr.rank_by} = {sr.best_metric_value:.4f}"
        )


@app.command()
def walkforward(
    strategy: str = typer.Option(..., help="z.B. sma_crossover, mean_reversion"),
    universe: str = typer.Option(...),
    start: datetime = typer.Option(..., formats=["%Y-%m-%d"]),
    end: datetime = typer.Option(..., formats=["%Y-%m-%d"]),
    folds: int = typer.Option(4, help="Anzahl der Walk-Forward-Folds"),
    train_ratio: float = typer.Option(0.6, help="Anteil der In-Sample-Daten pro Fold"),
    rank_by: str = typer.Option("sharpe", help="Metrik für den In-Sample Sweep"),
    param_space_json: str = typer.Option(
        "",
        "--params",
        help='JSON param_space, z.B. \'{"fast": [10,20], "slow": [100]}\'',
    ),
    out: Path = typer.Option(Path("backtests/walkforward"), help="Ergebnis-Verzeichnis"),
) -> None:
    """Führt eine Walk-Forward-Validation durch (Sweep auf Train, Test auf Test, rollierend)."""
    import json as _json

    cfg = _load_universe_yaml(universe)
    md = load_universe(
        universe=universe,
        symbols=cfg["symbols"],
        start=start.date(),
        end=end.date(),
        timeframe=Timeframe(cfg.get("timeframe", "1d")),
    )

    if param_space_json:
        param_space = _json.loads(param_space_json)
    else:
        param_space = _default_param_space(strategy)

    _, spec = _build_spec(strategy, universe, cfg, 20, 100, 20, 2.0, 0.0)
    spec = spec.model_copy(update={"param_space": param_space})

    console.print(
        f"[bold]Walk-Forward:[/bold] {strategy} über {folds} Folds (train={train_ratio:.0%})"
    )

    result = run_walk_forward(spec, md, n_folds=folds, train_ratio=train_ratio, rank_by=rank_by)

    _print_walkforward_result(result)

    # Save to file
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f"wf_{strategy}_{folds}folds__{start.date()}__{end.date()}.json"
    result_path.write_text(
        _json.dumps(result.model_dump(mode="json", exclude_none=True), indent=2, default=str)
    )
    console.print(f"[blue]→[/blue] {result_path}")


def _print_walkforward_result(wf: WalkForwardResult) -> None:
    """Gibt das Walk-Forward-Ergebnis als Rich-Tabelle aus."""
    console.print("\n[bold]Walk-Forward Folds[/bold]")
    t = Table()
    t.add_column("Fold")
    t.add_column("Zeitraum (Train / Test)")
    t.add_column("Gewählte Params")
    t.add_column("IS Sharpe", justify="right")
    t.add_column("OOS Sharpe", justify="right")
    t.add_column("OOS CAGR", justify="right")

    for f in wf.folds:
        params_str = ", ".join(f"{k}={v}" for k, v in f.chosen_params.items())
        t.add_row(
            str(f.fold_index),
            f"Train: {f.train_start}..{f.train_end}\nTest: {f.test_start}..{f.test_end}",
            params_str,
            f"{f.train_sharpe:.2f}",
            f"{f.test_sharpe:.2f}",
            f"{f.test_cagr:.2%}",
        )
    console.print(t)

    console.print("\n[bold]Aggregierte Metriken[/bold]")
    deg_color = "green" if wf.degradation > 0.8 else ("yellow" if wf.degradation > 0.5 else "red")
    console.print(f"IS Sharpe Mean:  {wf.is_sharpe_mean:.2f}")
    console.print(f"OOS Sharpe Mean: {wf.oos_sharpe_mean:.2f}")
    console.print(f"Degradation:     [{deg_color}]{wf.degradation:.2f}[/{deg_color}] (OOS / IS)")


def _build_spec(
    strategy: str,
    universe: str,
    cfg: dict,
    fast: int,
    slow: int,
    lookback: int,
    entry_z: float,
    exit_z: float,
) -> tuple[str, StrategySpec]:
    if strategy == "sma_crossover":
        params = {"fast": fast, "slow": slow}
        sid = f"sma_{fast}_{slow}"
        class_path = "strategies.templates.sma_crossover:SmaCrossover"
        klass = "trend_following"
    elif strategy == "mean_reversion":
        params = {"lookback": lookback, "entry_z": entry_z, "exit_z": exit_z}
        sid = f"mr_{lookback}_{entry_z}_{exit_z}"
        class_path = "strategies.templates.mean_reversion:MeanReversion"
        klass = "mean_reversion"
    else:
        raise typer.BadParameter(f"Unbekannte Strategie: {strategy}")

    return sid, StrategySpec(
        strategy_id=sid,
        name=f"{strategy} {params}",
        class_path=class_path,
        strategy_class=klass,
        universe=universe,
        timeframe=Timeframe(cfg.get("timeframe", "1d")),
        params=params,
    )


def _print_result(r) -> None:
    t = Table(title=f"Backtest {r.strategy_id}  {r.start}..{r.end}")
    t.add_column("Metrik")
    t.add_column("Wert", justify="right")
    rows = [
        ("CAGR", f"{r.cagr:.2%}"),
        ("Sharpe", f"{r.sharpe:.2f}"),
        ("Sortino", f"{r.sortino:.2f}"),
        ("Calmar", f"{r.calmar:.2f}"),
        ("Max DD", f"{r.max_drawdown:.2%}"),
        ("Ulcer", f"{r.ulcer_index:.4f}"),
        ("Trades", str(r.trades.n_trades)),
        ("Win Rate", f"{r.trades.win_rate:.2%}"),
        ("Profit Factor", f"{r.trades.profit_factor:.2f}"),
    ]
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)


SORT_KEYS = {
    "sharpe": "sharpe",
    "sortino": "sortino",
    "cagr": "cagr",
    "calmar": "calmar",
    "maxdd": "max_drawdown",
    "ulcer": "ulcer_index",
    "trades": "n_trades",
    "winrate": "win_rate",
    "pf": "profit_factor",
}


@app.command()
def compare(
    results_dir: Path = typer.Option(Path("backtests/results"), help="Ordner mit JSON-Ergebnissen"),
    sort_by: str = typer.Option("sharpe", help=f"Sortierschlüssel: {', '.join(SORT_KEYS)}"),
    top: int = typer.Option(20, help="Top-N anzeigen"),
    pattern: str = typer.Option("", help="Filtert strategy_id per Substring"),
    ascending: bool = typer.Option(
        False, "--asc", help="Aufsteigend sortieren (Default: absteigend)"
    ),
) -> None:
    """Vergleicht alle Backtest-Ergebnisse in einem Verzeichnis als Tabelle."""
    if sort_by not in SORT_KEYS:
        raise typer.BadParameter(f"sort_by muss eines von {list(SORT_KEYS)} sein")

    if not results_dir.exists():
        raise typer.BadParameter(f"Verzeichnis fehlt: {results_dir}")

    rows = []
    for jf in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except json.JSONDecodeError:
            console.print(f"[yellow]Skip {jf.name}: kaputtes JSON[/yellow]")
            continue
        if pattern and pattern not in data.get("strategy_id", ""):
            continue
        trades = data.get("trades", {})
        rows.append(
            {
                "file": jf.stem,
                "strategy_id": data.get("strategy_id", "?"),
                "start": data.get("start", "?"),
                "end": data.get("end", "?"),
                "cagr": float(data.get("cagr", 0.0)),
                "sharpe": float(data.get("sharpe", 0.0)),
                "sortino": float(data.get("sortino", 0.0)),
                "calmar": float(data.get("calmar", 0.0)),
                "max_drawdown": float(data.get("max_drawdown", 0.0)),
                "ulcer_index": float(data.get("ulcer_index", 0.0)),
                "n_trades": int(trades.get("n_trades", 0)),
                "win_rate": float(trades.get("win_rate", 0.0)),
                "profit_factor": float(trades.get("profit_factor", 0.0)),
            }
        )

    if not rows:
        console.print("[yellow]Keine Ergebnisse gefunden.[/yellow]")
        return

    key = SORT_KEYS[sort_by]
    rows.sort(key=lambda r: r[key], reverse=not ascending)
    rows = rows[:top]

    t = Table(title=f"Backtests — sortiert nach {sort_by} ({'asc' if ascending else 'desc'})")
    for col in (
        "Strategy",
        "Range",
        "CAGR",
        "Sharpe",
        "Sort",
        "Calm",
        "MaxDD",
        "Ulcer",
        "Trd",
        "Win",
        "PF",
    ):
        t.add_column(col, justify="right" if col not in ("Strategy", "Range") else "left")

    best = rows[0][key]
    for r in rows:
        highlight = "[bold green]" if r[key] == best else ""
        end_h = "[/bold green]" if highlight else ""
        t.add_row(
            r["strategy_id"],
            f"{r['start']}..{r['end']}",
            f"{r['cagr']:.2%}",
            f"{highlight}{r['sharpe']:.2f}{end_h}",
            f"{r['sortino']:.2f}",
            f"{r['calmar']:.2f}",
            f"{r['max_drawdown']:.2%}",
            f"{r['ulcer_index']:.4f}",
            str(r["n_trades"]),
            f"{r['win_rate']:.0%}",
            f"{r['profit_factor']:.2f}",
        )
    console.print(t)
    console.print(
        f"[dim]{len(rows)} Zeilen angezeigt aus {len(list(results_dir.glob('*.json')))} Files[/dim]"
    )


@app.command()
def report(
    results_dir: Path = typer.Option(Path("backtests/results"), help="Ordner mit JSON-Ergebnissen"),
    out: Path = typer.Option(Path("reports/backtest_report.md"), help="Ziel-Markdown-Datei"),
    sort_by: str = typer.Option("sharpe", help=f"Sortierschlüssel: {', '.join(SORT_KEYS)}"),
    top: int = typer.Option(20, help="Top-N im Ranking"),
    title: str = typer.Option("Backtest-Report", help="Titel im Markdown"),
) -> None:
    """Erzeugt einen Markdown-Report aus allen Backtest-JSONs, copy-paste-bereit für PRs."""
    if sort_by not in SORT_KEYS:
        raise typer.BadParameter(f"sort_by muss eines von {list(SORT_KEYS)} sein")
    if not results_dir.exists():
        raise typer.BadParameter(f"Verzeichnis fehlt: {results_dir}")

    rows = _load_results(results_dir)
    if not rows:
        console.print("[yellow]Keine Ergebnisse gefunden.[/yellow]")
        return

    md = _render_markdown(rows, sort_by=sort_by, top=top, title=title)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    console.print(f"[green]OK[/green] Report geschrieben: {out} ({len(rows)} Ergebnisse)")
    console.print(f"[dim]Zum Einbetten: cat {out} | xclip -selection clipboard[/dim]")


def _load_results(results_dir: Path) -> list[dict]:
    rows = []
    for jf in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except json.JSONDecodeError:
            continue
        trades = data.get("trades", {})
        rows.append(
            {
                "file": jf.stem,
                "strategy_id": data.get("strategy_id", "?"),
                "start": data.get("start", "?"),
                "end": data.get("end", "?"),
                "cagr": float(data.get("cagr", 0.0)),
                "sharpe": float(data.get("sharpe", 0.0)),
                "sortino": float(data.get("sortino", 0.0)),
                "calmar": float(data.get("calmar", 0.0)),
                "max_drawdown": float(data.get("max_drawdown", 0.0)),
                "ulcer_index": float(data.get("ulcer_index", 0.0)),
                "n_trades": int(trades.get("n_trades", 0)),
                "win_rate": float(trades.get("win_rate", 0.0)),
                "profit_factor": float(trades.get("profit_factor", 0.0)),
            }
        )
    return rows


def _render_markdown(rows: list[dict], sort_by: str, top: int, title: str) -> str:
    key = SORT_KEYS[sort_by]
    sorted_rows = sorted(rows, key=lambda r: r[key], reverse=True)[:top]

    lines: list[str] = []
    lines.append(f"# 📈 {title}")
    lines.append("")
    lines.append("> [!info] Report Info")
    lines.append(f"> **Generiert am:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> **Backtests gesamt:** {len(rows)}")
    lines.append(f"> **Sortiert nach:** `{sort_by}`")
    lines.append("")

    # Summary
    def avg(k):
        return sum(r[k] for r in rows) / len(rows)
    best = max(rows, key=lambda r: r["sharpe"])
    worst_dd = min(rows, key=lambda r: r["max_drawdown"])

    lines.append("## 📊 Performance Übersicht")
    lines.append("")
    lines.append("> [!abstract] Durchschnittswerte")
    lines.append(f"> - **Sharpe:** {avg('sharpe'):.2f}")
    lines.append(f"> - **CAGR:** {avg('cagr'):.2%}")
    lines.append(f"> - **Max DD:** {avg('max_drawdown'):.2%}")
    lines.append("")
    lines.append("> [!success] Highlights")
    lines.append(
        f"> - **Bester Sharpe:** `{best['strategy_id']}` "
        f"({best['sharpe']:.2f}, CAGR {best['cagr']:.2%}, Max DD {best['max_drawdown']:.2%})"
    )
    lines.append(
        f"> - **Tiefster Drawdown:** `{worst_dd['strategy_id']}` ({worst_dd['max_drawdown']:.2%})"
    )
    lines.append("")

    # Ranking-Tabelle
    lines.append(f"## 🏆 Ranking nach {sort_by} (Top {top})")
    lines.append("")
    lines.append(
        "| Rang | Strategy | Zeitraum | CAGR | Sharpe | Sortino | Calmar | MaxDD | Ulcer | Trades | Win% | PF |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(sorted_rows, 1):
        marker = " 🥇" if i == 1 else (" 🥈" if i == 2 else (" 🥉" if i == 3 else ""))
        lines.append(
            f"| {i}{marker} | `{r['strategy_id']}` | {r['start']}..{r['end']} | "
            f"{r['cagr']:.2%} | **{r['sharpe']:.2f}** | {r['sortino']:.2f} | {r['calmar']:.2f} | "
            f"{r['max_drawdown']:.2%} | {r['ulcer_index']:.4f} | {r['n_trades']} | "
            f"{r['win_rate']:.0%} | {r['profit_factor']:.2f} |"
        )
    lines.append("")

    # Per-Dimension Bestenliste
    lines.append("## 🎯 Bestenlisten pro Dimension")
    lines.append("")
    dims = [
        ("Höchster Sharpe", "sharpe", "{:.2f}"),
        ("Höchster CAGR", "cagr", "{:.2%}"),
        ("Geringster Drawdown", "max_drawdown", "{:.2%}"),
        ("Höchster Profit Factor", "profit_factor", "{:.2f}"),
        ("Höchster Calmar", "calmar", "{:.2f}"),
    ]
    for label, k, fmt in dims:
        # max_drawdown ist negativ — "geringster Drawdown" heißt am nächsten zu 0, also descending
        reverse = True
        top3 = sorted(rows, key=lambda r: r[k], reverse=reverse)[:3]
        bullets = [f"`{r['strategy_id']}` ({fmt.format(r[k])})" for r in top3]
        lines.append(f"- **{label}:** " + " · ".join(bullets))
    lines.append("")

    # Beobachtungen-Stub
    lines.append("## 📝 Beobachtungen")
    lines.append("")
    lines.append("> [!note] Erkenntnisse & Analyse")
    lines.append(
        "> _(Manuell ergänzen: Was sticht heraus? Welche Parameterregion lohnt weitere Tests? Gibt es Cluster?)_"
    )
    lines.append("")

    return "\n".join(lines)


@app.command()
def plan() -> None:
    """Druckt den 5-Phasen-Plan auf den Terminal."""
    phases = [
        (
            "Phase 1 — Fundament",
            "Daten, Modelle, Strategie-Interface, Backtest-Runner, Obsidian-Sync",
        ),
        (
            "Phase 2 — Research-Framework",
            "Templates, Sweeps, Walk-Forward, Score, Rejection-Regeln",
        ),
        ("Phase 3 — KI-Unterstützung", "Agenten priorisieren Tests, schreiben Memos"),
        (
            "Phase 4 — Paper Trading (IBKR)",
            "Monitoring, Alerts, Drift-Erkennung, menschliche Freigabe",
        ),
        ("Phase 5 — Skalierung", "Multi-Asset, Intraday, Event-basiert"),
    ]
    t = Table(title="QuantRace — Phasenplan")
    t.add_column("Phase")
    t.add_column("Inhalt")
    for k, v in phases:
        t.add_row(k, v)
    console.print(t)


if __name__ == "__main__":
    app()
