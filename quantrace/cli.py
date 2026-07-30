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

from quantrace import strategy_registry
from quantrace.data_agent import close_prices, load_universe
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
    provider: str | None = typer.Option(
        None,
        help="OpenBB-Provider: yfinance, tiingo, fmp, polygon. Default: TIINGO_TOKEN gesetzt → tiingo, sonst yfinance.",
    ),
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
    strategy: str = typer.Option(
        "", help="z.B. sma_crossover, mean_reversion, buy_and_hold"
    ),
    graph_spec: str = typer.Option(
        "",
        "--graph-spec",
        help="Slug einer Graph-Strategie aus 02 Strategien/ (visueller Builder). "
        "Alternative zu --strategy.",
    ),
    universe: str = typer.Option(...),
    start: datetime = typer.Option(..., formats=["%Y-%m-%d"]),
    end: datetime = typer.Option(..., formats=["%Y-%m-%d"]),
    fast: int = typer.Option(20, hidden=True),
    slow: int = typer.Option(100, hidden=True),
    lookback: int = typer.Option(20, hidden=True),
    entry_z: float = typer.Option(2.0, hidden=True),
    exit_z: float = typer.Option(0.0, hidden=True),
    param_json: str = typer.Option(
        "",
        "--params",
        help='JSON-Strategie-Params, z.B. \'{"fast": 20, "slow": 100}\'. '
        "Überschreibt die typisierten Optionen und gilt für alle Strategien.",
    ),
    cost_model: str = typer.Option(
        "flat",
        help="Kostenmodell: 'flat' (fees/slippage global) oder "
        "'per_asset_class' (config/costs.yaml, pro Symbol)",
    ),
    capital_model: str = typer.Option(
        "shared",
        help="Kapitalmodell: 'shared' (ein Konto, Equal-Weight über aktive "
        "Positionen) oder 'independent' (Alt: unabhängige Sleeves, Mittelwert)",
    ),
    out: Path = typer.Option(Path("backtests/results"), help="Ergebnis-Verzeichnis"),
) -> None:
    """Führt einen Backtest aus und schreibt das Ergebnis als JSON."""
    import json as _json

    from quantrace.backtest_runner import run_backtest

    cfg = _load_universe_yaml(universe)
    md = load_universe(
        universe=universe,
        symbols=cfg["symbols"],
        start=start.date(),
        end=end.date(),
        timeframe=Timeframe(cfg.get("timeframe", "1d")),
    )

    if param_json:
        params = _json.loads(param_json)
    else:
        params = _legacy_params(strategy, fast, slow, lookback, entry_z, exit_z)
    strategy_id, spec = _resolve_spec(strategy, graph_spec, universe, cfg, params)
    result = run_backtest(spec, md, _backtest_config(cost_model, capital_model))

    # Attach regime-conditioned performance metrics while equity_curve is in memory.
    if result.equity_curve is not None:
        try:
            from quantrace.regime.backtesting import regime_conditioned_metrics

            rm = regime_conditioned_metrics(result.equity_curve, close_prices(md))
            result = result.model_copy(update={"regime_metrics": rm})
        except Exception as _re:
            logging.getLogger(__name__).warning("regime metrics skipped: %s", _re)

    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f"{strategy_id}__{start.date()}__{end.date()}.json"
    result_path.write_text(
        json.dumps(result.model_dump(mode="json", exclude={"equity_curve"}), indent=2, default=str)
    )
    _print_result(result)
    console.print(f"[blue]→[/blue] {result_path}")


@app.command()
def sweep(
    strategy: str = typer.Option("", help="z.B. sma_crossover, mean_reversion"),
    graph_spec: str = typer.Option(
        "",
        "--graph-spec",
        help="Slug einer Graph-Strategie aus 02 Strategien/ (visueller Builder). "
        "Alternative zu --strategy.",
    ),
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
    cost_model: str = typer.Option(
        "flat",
        help="Kostenmodell: 'flat' oder 'per_asset_class' (config/costs.yaml)",
    ),
    capital_model: str = typer.Option(
        "shared",
        help="Kapitalmodell: 'shared' (ein Konto) oder 'independent' (Alt-Semantik)",
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
    label, spec = _resolve_spec(strategy, graph_spec, universe, cfg, {})
    if param_space_json:
        param_space = _json.loads(param_space_json)
    elif graph_spec:
        # Graph-Grids stehen im Frontmatter der Note (dotted "<node>.<param>").
        param_space = spec.param_space
        if not param_space:
            raise typer.BadParameter(
                f"Graph-Spec '{graph_spec}' hat kein `param_space` im Frontmatter — "
                "Grid mit --params angeben oder im Builder setzen."
            )
    else:
        param_space = _default_param_space(strategy)

    spec = spec.model_copy(update={"param_space": param_space})

    console.print(
        f"[bold]Sweep:[/bold] {label} × {sum(1 for _ in __import__('itertools').product(*param_space.values()))}"
        f" Kombinationen, rank_by={rank_by}"
    )

    result = run_sweep(
        spec, md, config=_backtest_config(cost_model, capital_model), rank_by=rank_by
    )

    _print_sweep_result(result)

    # Ergebnis speichern
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f"sweep_{label}__{start.date()}__{end.date()}.json"
    result_path.write_text(
        _json.dumps(result.model_dump(mode="json", exclude_none=True), indent=2, default=str)
    )
    console.print(f"[blue]→[/blue] {result_path}")


def _backtest_config(cost_model: str, capital_model: str = "shared") -> BacktestConfig:
    """BacktestConfig aus CLI-Optionen — validiert Kosten- + Kapitalmodell früh."""
    from quantrace.models import CAPITAL_MODELS, COST_MODELS

    if cost_model not in COST_MODELS:
        raise typer.BadParameter(
            f"Unbekanntes Kostenmodell '{cost_model}'. Erlaubt: {', '.join(COST_MODELS)}."
        )
    if capital_model not in CAPITAL_MODELS:
        raise typer.BadParameter(
            f"Unbekanntes Kapitalmodell '{capital_model}'. Erlaubt: {', '.join(CAPITAL_MODELS)}."
        )
    return BacktestConfig(cost_model=cost_model, capital_model=capital_model)  # type: ignore[arg-type]


def _default_param_space(strategy: str) -> dict:
    """Vordefinierte Sweep-Grids aus der Registry."""
    if not strategy_registry.is_registered(strategy):
        raise typer.BadParameter(
            f"Unbekannte Strategie '{strategy}'. Bekannt: "
            f"{', '.join(strategy_registry.known_strategies())}"
        )
    space = strategy_registry.default_param_space(strategy)
    if not space:
        raise typer.BadParameter(
            f"Strategie '{strategy}' hat keine tunebaren Parameter — Sweep/Walk-Forward "
            "sind hier sinnlos. Nutze einen Single-Backtest oder --params mit JSON."
        )
    return space


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
    strategy: str = typer.Option("", help="z.B. sma_crossover, mean_reversion"),
    graph_spec: str = typer.Option(
        "",
        "--graph-spec",
        help="Slug einer Graph-Strategie aus 02 Strategien/ (visueller Builder). "
        "Alternative zu --strategy.",
    ),
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
    cost_model: str = typer.Option(
        "flat",
        help="Kostenmodell: 'flat' oder 'per_asset_class' (config/costs.yaml)",
    ),
    capital_model: str = typer.Option(
        "shared",
        help="Kapitalmodell: 'shared' (ein Konto) oder 'independent' (Alt-Semantik)",
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

    label, spec = _resolve_spec(strategy, graph_spec, universe, cfg, {})
    if param_space_json:
        param_space = _json.loads(param_space_json)
    elif graph_spec:
        param_space = spec.param_space
        if not param_space:
            raise typer.BadParameter(
                f"Graph-Spec '{graph_spec}' hat kein `param_space` im Frontmatter — "
                "Grid mit --params angeben oder im Builder setzen."
            )
    else:
        param_space = _default_param_space(strategy)

    spec = spec.model_copy(update={"param_space": param_space})

    console.print(
        f"[bold]Walk-Forward:[/bold] {label} über {folds} Folds (train={train_ratio:.0%})"
    )

    result = run_walk_forward(
        spec,
        md,
        config=_backtest_config(cost_model, capital_model),
        n_folds=folds,
        train_ratio=train_ratio,
        rank_by=rank_by,
    )

    _print_walkforward_result(result)

    # Save to file
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / f"wf_{label}_{folds}folds__{start.date()}__{end.date()}.json"
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


def _legacy_params(
    strategy: str, fast: int, slow: int, lookback: int, entry_z: float, exit_z: float
) -> dict:
    """Mappt die typisierten CLI-Optionen auf die Param-Dicts der zwei Alt-Strategien.

    Für alle anderen Strategien gilt ein leeres Dict → die Klassen-Defaults greifen
    (siehe strategy_registry.build_spec). So müssen sma/mr ihre `--fast 20 --slow 1`
    -Ergonomie nicht verlieren, während die übrigen 8 Strategien sauber über
    `--params` oder ihre Defaults laufen.
    """
    if strategy == "sma_crossover":
        return {"fast": fast, "slow": slow}
    if strategy == "mean_reversion":
        return {"lookback": lookback, "entry_z": entry_z, "exit_z": exit_z}
    return {}


def _build_spec(
    strategy: str,
    universe: str,
    cfg: dict,
    params: dict | None = None,
) -> tuple[str, StrategySpec]:
    try:
        return strategy_registry.build_spec(
            strategy, universe, Timeframe(cfg.get("timeframe", "1d")), params
        )
    except KeyError as e:
        raise typer.BadParameter(str(e)) from e


def _resolve_spec(
    strategy: str | None,
    graph_spec: str | None,
    universe: str,
    cfg: dict,
    params: dict | None = None,
) -> tuple[str, StrategySpec]:
    """Registry-Strategie ODER Graph-Spec aus dem Vault — genau eine von beiden.

    Der Graph-Pfad (#188) lässt eine im visuellen Builder gebaute Strategie
    durch denselben Runner laufen; die Registry bleibt die statische Whitelist
    der Code-Strategien.
    """
    if bool(strategy) == bool(graph_spec):
        raise typer.BadParameter(
            "Genau eine von --strategy oder --graph-spec angeben."
        )
    if graph_spec:
        from quantrace.graph import vault as graph_vault
        from quantrace.graph.compiler import GraphValidationError

        try:
            return graph_vault.build_spec(
                graph_spec, universe, Timeframe(cfg.get("timeframe", "1d")), params
            )
        except (FileNotFoundError, ValueError) as e:
            raise typer.BadParameter(str(e)) from e
        except GraphValidationError as e:
            raise typer.BadParameter(
                f"Graph-Spec '{graph_spec}' ist ungültig:\n" + "\n".join(e.errors)
            ) from e
    return _build_spec(strategy or "", universe, cfg, params)


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
def paper_status() -> None:
    """Verbindet sich mit Alpaca Paper-Account und gibt Kontoinfo + Positionen aus.

    Smoke-Test für die Phase-2-Integration. Setze ALPACA_API_KEY und
    ALPACA_SECRET_KEY in der Env, dann läuft das ohne weitere Konfiguration.
    """
    from quantrace.brokers import get_broker

    try:
        broker = get_broker("alpaca")
        broker.connect()
    except (RuntimeError, ImportError) as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(1) from exc

    equity = broker.account_value()
    positions = broker.positions()

    console.print(f"[green]✓[/green] Alpaca paper account: equity=${equity:,.2f}")
    if not positions:
        console.print("[dim]Keine offenen Positionen.[/dim]")
        return

    t = Table(title="Offene Positionen")
    t.add_column("Symbol")
    t.add_column("Qty", justify="right")
    t.add_column("Avg Cost", justify="right")
    t.add_column("Market Value", justify="right")
    for p in positions:
        t.add_row(
            p.symbol,
            f"{p.quantity:.4f}",
            f"${p.avg_cost:,.2f}",
            f"${p.market_value:,.2f}",
        )
    console.print(t)


@app.command()
def regime(
    universe: str = typer.Option(..., help="Name in data/universes/*.yaml"),
    start: datetime = typer.Option(..., formats=["%Y-%m-%d"]),
    end: datetime = typer.Option(..., formats=["%Y-%m-%d"]),
    n_states: int = typer.Option(3, help="Anzahl HMM-Regime (2–5)"),
    feature_window: int = typer.Option(21, help="Trailing-Fenster für Trend/Vol-Features"),
) -> None:
    """Schätzt das Markt-Regime eines Universums per Hidden-Markov-Model.

    Fittet ein Gaussian-HMM auf Trend-/Vol-Features des Equal-Weight-Benchmarks
    und gibt das aktuelle (kausale) Regime, dessen Konfidenz, die Regime-
    Verteilung über den Zeitraum und die geschätzten Regime-Charakteristika aus.
    """
    from quantrace.regime import RegimeDetector

    cfg = _load_universe_yaml(universe)
    md = load_universe(
        universe=universe,
        symbols=cfg["symbols"],
        start=start.date(),
        end=end.date(),
        timeframe=Timeframe(cfg.get("timeframe", "1d")),
    )

    bench = close_prices(md).mean(axis=1)
    det = RegimeDetector(n_states=n_states, feature_window=feature_window).fit(bench)
    snap = det.current_regime(bench)
    series = det.regime_series(bench)

    emoji = {
        "crisis": "🔴", "risk_off": "🟠", "neutral": "🟡",
        "risk_on": "🟢", "euphoria": "🚀",
    }
    console.print(
        f"\n[bold]Aktuelles Regime ({snap.as_of.date()}):[/bold] "
        f"{emoji.get(snap.label, '•')} [bold]{snap.label}[/bold] "
        f"([cyan]{snap.confidence:.0%}[/cyan] Konfidenz)"
    )

    prob_t = Table(title="Regime-Wahrscheinlichkeiten (heute)")
    prob_t.add_column("Regime")
    prob_t.add_column("P", justify="right")
    for label in det.labels:
        p = snap.probabilities.get(label, 0.0)
        prob_t.add_row(f"{emoji.get(label, '•')} {label}", f"{p:.1%}")
    console.print(prob_t)

    dist = series.value_counts(normalize=True)
    dist_t = Table(title=f"Regime-Verteilung {start.date()}..{end.date()}")
    dist_t.add_column("Regime")
    dist_t.add_column("Anteil Tage", justify="right")
    for label in det.labels:
        share = float(dist.get(label, 0.0))
        dist_t.add_row(f"{emoji.get(label, '•')} {label}", f"{share:.1%}")
    console.print(dist_t)

    means = det.hmm.means_
    char_t = Table(title="Regime-Charakteristik (annualisiert, geschätzt)")
    char_t.add_column("Regime")
    char_t.add_column("Trend μ", justify="right")
    char_t.add_column("Vol σ", justify="right")
    for state, label in sorted(det.state_to_label_.items(), key=lambda kv: det.labels.index(kv[1])):
        char_t.add_row(
            f"{emoji.get(label, '•')} {label}",
            f"{means[state, 0]:+.1%}",
            f"{means[state, 1]:.1%}",
        )
    console.print(char_t)


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
