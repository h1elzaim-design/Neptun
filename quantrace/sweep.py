"""Parameter-Sweep — Grid-Search über den param_space einer StrategySpec.

Erzeugt das kartesische Produkt aller Werte in `StrategySpec.param_space`,
führt pro Kombination einen Backtest durch und liefert die Ergebnisse
sortiert nach einer konfigurierbaren Metrik zurück.

Beispiel:
    spec.param_space = {"fast": [10, 20, 50], "slow": [100, 200]}
    → 6 Backtests (3×2)
"""

from __future__ import annotations

import itertools
import logging
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from quantrace.backtest_runner import run_backtest
from quantrace.models import (
    BacktestConfig,
    BacktestResult,
    MarketData,
    StrategySpec,
)
from quantrace.stats import (
    DEFAULT_FDR_ALPHA,
    annualised_sharpe,
    benjamini_hochberg,
    bootstrap_drawdown_distribution,
    bootstrap_sharpe_ci,
    deflated_sharpe_from_summary,
    probability_of_backtest_overfitting,
    sample_skew_kurt,
    sharpe_p_value,
)

log = logging.getLogger(__name__)

PERIODS_PER_YEAR = 252.0


@dataclass(frozen=True, slots=True)
class RunReturnStats:
    """Per-period return statistics of one sweep run, read off the in-memory
    equity curve (which is excluded from the serialised JSON)."""

    n_obs: int
    sr_period: float  # μ̂/σ̂ of per-period returns (NOT annualised)
    skew: float
    kurt: float


def _run_returns(result: BacktestResult) -> np.ndarray | None:
    """Per-period return stream off a run's in-memory equity curve.

    Returns None when there isn't enough of an equity curve to work with
    (degenerate slice, flat equity, missing curve).
    """
    eq = getattr(result, "equity_curve", None)
    if eq is None:
        return None
    vals = np.asarray(eq, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 4:
        return None
    rets = np.diff(vals) / vals[:-1]
    rets = rets[np.isfinite(rets)]
    if rets.size < 3:
        return None
    return rets


def _stats_from_returns(rets: np.ndarray | None) -> RunReturnStats | None:
    """Summary statistics (T, per-period Sharpe, moments) of a return stream."""
    if rets is None:
        return None
    # periods_per_year=1 → per-period Sharpe; Degeneriert-Guard (σ≈0) lebt
    # damit nur in quantrace.stats.sharpe.
    sr_period = annualised_sharpe(rets, periods_per_year=1.0)
    skew, kurt = sample_skew_kurt(rets)
    return RunReturnStats(
        n_obs=int(rets.size), sr_period=sr_period, skew=float(skew), kurt=float(kurt)
    )


def _run_return_stats(result: BacktestResult) -> RunReturnStats | None:
    """n_obs, per-period Sharpe and sample moments of a run's return stream."""
    return _stats_from_returns(_run_returns(result))


def _winner_moments(result: BacktestResult) -> tuple[float | None, float | None]:
    """Sample skew/kurtosis of the winning run's per-period returns.

    Kept as the narrow moments view over :func:`_run_return_stats` so the
    persisted SweepResult carries the two scalars the Deflated Sharpe needs to
    drop the Gaussian-null assumption. Returns (None, None) when there isn't
    enough of an equity curve to estimate moments.
    """
    stats = _run_return_stats(result)
    if stats is None:
        return None, None
    return stats.skew, stats.kurt

# Metriken, nach denen sortiert werden kann (höher = besser, außer max_drawdown/ulcer)
RANKABLE_METRICS = Literal[
    "sharpe",
    "sortino",
    "cagr",
    "calmar",
    "max_drawdown",
    "ulcer_index",
    "profit_factor",
]

# Metriken, bei denen niedriger besser ist (Sortierung aufsteigend)
_LOWER_IS_BETTER = {"max_drawdown", "ulcer_index"}


class SweepRun(BaseModel):
    """Ein einzelner Sweep-Lauf: Parameter + Backtest-Ergebnis."""

    params: dict[str, Any]
    result: BacktestResult
    # Multiple-testing stats (H0: true SR ≤ 0, Mertens SE with the run's own
    # return moments; q via Benjamini-Hochberg over all completed runs).
    # None on old results or when the equity curve was too short to test.
    p_value: float | None = None
    q_value: float | None = None
    fdr_significant: bool | None = None


class SweepResult(BaseModel):
    """Ergebnis eines kompletten Parameter-Sweeps."""

    strategy_id: str
    #: Siehe BacktestResult.periods_per_year (#184).
    periods_per_year: float = 252.0
    param_space: dict[str, list[Any]]
    rank_by: str
    total_combinations: int
    completed: int
    failed: int
    runs: list[SweepRun] = Field(default_factory=list)
    best_params: dict[str, Any] = Field(default_factory=dict)
    best_metric_value: float = 0.0
    # Return moments of the winning run, persisted so the Deflated Sharpe can use
    # the real skew/kurtosis instead of a Gaussian null (the equity curve itself
    # is excluded from the JSON to keep payloads small). None on old results.
    best_skew: float | None = None
    best_kurt: float | None = None
    # Statistical discipline of the selection, computed at sweep time from the
    # in-memory return paths (exact — no summary-statistics fallback needed):
    # - best_n_obs: T behind the winner (drives the Mertens SE downstream)
    # - best_dsr:   Deflated Sharpe = P[true SR > E[max SR | null, N trials]]
    # - best_psr:   PSR of the winner vs 0 (no deflation), for contrast
    # - expected_max_sharpe_annual: the selection-bias bar, annualised
    # - fdr: Benjamini-Hochberg summary over all completed runs
    # All None on old results.
    best_n_obs: int | None = None
    best_dsr: float | None = None
    best_psr: float | None = None
    expected_max_sharpe_annual: float | None = None
    fdr: dict[str, Any] | None = None
    # CSCV Probability of Backtest Overfitting (Bailey et al. 2017) over the
    # aligned trial-return matrix: P[IS winner ranks bottom-half OOS]. ~0.5 =
    # selection is noise. None on old results / too little data.
    pbo: dict[str, Any] | None = None
    # Stationary-bootstrap inference for the winner (Politis & Romano 1994):
    # {"sharpe": {...ci/p_value...}, "max_drawdown": {...quantiles...}}.
    # None on old results / too little data.
    best_bootstrap: dict[str, Any] | None = None
    # The winner's equity path ([{date, value}]). Persisted so downstream
    # analyses (portfolio uniqueness, factor attribution, bootstrap) can work
    # on the selected strategy's actual return stream — per-run curves stay
    # excluded from the JSON, only rank 0 keeps its path. None on old results.
    best_equity: list[dict[str, Any]] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def best_run(self) -> SweepRun | None:
        return self.runs[0] if self.runs else None


def _extract_metric(result: BacktestResult, metric: str) -> float:
    """Liest eine Metrik aus dem BacktestResult."""
    if hasattr(result, metric):
        return float(getattr(result, metric))
    if hasattr(result.trades, metric):
        return float(getattr(result.trades, metric))
    raise ValueError(f"Unbekannte Metrik: {metric}")


def _generate_param_grid(param_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Erzeugt das kartesische Produkt aller Parameter-Werte."""
    if not param_space:
        return [{}]
    keys = sorted(param_space.keys())
    values = [param_space[k] for k in keys]
    return [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*values)]


# ---------------------------------------------------------------------------
# Parallele Grid-Ausführung (#210 Punkt 1)
#
# Jede Kombination ist unabhängig — embarrassingly parallel. Der Gewinn ist
# keine Bequemlichkeit: solange Sweep×Walk-Forward teuer ist, wird der billige
# Pfad genommen (so kam `grid_global_macro` ohne WF ins Buch), und die
# Disziplin-Schicht erodiert nicht durch Umgehung, sondern durch Laufzeit.
#
# **Prozesse, nicht Threads:** vectorbt/numba rechnen unter dem GIL.
#
# **`spawn`, nicht `fork`:** unter fork erbt das Kind den bereits
# initialisierten Thread-Pool von OpenBLAS/numba. Die Thread-Limits unten
# griffen dann nicht — die Bibliotheken lesen ihre Env-Variablen beim *Import*,
# und der ist im Elternprozess längst passiert. Zwei Prozesse mit je zwei
# BLAS-Threads auf zwei Kernen sind Überbuchung; der Sweep würde langsamer statt
# schneller. `spawn` importiert im Kind frisch, damit greifen die Limits.
#
# Der Preis von spawn ist ein vectorbt-Import pro Worker (~Sekunden). Deshalb
# der Mindest-Grid-Schwellwert: bei vier Kombinationen frisst der Poolstart den
# Gewinn auf.

#: Vom Worker-Initializer gesetzt. Die MarketData wandert **einmal pro Worker**
#: über die Prozessgrenze (via `initargs`), nicht einmal pro Kombination.
_W_SPEC: StrategySpec | None = None
_W_DATA: MarketData | None = None
_W_CONFIG: BacktestConfig | None = None

#: Jeder Worker rechnet single-threaded. Ohne das multipliziert sich
#: Prozess-Parallelität mit BLAS-Thread-Parallelität.
_THREAD_LIMIT_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: Unter diesem Grid lohnt der Poolstart nicht — und die Grenze hängt stark
#: davon ab, wie der Prozess entsteht. Gemessen auf 4 Kernen (Zahlen in der
#: PR-Beschreibung):
#:
#:   fork :  24 Kombis → 4.4× (2 Proz.),  200 → 1.8×/3.3× (2/4 Proz.)
#:   spawn:  24 Kombis → 0.85×,           200 → 1.26×/1.54×
#:
#: Unter `spawn` zahlt jeder Worker vectorbt-Import **und** eine volle
#: numba-JIT-Runde; das kommt erst jenseits von ~100 Kombinationen herein, und
#: darunter ist parallel schlicht langsamer als seriell. Unter `fork` erbt das
#: Kind beides, der Fixkostenblock entfällt.
_MIN_GRID_BY_START_METHOD = {"fork": 4, "forkserver": 4, "spawn": 128}
DEFAULT_MIN_GRID_FOR_PARALLEL = 4


def _worker_init(spec: StrategySpec, data: MarketData, config: BacktestConfig) -> None:
    """Läuft einmal pro Worker-Prozess, vor der ersten Kombination."""
    for var in _THREAD_LIMIT_VARS:
        os.environ[var] = "1"
    global _W_SPEC, _W_DATA, _W_CONFIG
    _W_SPEC, _W_DATA, _W_CONFIG = spec, data, config


def _worker_run(item: tuple[int, dict[str, Any]]) -> tuple[int, BacktestResult | None, str | None]:
    """Eine Kombination im Worker. Gibt **immer** den Grid-Index zurück.

    Der Index ist der Grund, warum das Ergebnis deterministisch bleibt: der
    Aufrufer sortiert danach ein, nicht nach Fertigstellungsreihenfolge.

    Fehler werden als Text zurückgegeben statt geworfen. Eine Exception über die
    Prozessgrenze würde den `map`-Aufruf abbrechen und damit den ganzen Sweep
    kippen — heute zählt die serielle Schleife pro Kombination hoch und läuft
    weiter, und genau das muss erhalten bleiben.
    """
    index, params = item
    if _W_SPEC is None or _W_DATA is None or _W_CONFIG is None:  # pragma: no cover
        return index, None, "Worker ohne Initialisierung"
    try:
        run_spec = _W_SPEC.model_copy(update={"params": {**_W_SPEC.params, **params}})
        return index, run_backtest(run_spec, _W_DATA, _W_CONFIG), None
    except Exception as exc:  # noqa: BLE001 — 1:1 das Verhalten der seriellen Schleife
        return index, None, str(exc)


def available_cpus() -> int:
    """Wie viele Kerne uns **tatsächlich** zustehen.

    `os.cpu_count()` meldet die Kerne der *Maschine*, nicht das Kontingent des
    Containers. Die Container App hat 2 vCPU (``infra/azure/container-app.bicep``),
    läuft aber auf einem deutlich größeren Knoten — nach `cpu_count` würden wir
    ein Vielfaches an Prozessen starten. Jeder davon importiert vectorbt/numba
    mit dreistelligem MB-RSS; in 4 GB ist das ein OOM-Kill, kein Speedup.

    Die cgroup-Quote ist die einzige Quelle, die das Kontingent kennt. Reihenfolge:
    cgroup v2 → v1 → CPU-Affinität → `cpu_count`.
    """
    # cgroup v2: "<quota> <period>" oder "max <period>" (= unbegrenzt)
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            raw = fh.read().split()
        if len(raw) == 2 and raw[0] != "max":
            quota, period = int(raw[0]), int(raw[1])
            if quota > 0 and period > 0:
                return max(1, quota // period)
    except (OSError, ValueError):
        pass

    # cgroup v1: getrennte Dateien, quota == -1 heißt unbegrenzt
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass

    # Affinität fängt Taskset/Pinning ab, aber keine Quote.
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover — nicht-POSIX
        pass

    return max(1, os.cpu_count() or 1)


def resolve_workers(n_grid: int, requested: int | None = None) -> int:
    """Wie viele Prozesse für dieses Grid — 1 heißt seriell.

    Reihenfolge: expliziter Parameter, dann ``QUANTRACE_SWEEP_WORKERS``, dann
    die sichtbaren Kerne. ``QUANTRACE_SWEEP_WORKERS=1`` ist der Rollback-Pfad,
    analog zu ``BACKTEST_WORKER=local``: eine Env-Variable, die den alten Code
    zurückholt, ohne ein Deployment.

    Mehr Prozesse als Kombinationen wären leere Worker; mehr als Kerne wäre
    Überbuchung. Beides wird gekappt.
    """
    if n_grid <= 1:
        return 1

    if requested is None:
        raw = os.environ.get("QUANTRACE_SWEEP_WORKERS", "").strip()
        if raw:
            try:
                requested = int(raw)
            except ValueError:
                log.warning("QUANTRACE_SWEEP_WORKERS=%r ist keine Zahl — ignoriert.", raw)

    if requested is None:
        requested = available_cpus()
    if requested <= 1:
        return 1

    min_grid = _MIN_GRID_BY_START_METHOD.get(_start_method(), DEFAULT_MIN_GRID_FOR_PARALLEL)
    raw_min = os.environ.get("QUANTRACE_SWEEP_MIN_GRID", "").strip()
    if raw_min:
        try:
            min_grid = int(raw_min)
        except ValueError:
            log.warning("QUANTRACE_SWEEP_MIN_GRID=%r ist keine Zahl — ignoriert.", raw_min)
    if n_grid < min_grid:
        return 1

    return max(1, min(requested, n_grid))


def _run_grid_serial(
    spec: StrategySpec,
    data: MarketData,
    config: BacktestConfig,
    grid: list[dict[str, Any]],
) -> tuple[list[SweepRun], int]:
    """Die ursprüngliche Schleife — auch der Fallback, wenn der Pool stirbt."""
    runs: list[SweepRun] = []
    failed = 0
    total = len(grid)

    for i, params in enumerate(grid, 1):
        # Neues Spec mit konkreten Params erzeugen. Merge statt Ersetzen:
        # Basis-Params außerhalb des Grids (explizite Overrides, oder `graph`
        # bei GraphStrategy) müssen den Sweep überleben — nur die Grid-Keys
        # variieren. Für Registry-Strategien ist das wertidentisch zum alten
        # Verhalten (Basis = Klassen-Defaults, die der Ctor ohnehin mergte).
        run_spec = spec.model_copy(update={"params": {**spec.params, **params}})
        run_id = _run_id(params)
        log.info("  [%d/%d] %s", i, total, run_id)

        try:
            result = run_backtest(run_spec, data, config)
            runs.append(SweepRun(params=params, result=result))
        except Exception as exc:
            log.warning("  FAIL [%d/%d] %s: %s", i, total, run_id, exc)
            failed += 1

    return runs, failed


def _collect_ordered(
    grid: list[dict[str, Any]],
    outcomes: list[tuple[int, BacktestResult | None, str | None]],
) -> tuple[list[SweepRun], int]:
    """Worker-Ergebnisse → Runs in **Grid-Reihenfolge**.

    Warum das zählt: `sweep()` sortiert am Ende nach `rank_by`, und Pythons Sort
    ist stabil. Bei Gleichstand entscheidet also die Einfügereihenfolge — käme
    die aus der Fertigstellung, hinge das Ergebnis eines Sweeps daran, welcher
    Worker zufällig zuerst fertig war. Zwei Läufe über dieselben Daten könnten
    verschiedene „beste" Parameter melden.
    """
    slots: list[SweepRun | None] = [None] * len(grid)
    failed = 0
    total = len(grid)

    for index, result, error in outcomes:
        if error is not None or result is None:
            log.warning("  FAIL [%d/%d] %s: %s", index + 1, total, _run_id(grid[index]), error)
            failed += 1
            continue
        slots[index] = SweepRun(params=grid[index], result=result)

    return [r for r in slots if r is not None], failed


def _run_grid_parallel(
    spec: StrategySpec,
    data: MarketData,
    config: BacktestConfig,
    grid: list[dict[str, Any]],
    workers: int,
) -> tuple[list[SweepRun], int]:
    """Grid über einen Prozess-Pool. Fällt bei kaputtem Pool auf seriell zurück."""
    total = len(grid)
    log.info("  Grid über %d Prozesse (%d Kombinationen)", workers, total)

    ctx = multiprocessing.get_context(_start_method())
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(spec, data, config),
        ) as pool:
            outcomes = list(pool.map(_worker_run, list(enumerate(grid))))
    except Exception as exc:  # noqa: BLE001 — BrokenProcessPool und Verwandte
        # Ein gestorbener Pool (OOM-Kill, kaputter Fork) darf den Sweep nicht
        # kosten. Seriell ist langsam, aber es liefert ein Ergebnis — und die
        # Alternative wäre, eine Stunde Rechenzeit wegzuwerfen.
        log.warning("Prozess-Pool gescheitert (%s) — Grid läuft seriell weiter.", exc)
        return _run_grid_serial(spec, data, config, grid)

    return _collect_ordered(grid, outcomes)


def _run_id(params: dict[str, Any]) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(params.items()))


def _start_method() -> str:
    """`fork` wo verfügbar, sonst `spawn`. Überschreibbar per Env.

    Gemessen auf 4 Kernen, 24–200 Kombinationen (siehe PR-Beschreibung): `spawn`
    kostet pro Worker einen vectorbt-Import **plus** eine komplette
    numba-JIT-Runde. Das sind mehrere Sekunden Fixkosten, die erst jenseits von
    ~100 Kombinationen wieder hereinkommen — bei kleinen Grids war der parallele
    Lauf schlicht langsamer als der serielle.

    `fork` erbt beides aus dem Elternprozess: kein Import, keine
    Neukompilierung. Der Preis ist, dass die Thread-Limits im Initializer nicht
    mehr greifen (BLAS liest sie beim Import, und der ist im Eltern längst
    passiert). Für diese Last ist das vertretbar — der Backtest ist
    numba-Schleifen und pandas, nicht großes BLAS. Wer auf Überbuchung stößt,
    setzt die Variablen vor dem Prozessstart oder wechselt auf `spawn`.
    """
    override = os.environ.get("QUANTRACE_SWEEP_START_METHOD", "").strip().lower()
    available = multiprocessing.get_all_start_methods()
    if override:
        if override in available:
            return override
        log.warning(
            "QUANTRACE_SWEEP_START_METHOD=%r nicht verfügbar (%s) — ignoriert.",
            override,
            ", ".join(available),
        )
    return "fork" if "fork" in available else "spawn"


def sweep(
    spec: StrategySpec,
    data: MarketData,
    config: BacktestConfig | None = None,
    rank_by: str = "sharpe",
    *,
    selection_stats: bool = True,
    max_workers: int | None = None,
) -> SweepResult:
    """Führt einen vollständigen Parameter-Sweep über spec.param_space durch.

    Args:
        spec: Strategiebeschreibung mit param_space.
        data: MarketData für den Backtest.
        config: BacktestConfig (optional, sonst Default).
        rank_by: Metrik zum Ranken (Default: sharpe).
        selection_stats: DSR/FDR über das Grid rechnen. ``False`` für
            Inner-Sweeps (z.B. pro Walk-Forward-Fold), deren Statistik
            der Aufrufer verwirft — spart pro Fold einen BH-Pass plus
            Return-Statistiken über jede Combo.
        max_workers: Prozesse für das Grid. ``None`` = automatisch
            (``QUANTRACE_SWEEP_WORKERS`` oder Kernzahl), ``1`` = seriell.
            Das Ergebnis ist von dieser Zahl **unabhängig** — sie ändert nur,
            wie lange es dauert.

    Returns:
        SweepResult mit allen Runs, sortiert nach rank_by.

    Raises:
        ValueError: Wenn param_space leer ist oder rank_by ungültig.
    """
    if not spec.param_space:
        raise ValueError(
            f"StrategySpec '{spec.strategy_id}' hat keinen param_space. "
            "Setze spec.param_space = {'param': [wert1, wert2, ...]}."
        )

    config = config or BacktestConfig()
    grid = _generate_param_grid(spec.param_space)
    total = len(grid)

    log.info(
        "Sweep für '%s': %d Kombinationen aus %s",
        spec.strategy_id,
        total,
        {k: len(v) for k, v in spec.param_space.items()},
    )

    workers = resolve_workers(total, max_workers)
    if workers > 1:
        runs, failed = _run_grid_parallel(spec, data, config, grid, workers)
    else:
        runs, failed = _run_grid_serial(spec, data, config, grid)

    # Sortieren
    reverse = rank_by not in _LOWER_IS_BETTER
    runs.sort(key=lambda r: _extract_metric(r.result, rank_by), reverse=reverse)

    best_params: dict[str, Any] = {}
    best_value = 0.0
    best_skew = best_kurt = None
    best_n_obs = best_dsr = best_psr = expected_max_annual = None
    fdr_summary: dict[str, Any] | None = None
    pbo_summary: dict[str, Any] | None = None
    bootstrap_summary: dict[str, Any] | None = None
    best_equity: list[dict[str, Any]] | None = None

    if runs:
        best_params = runs[0].params
        best_value = _extract_metric(runs[0].result, rank_by)

        # Per-run return streams + stats off the in-memory equity curves
        # (exact — the serialised JSON keeps only scalars). Order matches `runs`.
        run_rets = [_run_returns(r.result) for r in runs] if selection_stats else []
        run_stats = [_stats_from_returns(rr) for rr in run_rets]
        best_stats = run_stats[0] if run_stats else None
        if best_stats is not None:
            best_skew, best_kurt = best_stats.skew, best_stats.kurt
            best_n_obs = best_stats.n_obs

        # FDR über die *testbaren* Combos: eine Combo ohne brauchbare
        # Equity-Kurve ist kein Test (keine Daten, kein p-Wert) — sie in die
        # Familie zu zählen würde die BH-Schwellen der echten Tests künstlich
        # verschärfen, ohne dass irgendwo ein Fehlerrisiko entsteht. Sie
        # bleibt als n_untested im Summary sichtbar.
        tested = (
            [(run, st) for run, st in zip(runs, run_stats, strict=True) if st is not None]
            if run_stats
            else []
        )
        n_untested = len(run_stats) - len(tested)
        if len(tested) >= 2:
            p_values = [
                sharpe_p_value(
                    sharpe_period=st.sr_period, n_obs=st.n_obs, skew=st.skew, kurt=st.kurt
                )
                for _, st in tested
            ]
            fdr_res = benjamini_hochberg(p_values, alpha=DEFAULT_FDR_ALPHA)
            for (run, _), p, q, sig in zip(
                tested, p_values, fdr_res.q_values, fdr_res.significant, strict=True
            ):
                run.p_value = p
                run.q_value = q
                run.fdr_significant = sig
            fdr_summary = {
                "method": "benjamini_hochberg",
                "alpha": fdr_res.alpha,
                "n_tests": fdr_res.n_tests,
                "n_untested": n_untested,
                "n_significant": fdr_res.n_significant,
                "winner_p_value": runs[0].p_value,
                "winner_q_value": runs[0].q_value,
                "winner_significant": runs[0].fdr_significant,
                # BH is FDR-valid under PRDS; a param grid's combos are
                # positively correlated, so no Benjamini-Yekutieli factor.
                "dependence": "PRDS (positively correlated param grid)",
            }

        # Deflated Sharpe of the winner: V[SR] from the per-period trial
        # Sharpes (tight clusters = correlated combos = fewer effective
        # trials), Mertens SE from the winner's own T/skew/kurtosis.
        # N ist die Anzahl der Trials, aus denen V[SR] geschätzt wurde —
        # dieselbe Basis, sonst misst die Selection-Bar eine Breite, die zur
        # gemessenen Dispersion nicht passt (und der Evaluation-Agent, der
        # N = len(trial_sharpes) nutzt, käme auf eine andere Zahl).
        trial_srs_period = [st.sr_period for _, st in tested]
        if best_stats is not None and len(trial_srs_period) >= 2:
            dsr_res = deflated_sharpe_from_summary(
                observed_sharpe_period=best_stats.sr_period,
                trial_sharpes_period=trial_srs_period,
                n_obs=best_stats.n_obs,
                skew=best_stats.skew,
                kurt=best_stats.kurt,
            )
            best_dsr = dsr_res.dsr
            expected_max_annual = dsr_res.expected_max_sharpe_period * math.sqrt(
                PERIODS_PER_YEAR
            )
            if runs[0].p_value is not None:
                best_psr = 1.0 - runs[0].p_value

        # CSCV-PBO (Bailey et al. 2017): würde der IS-Winner auch OOS oben
        # ranken? Braucht die zeitalignierte T×N-Matrix aller Trial-Returns —
        # die existiert nur hier, solange die Equity-Kurven im Speicher sind.
        # Combos mit abweichender Return-Länge (degenerierte Slices) fliegen
        # raus und werden im Summary als n_excluded_trials ausgewiesen.
        winner_rets = run_rets[0] if run_rets else None
        if winner_rets is not None:
            aligned = [
                rr for rr in run_rets if rr is not None and rr.size == winner_rets.size
            ]
            if len(aligned) >= 4:
                try:
                    pbo_res = probability_of_backtest_overfitting(
                        np.column_stack(aligned)
                    )
                    pbo_summary = pbo_res.to_dict()
                    pbo_summary["n_excluded_trials"] = len(runs) - len(aligned)
                except ValueError as exc:
                    log.warning("PBO übersprungen: %s", exc)

            # Stationary-Bootstrap-Inferenz des Winners: Sharpe-KI + p-Wert
            # und die Max-Drawdown-Verteilung über dependenz-erhaltende
            # Resamples — die ehrliche Risiko-Angabe statt des einen Pfads.
            try:
                bootstrap_summary = {
                    "sharpe": bootstrap_sharpe_ci(winner_rets).to_dict(),
                    "max_drawdown": bootstrap_drawdown_distribution(
                        winner_rets
                    ).to_dict(),
                }
            except ValueError as exc:
                log.warning("Bootstrap übersprungen: %s", exc)

            # Winner-Equity-Pfad persistieren ([{date, value}]) — die Basis
            # für Portfolio-Uniqueness/Attribution auf dem Selektionsergebnis.
            eq = getattr(runs[0].result, "equity_curve", None)
            if eq is not None and hasattr(eq, "index"):
                pts = [
                    {"date": ts.date().isoformat(), "value": float(v)}
                    for ts, v in zip(eq.index, np.asarray(eq, dtype=float), strict=True)
                    if np.isfinite(v)
                ]
                if len(pts) >= 8:
                    best_equity = pts

        # Die aufgelöste Kosten-Tabelle ist für jede Combo identisch — sie
        # 200× im JSON zu persistieren bläht Sweep-Artefakte nur auf. Der
        # Winner (Rang 0) behält sie als Dokumentation der Annahmen.
        for run in runs[1:]:
            if run.result.config.symbol_costs is not None:
                run.result.config = run.result.config.model_copy(
                    update={"symbol_costs": None}
                )

    log.info(
        "Sweep fertig: %d/%d erfolgreich. Beste %s = %.4f mit %s (DSR=%s, PBO=%s)",
        len(runs),
        total,
        rank_by,
        best_value,
        best_params,
        f"{best_dsr:.3f}" if best_dsr is not None else "n/a",
        f"{pbo_summary['pbo']:.3f}" if pbo_summary else "n/a",
    )

    return SweepResult(
        strategy_id=spec.strategy_id,
        periods_per_year=float(data.periods_per_year),
        param_space=spec.param_space,
        rank_by=rank_by,
        total_combinations=total,
        completed=len(runs),
        failed=failed,
        runs=runs,
        best_params=best_params,
        best_metric_value=best_value,
        best_skew=best_skew,
        best_kurt=best_kurt,
        best_n_obs=best_n_obs,
        best_dsr=best_dsr,
        best_psr=best_psr,
        expected_max_sharpe_annual=expected_max_annual,
        fdr=fdr_summary,
        pbo=pbo_summary,
        best_bootstrap=bootstrap_summary,
        best_equity=best_equity,
    )
