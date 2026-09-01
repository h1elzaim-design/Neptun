"""Backtest-Runner auf vectorbt. Konsistente Kosten- und Slippage-Annahmen.

Der Runner kennt keine Strategielogik — er bekommt MarketData + Strategy +
BacktestConfig und liefert ein BacktestResult. Dadurch sind alle Strategien
unter identischen Bedingungen vergleichbar.

Kapitalmodell (ADR-003):
- ``capital_model="shared"`` (Default): EIN Konto über alle Symbole. Das
  Kapital wird gleichgewichtet über die gerade aktiven Positionen verteilt
  (k aktive → je size/k des Kontowerts), umgeschichtet nur bei Signalwechsel,
  dazwischen driften die Gewichte. k=0 → 100% Cash.
- ``capital_model="independent"``: Rollback — jede Spalte ist ihr eigenes
  Konto mit vollem init_cash, Equity = Mittelwert der Sleeves. Keine
  Kapazitäts-Restriktion; nur für Vergleiche mit Alt-Ergebnissen.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quantrace.data_agent import close_prices
from quantrace.models import (
    BacktestConfig,
    BacktestResult,
    MarketData,
    StrategySpec,
    TradeMetrics,
)
from quantrace.stats.capacity import turnover_from_orders
from quantrace.strategy import Strategy, load_strategy

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

#: Finite stand-in for an infinite profit factor (zero losing trades).
_PROFIT_FACTOR_CAP = 1e6


def run_backtest(
    spec: StrategySpec,
    data: MarketData,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Führt den Backtest aus und gibt das vollständige Ergebnis zurück."""
    config = config or BacktestConfig()
    strategy = load_strategy(spec)
    return _execute(strategy, spec.strategy_id, data, config)


def run_inline(
    strategy_id: str,
    strategy: Strategy,
    data: MarketData,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Variante ohne StrategySpec — bequem für Notebook-Forschung."""
    return _execute(strategy, strategy_id, data, config or BacktestConfig())


def _resolve_annualization(config: BacktestConfig, data: MarketData) -> float:
    """Perioden pro Jahr für diesen Lauf — aus den Daten, nicht aus dem Default.

    `BacktestConfig.annualization` bleibt als **Override** bestehen: wer sie
    explizit setzt, meint sie auch (Tests, Sonderfälle). Wer sie nicht anfasst,
    bekommt den Kalender des Universums.

    Warum die Unterscheidung über `model_fields_set` und nicht über einen
    ``None``-Default: der Feld-Default 252 ist seit jeher öffentlich, und ein
    Wechsel auf ``None`` würde jeden Aufrufer treffen, der ihn liest. So bleibt
    das Feld, was es war, und bekommt nur eine ehrlichere Herkunft (#184).
    """
    if "annualization" in config.model_fields_set:
        return float(config.annualization)
    return float(data.periods_per_year)


def _execute(
    strategy: Strategy,
    strategy_id: str,
    data: MarketData,
    config: BacktestConfig,
) -> BacktestResult:
    try:
        import vectorbt as vbt
    except ImportError as e:
        raise ImportError("vectorbt nicht installiert. `pip install vectorbt`.") from e

    close = close_prices(data)
    entries, exits = strategy.generate_signals(data)
    entries, exits = _lag_signals(entries, exits, config.execution_lag)
    # NACH dem Lag: der Zwangs-Exit ist keine Strategie-Entscheidung, die
    # verzögert gehandelt würde, sondern eine Ausführungsschranke.
    close, entries, exits = _close_untradable(
        close, entries, exits, getattr(data, "tradable", None)
    )

    fees, slippage, config = _cost_inputs(close, config, data)

    multi_asset = isinstance(close, pd.DataFrame) and close.shape[1] > 1
    if config.capital_model == "shared" and multi_asset:
        pf = _shared_portfolio(vbt, close, entries, exits, fees, slippage, config)
    else:
        # Einzel-Symbol (shared == independent per Konstruktion) oder explizites
        # capital_model="independent"-Rollback: jede Spalte ist ihr eigenes Konto
        # mit vollem init_cash; die Equity wird unten per Mittelwert aggregiert.
        # cash_sharing=False macht diese Semantik EXPLIZIT — der alte Aufruf
        # (cash_sharing=True ohne group_by) war vectorbt-versionsabhängig und
        # teilte auf neueren Versionen das Kapital in Spaltenreihenfolge zu
        # (erstes Symbol ~95%, Rest Krümel). Siehe ADR-003.
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=config.cash,
            fees=fees,
            slippage=slippage,
            size=config.size,
            size_type=config.size_type,
            freq=config.freq,
            cash_sharing=False,
        )

    annualization = _resolve_annualization(config, data)

    equity = _aggregate_equity(pf, config.cash)
    metrics = _compute_metrics(equity, annualization)
    trades = _trade_metrics(pf)
    turnover = _turnover_annual(pf, equity, annualization)

    return BacktestResult(
        strategy_id=strategy_id,
        data_hash=data.content_hash,
        config=config,
        periods_per_year=annualization,
        start=_to_date(close.index[0]),
        end=_to_date(close.index[-1]),
        # `start`/`end` oben sind längst die tatsächlichen Grenzen — was fehlte,
        # war das **angeforderte** Fenster daneben. Ohne beides nebeneinander
        # kann ein Leser den Unterschied nicht bemerken (#307).
        coverage=data.coverage,
        total_return=metrics["total_return"],
        cagr=metrics["cagr"],
        sharpe=metrics["sharpe"],
        sortino=metrics["sortino"],
        calmar=metrics["calmar"],
        max_drawdown=metrics["max_drawdown"],
        avg_drawdown=metrics["avg_drawdown"],
        ulcer_index=metrics["ulcer_index"],
        trades=trades,
        turnover_annual=turnover,
        equity_curve=equity,
    )


def _cost_inputs(
    close: pd.DataFrame | pd.Series,
    config: BacktestConfig,
    data: MarketData | None = None,
):
    """Fees/Slippage für vectorbt — skalar (flat) oder pro Spalte (per Klasse).

    ``cost_model="flat"`` reproduziert das bisherige Verhalten exakt: ein
    Skalar für alle Symbole. ``"per_asset_class"`` löst die Profile aus
    `config/costs.yaml` auf (bzw. nutzt vorbelegte ``config.symbol_costs`` als
    explizites Override) und baut per-Spalte-Arrays; die effektive Slippage
    pro Seite ist ``slippage_bps + spread_bps/2``. Die aufgelöste Tabelle wird
    an die zurückgegebene Config gehängt, damit das persistierte Ergebnis
    seine Kosten-Annahmen dokumentiert.

    ``data.cost_class`` — gesetzt von konstruierten Universen — dient als
    Rückfall für Symbole ohne eigenen Eintrag. Ohne ihn bekäme ein
    Regel-Universum aus hunderten Small Caps die Kosten von SPY.
    """
    if config.cost_model != "per_asset_class":
        return config.fees_bps / 10_000.0, config.slippage_bps / 10_000.0, config

    symbols = list(close.columns) if isinstance(close, pd.DataFrame) else [close.name or "?"]

    table = config.symbol_costs
    if table is None:
        from quantrace.costs import resolve_symbol_costs

        table = resolve_symbol_costs(
            [str(s) for s in symbols],
            fallback_class=data.cost_class if data is not None else None,
        )
    else:
        missing = [s for s in symbols if str(s) not in table]
        if missing:
            raise ValueError(
                f"cost_model=per_asset_class: symbol_costs vorbelegt, aber ohne {missing}"
            )

    fees = np.array([table[str(s)].fees_bps for s in symbols], dtype=float) / 10_000.0
    slippage = (
        np.array([table[str(s)].effective_slippage_bps for s in symbols], dtype=float)
        / 10_000.0
    )
    if isinstance(close, pd.Series):
        # Einzel-Symbol: Skalare statt 1-Element-Arrays (vectorbt-Broadcasting).
        fees, slippage = float(fees[0]), float(slippage[0])

    config = config.model_copy(update={"symbol_costs": dict(table)})
    return fees, slippage, config


def _lag_signals(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    lag: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delay execution by `lag` bars to remove same-bar look-ahead.

    A strategy decides on bar *t* using prices observed up to and including
    `close[t]`. You cannot also *trade* at `close[t]` — that price is only
    known once the bar has closed. Shifting the boolean signals forward by one
    bar makes vectorbt fill at `close[t+1]`, so every order uses strictly
    past-or-present information. This is applied at the runner level (not per
    strategy) so the execution model is identical across the whole catalogue.

    `lag <= 0` is a deliberate opt-out for signals already lagged upstream.
    """
    if lag <= 0:
        return entries, exits
    return (
        entries.shift(lag, fill_value=False),
        exits.shift(lag, fill_value=False),
    )


def _close_untradable(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    tradable: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """**Was keinen Kurs hat, kann nicht gehalten werden — aber Lücke ≠ Ende.**

    Ohne diese Regel überlebt eine Position das Papier. Nachgemessen an dem
    Fall, für den der Lake gebaut wurde: ein Symbol handelt ab Bar 20 nicht
    mehr, die Strategie hat keinen Exit gefeuert, weil ihre Indikatoren auf
    ``NaN`` laufen und jeder Vergleich damit ``False`` ergibt. ``_held_mask``
    liest nur Signale, ``from_orders`` kann bei ``NaN`` nicht verkaufen.
    Ergebnis ohne Korrektur: 47.500 $ eingefroren zum letzten bekannten Kurs,
    das zweite Papier dauerhaft auf 47,5 % des Buches gedeckelt statt auf 95 %.

    **Zwei Ursachen, die im Rahmen gleich aussehen und es nicht sind:**

    * **Kein Kurs beobachtet.** Handelsaussetzung, fehlende Partition, ein Tag,
      an dem dieses Papier nicht handelte, während andere es taten — im
      Bulk-Lake der Normalfall. Wer hier verkauft, verkauft wegen einer
      Datenlücke. Eine frühere Fassung tat genau das: eine einzelne fehlende
      Bar liquidierte die Position, und die Strategie stieg oft nie wieder ein,
      weil ihr Entry-Signal ein Crossover war. Die Qualitätsprüfung fängt das
      nicht ab — sie meldet ``NaN`` als *Warnung*, nicht als Fehler.
      **Also: durchhalten und den letzten Kurs fortschreiben.**
    * **Kein Kurs mehr, nie wieder.** Delisting, Insolvenz. Der Lauf endet für
      dieses Papier. **Also: verkaufen**, zum letzten beobachteten Schluss.

    Unterschieden wird an genau einem Merkmal: liegt *irgendwann später* noch
    ein Kurs? Das ist Buchhaltung über ein abgeschlossenes Ereignis, keine
    Handelsentscheidung — die Strategie hat den Zeitpunkt nicht gewählt.

    ``tradable`` ist die dritte Ursache und kommt nicht aus den Kursen, sondern
    aus dem Universum: ein Papier scheidet bei der Rekonstitution aus (#255).
    Das ist kein Datenproblem und wird deshalb auch nicht aus ``NaN``
    erschlossen, sondern ausdrücklich mitgegeben — sonst wäre ein Ausscheiden
    von einer Handelsaussetzung nicht zu unterscheiden, und je nach Rateweg
    würde entweder durchgehalten (falsch) oder bei jeder Lücke verkauft (auch
    falsch).
    """
    handelbar = close.notna()
    voll = bool(handelbar.all().all())
    if voll and tradable is None:
        return close, entries, exits

    raus = pd.DataFrame(False, index=close.index, columns=close.columns)

    if not voll:
        # Gibt es an oder nach diesem Bar noch einen Kurs? Rückwärts-cummax
        # über die Handelbarkeit. Nur wo das False ist, endet der Lauf.
        spaeter = handelbar[::-1].cummax()[::-1].astype(bool)
        endgueltig = ~spaeter & handelbar.shift(fill_value=False)
        raus |= endgueltig
        close = close.ffill()

    if tradable is not None:
        # Ausgeschieden: gestern Mitglied, heute nicht.
        mask = tradable.reindex(index=close.index, columns=close.columns).fillna(False)
        raus |= mask.shift(fill_value=False).astype(bool) & ~mask.astype(bool)
        handelbar &= mask.astype(bool)

    return close, entries & handelbar, exits | raus


def _held_mask(entries: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:
    """In-Position-Zustand pro Spalte aus Entry-/Exit-Events (long-only).

    Ein Entry setzt den Zustand auf "held", ein Exit auf "flat"; dazwischen
    wird der letzte Zustand fortgeschrieben (ffill), Start ist flat. Fallen
    Entry und Exit auf denselben Bar, gewinnt der **Exit** — im Zweifel flat
    ist die konservative Auflösung und deterministisch dokumentiert (das
    alte from_signals-Verhalten hing an vectorbt-Defaults).
    """
    sig = np.where(exits.to_numpy(), 0.0, np.where(entries.to_numpy(), 1.0, np.nan))
    state = pd.DataFrame(sig, index=entries.index, columns=entries.columns)
    return state.ffill().fillna(0.0).astype(bool)


def _shared_target_sizes(held: pd.DataFrame, invested_fraction: float) -> pd.DataFrame:
    """Zielgewichts-Matrix für den Shared-Pfad — NaN heißt "keine Order".

    Auf Bars, an denen sich die Mitgliedschaft ändert (irgendein Symbol kommt
    dazu oder fliegt raus), bekommt jede aktive Position das Ziel
    ``invested_fraction / k`` (k = Anzahl aktiver) und jede gerade
    ausscheidende das Ziel 0 (Vollverkauf). Alle übrigen Zellen sind NaN —
    vectorbt platziert dort keine Order, die Gewichte driften mit dem Markt
    bis zum nächsten Signalwechsel. k=0 heißt: alles verkaufen, 100% Cash.
    """
    k = held.sum(axis=1)
    weights = (
        held.astype(float).div(k.where(k > 0, other=np.nan), axis=0).fillna(0.0)
        * invested_fraction
    )
    changed = held.ne(held.shift(fill_value=False)).any(axis=1)
    # Orders nur für Spalten, die jetzt aktiv sind (neues 1/k-Ziel) oder gerade
    # aussteigen (Ziel 0) — nie gehaltene Spalten bleiben ohne Order (NaN).
    involved = held | held.shift(fill_value=False)
    order_mask = involved & changed.to_numpy()[:, None]
    return weights.where(order_mask, other=np.nan)


def _shared_portfolio(
    vbt,
    close: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    fees,
    slippage,
    config: BacktestConfig,
):
    """EIN Konto über alle Symbole: Equal-Weight über die aktiven Positionen.

    ``group_by=True + cash_sharing=True`` bündelt alle Spalten in eine Gruppe
    mit einem Cash-Pool; ``size_type="targetpercent"`` ordert auf das Ziel
    "x% des Kontowerts" (nicht des freien Cashs — die Zuteilung ist dadurch
    unabhängig von der Spaltenreihenfolge); ``call_seq="auto"`` führt
    Verkäufe vor Käufen aus, damit Umschichtungen aus dem freigewordenen
    Cash finanziert werden. Verifiziert gegen vectorbt 1.1.0.
    """
    held = _held_mask(entries, exits)
    size = _shared_target_sizes(held, config.size)
    return vbt.Portfolio.from_orders(
        close=close,
        size=size,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        call_seq="auto",
        init_cash=config.cash,
        fees=fees,
        slippage=slippage,
        freq=config.freq,
    )


def _aggregate_equity(pf, init_cash: float) -> pd.Series:
    val = pf.value()
    if isinstance(val, pd.DataFrame):
        # Nur noch im independent-Rollback erreichbar: jede Spalte ist ein
        # eigenes Konto mit vollem init_cash, der Mittelwert ist die
        # gleichgewichtete Sleeve-Aggregation (dokumentierte Alt-Semantik).
        # Der Shared-Pfad liefert bereits eine einzelne Konto-Serie.
        val = val.mean(axis=1)
    return val.astype(float)


def _compute_metrics(equity: pd.Series, ann: float) -> dict[str, float]:
    ret = equity.pct_change().dropna()
    if ret.empty or equity.iloc[0] == 0:
        return dict.fromkeys(
            [
                "total_return",
                "cagr",
                "sharpe",
                "sortino",
                "calmar",
                "max_drawdown",
                "avg_drawdown",
                "ulcer_index",
            ],
            0.0,
        )

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)

    # Sample std (ddof=1) — academic Sharpe/Sortino definition and consistent
    # with quantrace.stats.sharpe. ddof=0 would understate vol and inflate the
    # ratios. std of a single observation is NaN, which the `> 0` guard catches.
    std = ret.std(ddof=1)
    sharpe = float(np.sqrt(ann) * ret.mean() / std) if std > 0 else 0.0

    downside = ret[ret < 0].std(ddof=1)
    sortino = float(np.sqrt(ann) * ret.mean() / downside) if downside > 0 else 0.0

    running_max = equity.cummax()
    dd = equity / running_max - 1
    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    ulcer = float(np.sqrt((dd**2).mean()))
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

    metrics = {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd,
        "ulcer_index": ulcer,
    }
    # Never let NaN/inf leak into results — they break JSON serialisation and
    # silently corrupt sweep ranking (NaN comparisons are always False).
    return {k: (v if np.isfinite(v) else 0.0) for k, v in metrics.items()}


def _order_notionals(orders: pd.DataFrame) -> list[float]:
    """|Größe × Preis| je Order aus vectorbts ``records_readable``.

    Die Spaltennamen sind versionsabhängig kapitalisiert ("Size"/"size"),
    deshalb wird case-insensitiv aufgelöst. Fehlt eine der beiden Spalten,
    gibt es keinen Turnover — dann lieber gar keine Zahl als eine falsche.
    """
    if orders is None or orders.empty:
        return []
    lookup = {str(c).strip().lower(): c for c in orders.columns}
    size_col, price_col = lookup.get("size"), lookup.get("price")
    if size_col is None or price_col is None:
        return []
    notional = (
        orders[size_col].astype(float).abs() * orders[price_col].astype(float).abs()
    )
    return [float(x) for x in notional.to_numpy() if np.isfinite(x)]


def _turnover_annual(pf, equity: pd.Series, annualization: float) -> float | None:
    """Annualisierter Turnover aus den Order-Records; None wenn nicht ableitbar.

    Der Haupttreiber der realisierten Kosten — und ohne ihn ist jede
    Kapazitätsaussage geraten. Bewusst weich verdrahtet: Order-Records sind
    vectorbt-versionsabhängig, und ein fehlender Turnover darf niemals einen
    sonst gültigen Backtest scheitern lassen (die Analytik fällt dann auf die
    ausgewiesene Schätzung zurück).
    """
    try:
        orders = pf.orders.records_readable
    except Exception:  # pragma: no cover — vectorbt versionsabhängig
        return None

    notionals = _order_notionals(orders)
    if not notionals or equity.empty:
        return None

    try:
        profile = turnover_from_orders(
            notionals, equity.to_numpy(dtype=float), periods_per_year=float(annualization)
        )
    except ValueError:  # degeneriertes Konto (NAV ≤ 0) — keine sinnvolle Zahl
        return None
    return profile.annual_turnover


def _trade_metrics(pf) -> TradeMetrics:
    try:
        trades = pf.trades.records_readable
    except Exception:  # pragma: no cover — vectorbt versionsabhängig
        trades = pd.DataFrame()

    if trades.empty or "Return" not in trades.columns:
        return TradeMetrics(
            n_trades=0,
            win_rate=0.0,
            avg_trade_return=0.0,
            avg_winner=0.0,
            avg_loser=0.0,
            profit_factor=0.0,
            expectancy=0.0,
        )

    rets = trades["Return"].astype(float)
    winners = rets[rets > 0]
    losers = rets[rets < 0]
    gross_win = winners.sum()
    gross_loss = -losers.sum()
    # Cap at a large finite value instead of inf: a strategy with zero losing
    # trades would otherwise serialise to invalid JSON (`Infinity`) and poison
    # any downstream ranking or averaging.
    if gross_loss > 0:
        profit_factor = min(float(gross_win / gross_loss), _PROFIT_FACTOR_CAP)
    elif gross_win > 0:
        profit_factor = _PROFIT_FACTOR_CAP
    else:
        profit_factor = 0.0

    return TradeMetrics(
        n_trades=int(len(rets)),
        win_rate=float(len(winners) / len(rets)),
        avg_trade_return=float(rets.mean()),
        avg_winner=float(winners.mean()) if not winners.empty else 0.0,
        avg_loser=float(losers.mean()) if not losers.empty else 0.0,
        profit_factor=profit_factor,
        expectancy=float(rets.mean()),
    )


def _to_date(ts) -> date:
    return pd.Timestamp(ts).date()
