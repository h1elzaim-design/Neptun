"""Data Agent — lädt, normalisiert, versioniert und qualitätssichert Marktdaten.

High-End-Datalake (provider='tiingo'):
  1. Roh-OHLCV + Corporate Actions pro Symbol partitioniert (storage).
  2. Inkrementelles Fetch — nur fehlende Datumsbereiche werden nachgeladen.
  3. DuckDB liest den angeforderten Slice (lokal wie R2).
  4. Data-Quality-Gate (quality) prüft auf stille Fehler.
  5. Adjustierung beim Lesen (adjust) — point-in-time korrekt, nie stale.

Legacy-Pfad (yfinance/fmp/… via openbb): kombinierter Universum-Cache, adjustiert
beim Fetch. Optionales Extra `.[data]`.

Storage entscheidet lokal vs. R2 allein über Env (QUANTRACE_DATA_LAKE).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from quantrace import adjust, quality, storage
from quantrace.calendars import DEFAULT_CALENDAR, calendar_for_class, validate_universe_calendar
from quantrace.data_providers import bootstrap_credentials, default_provider
from quantrace.models import MarketData, Timeframe
from quantrace.providers import tiingo, tiingo_crypto

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = storage.DEFAULT_LOCAL_DIR
_OHLCV = ["open", "high", "low", "close", "volume"]

# Max parallel workers for R2/storage I/O.  32 is the boto3 connection-pool
# default; going higher wastes resources without benefit on R2.
_MAX_IO_WORKERS = 32


def load_universe(
    universe: str,
    symbols: list[str],
    start: date,
    end: date,
    timeframe: Timeframe = Timeframe.DAILY,
    provider: str | None = None,
    adjusted: bool = True,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    calendar: str | None = None,
) -> MarketData:
    """Lädt OHLCV für eine Symbolliste und gibt ein normalisiertes MarketData zurück.

    Frame-Layout:
        MultiIndex columns: (symbol, field) mit field ∈ {open, high, low, close, volume}
        DatetimeIndex tz-naiv, sortiert.
    """
    # Ein Universum, ein Kalender (#184) — vor jedem I/O, damit ein gemischtes
    # Universum nicht erst Daten zieht und dann scheitert.
    validate_universe_calendar(symbols, calendar, universe=universe)

    provider = provider or default_provider()
    if provider == "tiingo":
        return _load_via_lake(
            universe, symbols, start, end, timeframe, adjusted, force_refresh, calendar
        )
    return _load_via_openbb_cache(
        universe, symbols, start, end, timeframe, provider, adjusted, cache_dir,
        force_refresh, calendar,
    )


# ---------------------------------------------------------------------------
# Lake-Pfad (Tiingo): partitioniert, inkrementell, adjust-on-read
# ---------------------------------------------------------------------------

class _SymbolPlan(NamedTuple):
    symbol: str
    stored: pd.DataFrame | None          # already-cached raw data (may be None)
    cov: tuple[date, date] | None        # existing coverage window (may be None)
    need: list[tuple[date, date]]        # gaps that must be fetched from Tiingo


def _plan_symbol(symbol: str, start: date, end: date, force_refresh: bool) -> _SymbolPlan:
    """Read coverage + cached data for one symbol and decide what to fetch.

    This is the only function that does R2 I/O per symbol; it is called in a
    thread pool so all symbols are checked in parallel.
    """
    cov = None if force_refresh else storage.read_coverage(symbol)
    stored = None if force_refresh else storage.read_symbol_raw(symbol)

    if cov is None:
        need: list[tuple[date, date]] = [(start, end)]
    else:
        cov_start, cov_end = cov
        need = []
        if start < cov_start:
            need.append((start, cov_start - timedelta(days=1)))
        if end > cov_end:
            need.append((cov_end + timedelta(days=1), end))

    return _SymbolPlan(symbol=symbol, stored=stored, cov=cov, need=need)


def _apply_fetched(
    plan: _SymbolPlan,
    fetched_frames: dict[str, pd.DataFrame],
    start: date,
    end: date,
) -> None:
    """Merge freshly-fetched data with the cached partition and persist.

    Called in a thread pool after all Tiingo fetches complete.
    """
    sym = plan.symbol
    new_parts = [fetched_frames[sym]] if sym in fetched_frames and not fetched_frames[sym].empty else []

    parts = (
        ([plan.stored] if plan.stored is not None and not plan.stored.empty else [])
        + new_parts
    )
    if not parts:
        if plan.stored is None or plan.stored.empty:
            raise RuntimeError(f"Tiingo lieferte keine Daten für {sym} ({start}..{end}).")
        merged = plan.stored
    else:
        merged = pd.concat(parts).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        storage.write_symbol_raw(sym, merged)
        log.info(
            "Lake-Partition aktualisiert: %s (%d Zeilen) → %s",
            sym,
            len(merged),
            storage.symbol_partition(sym),
        )

    new_start = start if plan.cov is None else min(start, plan.cov[0])
    new_end = end if plan.cov is None else max(end, plan.cov[1])
    storage.write_coverage(sym, new_start, new_end)


def _provider_for(calendar_name: str):
    """Kalender → Provider-Modul für den Roh-Fetch (#184, Schritt C).

    Nachschlag zur Laufzeit statt einer Modul-Konstante: Tests ersetzen
    ``data_agent.tiingo``, und eine beim Import gebundene Referenz würde daran
    vorbeilaufen.
    """
    return tiingo_crypto if calendar_name == "crypto_24_7" else tiingo


def _split_by_calendar(
    fetch_ranges: dict[str, tuple[date, date]],
) -> dict[str, dict[str, tuple[date, date]]]:
    """Fetch-Bereiche nach Kalender aufteilen — **pro Symbol**, nicht pro Universum.

    Der Kalender-Guardrail (`validate_universe_calendar`) sorgt dafür, dass ein
    *Universum* homogen ist. Der Lake ist aber breiter als ein Universum: der
    Daily-Plan lädt die Symbole des Buchs, und das Buch darf legitim einen
    Equity- und einen Crypto-Sleeve enthalten. Die Weiche gehört deshalb hier
    ans Symbol, nicht an den Aufrufer.

    Zugeordnet wird über die Kostenklasse aus `config/costs.yaml` — dieselbe
    Quelle, die der Guardrail benutzt. Ein unklassifiziertes Symbol landet auf
    ``us_equity``; das ist der bestehende Zustand und keine neue Annahme.
    """
    from quantrace.costs import resolve_symbol_costs

    resolved = resolve_symbol_costs(sorted(fetch_ranges))
    out: dict[str, dict[str, tuple[date, date]]] = {}
    for sym, window in fetch_ranges.items():
        profile = resolved.get(sym)
        cal = calendar_for_class(profile.asset_class) if profile else DEFAULT_CALENDAR
        out.setdefault(cal, {})[sym] = window
    return out


def refresh_symbols(
    symbols: list[str],
    start: date,
    end: date,
    *,
    force_refresh: bool = False,
) -> list[str]:
    """Den Lake für ``symbols`` über ``start..end`` aktuell machen.

    Liest pro Symbol die vorhandene Abdeckung, holt **nur die Lücke** von
    Tiingo und schreibt Partition + Coverage zurück. Gibt die Symbole zurück,
    für die tatsächlich etwas geholt wurde (leer = alles war schon da).

    Welcher Tiingo-Endpunkt gefragt wird, entscheidet ``_split_by_calendar``
    pro Symbol (#184): Crypto-Paare gehen an den Crypto-Endpunkt, alles andere
    an EOD. Der Lake dahinter ist derselbe — gleiche Partitionen, gleiches
    Spaltenlayout, gleiche Adjustierung beim Lesen.

    Aus ``_load_via_lake`` herausgezogen, weil es zwei Aufrufer gibt: den
    Backtest-Pfad, der die Daten anschließend selbst liest, und den
    Daily-Plan-Job, der nur sicherstellen will, dass die Kurse aktuell sind,
    bevor er plant (#192). Zwei Kopien wären zwei Verhalten — und ausgerechnet
    beim Nachladen von Kursen darf es nur eines geben.

    Fehlt ein einzelnes Symbol bei Tiingo, überspringt
    ``fetch_universe_raw_ranges`` es mit einer Warnung; der Lake behält dann
    seinen alten Stand für dieses Symbol. Der Aufrufer entscheidet, ob ihn das
    stört — der Daily-Plan macht daraus einen Alert.
    """
    if not symbols:
        return []

    workers = min(_MAX_IO_WORKERS, len(symbols))

    # ── Step 1: Read coverage for all symbols in parallel ──────────────────
    plans: list[_SymbolPlan] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_plan_symbol, sym, start, end, force_refresh): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            plans.append(fut.result())

    # ── Step 2: Batch-fetch only the symbols/ranges that are missing ────────
    plans_needing_fetch = [p for p in plans if p.need]
    if not plans_needing_fetch:
        return []

    # Per symbol: fetch only the actual gap (bounding box of its missing
    # ranges), never the full start..end.  A symbol with an existing
    # coverage window in the middle can have both a before- and an
    # after-gap; we request the span between them (a small re-fetch of the
    # covered middle) rather than adding multi-range request logic — but the
    # common incremental case (extend forward or backward) fetches exactly
    # the one gap.  fetch_universe_raw_ranges shares a single httpx.Client.
    fetch_ranges = {
        p.symbol: (min(g[0] for g in p.need), max(g[1] for g in p.need))
        for p in plans_needing_fetch
    }
    log.info(
        "Tiingo-Fetch für %d/%d Symbole (gaps: %s)",
        len(fetch_ranges),
        len(symbols),
        {s: (str(a), str(b)) for s, (a, b) in fetch_ranges.items()},
    )
    fetched_frames: dict[str, pd.DataFrame] = {}
    for calendar_name, subset in _split_by_calendar(fetch_ranges).items():
        provider = _provider_for(calendar_name)
        log.info("  → %s über %s (%d Symbole)", calendar_name, provider.__name__, len(subset))
        fetched_frames.update(provider.fetch_universe_raw_ranges(subset))

    # ── Step 3: Persist updated partitions + coverage in parallel ──────────
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures_write = {
            pool.submit(_apply_fetched, p, fetched_frames, start, end): p.symbol
            for p in plans_needing_fetch
        }
        for fut in as_completed(futures_write):
            fut.result()  # re-raise any exception from the worker

    return sorted(fetched_frames)


def _load_via_lake(
    universe: str,
    symbols: list[str],
    start: date,
    end: date,
    timeframe: Timeframe,
    adjusted: bool,
    force_refresh: bool,
    calendar: str | None = None,
) -> MarketData:
    if timeframe is not Timeframe.DAILY:
        raise ValueError("Lake/Tiingo unterstützt aktuell nur Daily-Daten.")

    refresh_symbols(symbols, start, end, force_refresh=force_refresh)

    # ── Step 4: DuckDB bulk-read from lake ──────────────────────────────────
    long_df = storage.read_symbols(symbols, start, end)
    if long_df.empty:
        raise RuntimeError(f"Keine Daten im Lake für {symbols} {start}..{end}")

    raw_frames: dict[str, pd.DataFrame] = {}
    for sym, group in long_df.groupby("symbol"):
        f = group.drop(columns=["symbol"]).copy()
        f["date"] = pd.to_datetime(f["date"])
        raw_frames[str(sym)] = f.set_index("date").sort_index()

    # Quality-Gate auf den Roh-OHLCV — Fehler (z.B. high<low, Duplikate) stoppen.
    report = quality.check_universe(
        {s: f[[c for c in _OHLCV if c in f.columns]] for s, f in raw_frames.items()}
    )
    report.log()
    if not report.ok:
        errs = [(i.symbol, i.kind, i.detail) for i in report.issues if i.severity == "error"]
        raise ValueError(f"Data-Quality-Fehler im Universum {universe}: {errs}")

    per_out: dict[str, pd.DataFrame] = {}
    for sym, f in raw_frames.items():
        per_out[sym] = adjust.adjust_ohlcv(f) if adjusted else f[_OHLCV].copy()

    combined = pd.concat(per_out, axis=1)
    combined.columns.names = ["symbol", "field"]
    combined = combined.sort_index().dropna(how="all")

    return MarketData(
        universe=universe,
        symbols=sorted(per_out),
        timeframe=timeframe,
        start=start,
        end=end,
        provider="tiingo",
        adjusted=adjusted,
        calendar=calendar or DEFAULT_CALENDAR,
        frame=combined,
    )


# ---------------------------------------------------------------------------
# Legacy-Pfad (openbb): kombinierter Universum-Cache
# ---------------------------------------------------------------------------

def _cache_filename(
    universe: str, timeframe: Timeframe, start: date, end: date, provider: str
) -> str:
    return f"{universe}__{timeframe.value}__{start}__{end}__{provider}.parquet"


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    flat = frame.copy()
    flat.columns = [f"{sym}|{field}" for sym, field in flat.columns]
    return flat


def _unflatten(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        frame.columns = pd.MultiIndex.from_tuples(
            [tuple(c.split("|", 1)) for c in frame.columns]
        )
    frame.columns.names = ["symbol", "field"]
    frame.index = pd.to_datetime(frame.index)
    return frame


def _load_via_openbb_cache(
    universe: str,
    symbols: list[str],
    start: date,
    end: date,
    timeframe: Timeframe,
    provider: str,
    adjusted: bool,
    cache_dir: Path | None,
    force_refresh: bool,
    calendar: str | None = None,
) -> MarketData:
    filename = _cache_filename(universe, timeframe, start, end, provider)
    uri = str(cache_dir / filename) if cache_dir is not None else storage.cache_path(filename)

    if not force_refresh and storage.exists(uri):
        log.info("Lade aus Cache: %s", uri)
        frame = _unflatten(storage.read_parquet(uri))
    else:
        frame = _fetch_via_openbb(symbols, start, end, timeframe, provider, adjusted)
        storage.write_parquet(_flatten(frame), uri)
        log.info("Cache geschrieben: %s (%d Zeilen)", uri, len(frame))

    return MarketData(
        universe=universe,
        symbols=sorted(set(frame.columns.get_level_values("symbol"))),
        timeframe=timeframe,
        start=start,
        end=end,
        provider=provider,
        adjusted=adjusted,
        calendar=calendar or DEFAULT_CALENDAR,
        frame=frame,
    )


def _fetch_via_openbb(
    symbols: list[str],
    start: date,
    end: date,
    timeframe: Timeframe,
    provider: str,
    adjusted: bool,
) -> pd.DataFrame:
    """OpenBB equity.price.historical pro Symbol, dann zu MultiIndex-Frame mergen."""
    try:
        from openbb import obb
    except ImportError as e:
        raise ImportError(
            f"Provider {provider!r} braucht openbb (nicht installiert). "
            "Nutze provider='tiingo' oder installiere `pip install -e \".[data]\"`."
        ) from e

    bootstrap_credentials()
    interval = {Timeframe.DAILY: "1d", Timeframe.HOURLY: "1h", Timeframe.MINUTE: "1m"}[timeframe]

    per_symbol: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            obj = obb.equity.price.historical(
                symbol=sym,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                interval=interval,
                provider=provider,
                adjusted=adjusted,
            )
            df = obj.to_df() if hasattr(obj, "to_df") else pd.DataFrame(obj.results)
        except Exception as exc:
            log.warning("Symbol %s konnte nicht geladen werden: %s", sym, exc)
            continue

        df = _normalize_ohlcv(df)
        if df.empty:
            continue
        per_symbol[sym] = df

    if not per_symbol:
        raise RuntimeError(
            f"Keine Daten geladen für {len(symbols)} Symbole via provider={provider}"
        )

    combined = pd.concat(per_symbol, axis=1)
    combined.columns.names = ["symbol", "field"]
    combined = combined.sort_index().dropna(how="all")
    return combined


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Vereinheitlicht Spaltennamen und Index. Tolerant gegenüber Provider-Quirks."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ("date", "Date", "timestamp", "Timestamp"):
            if cand in df.columns:
                df = df.set_index(cand)
                break
        df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    df.columns = [str(c).lower() for c in df.columns]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].astype(float)


def close_prices(md: MarketData) -> pd.DataFrame:
    """Extrahiert die Close-Matrix (Index=Zeit, Spalten=Symbol)."""
    return md.frame.xs("close", level="field", axis=1)
