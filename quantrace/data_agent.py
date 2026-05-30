"""Data Agent — lädt, normalisiert und versioniert Daten über OpenBB.

Standardprovider: yfinance (kostenfrei, kein API-Key). Für Produktivbetrieb
sollte ein bezahlter Provider (FMP, Polygon, Intrinio) konfiguriert werden.

Cache-Strategie: ein Parquet pro (universe, timeframe, start, end, provider)
unter data/processed/. Wiederholte Calls treffen den Cache und sind reproduzierbar.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from quantrace.models import MarketData, Timeframe

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/processed")


def _cache_path(
    cache_dir: Path,
    universe: str,
    timeframe: Timeframe,
    start: date,
    end: date,
    provider: str,
) -> Path:
    fname = f"{universe}__{timeframe.value}__{start}__{end}__{provider}.parquet"
    return cache_dir / fname


def load_universe(
    universe: str,
    symbols: list[str],
    start: date,
    end: date,
    timeframe: Timeframe = Timeframe.DAILY,
    provider: str = "yfinance",
    adjusted: bool = True,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> MarketData:
    """Lädt OHLCV für eine Symbolliste und gibt ein normalisiertes MarketData zurück.

    Frame-Layout:
        MultiIndex columns: (symbol, field) mit field ∈ {open, high, low, close, volume}
        DatetimeIndex tz-naiv, sortiert.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(cache_dir, universe, timeframe, start, end, provider)

    if cache.exists() and not force_refresh:
        log.info("Lade aus Cache: %s", cache)
        frame = pd.read_parquet(cache)
        # MultiIndex muss aus Parquet rekonstruiert werden
        if not isinstance(frame.columns, pd.MultiIndex):
            frame.columns = pd.MultiIndex.from_tuples(
                [tuple(c.split("|", 1)) for c in frame.columns]
            )
        frame.columns.names = ["symbol", "field"]
        frame.index = pd.to_datetime(frame.index)
        return MarketData(
            universe=universe,
            symbols=sorted(set(frame.columns.get_level_values("symbol"))),
            timeframe=timeframe,
            start=start,
            end=end,
            provider=provider,
            adjusted=adjusted,
            frame=frame,
        )

    frame = _fetch_via_openbb(symbols, start, end, timeframe, provider, adjusted)
    _write_cache(frame, cache)
    return MarketData(
        universe=universe,
        symbols=sorted(set(frame.columns.get_level_values("symbol"))),
        timeframe=timeframe,
        start=start,
        end=end,
        provider=provider,
        adjusted=adjusted,
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
            "openbb ist nicht installiert. `pip install -e \".[data]\"` oder `pip install openbb`."
        ) from e

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


def _write_cache(frame: pd.DataFrame, path: Path) -> None:
    flat = frame.copy()
    flat.columns = [f"{sym}|{field}" for sym, field in flat.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    flat.to_parquet(path)
    log.info("Cache geschrieben: %s (%d Zeilen)", path, len(flat))


def close_prices(md: MarketData) -> pd.DataFrame:
    """Extrahiert die Close-Matrix (Index=Zeit, Spalten=Symbol)."""
    return md.frame.xs("close", level="field", axis=1)
