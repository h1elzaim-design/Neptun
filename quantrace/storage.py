"""Parquet-Storage-Layer — lokal oder S3-kompatibel (Cloudflare R2).

Bulk-Kursdaten gehören weder in Postgres (bläht auf, wird langsam, teuer) noch
auf den ephemeren Render-Container-Disk (Cache ist nach jedem Deploy weg).
Dieses Modul abstrahiert Parquet-I/O so, dass derselbe Code lokal
(`data/processed/`) und gegen ein R2/S3-Bucket läuft — der Provider entscheidet
sich allein über Env-Variablen.

Konfiguration:
    QUANTRACE_DATA_LAKE    "s3://bucket/prefix"  → R2/S3   |  unset → lokal (data/processed)
    R2_ENDPOINT_URL        https://<account>.r2.cloudflarestorage.com
    AWS_ACCESS_KEY_ID      R2 Access Key ID
    AWS_SECRET_ACCESS_KEY  R2 Secret Access Key

Der `s3fs`-Import passiert lazy — nur wenn tatsächlich ein s3://-Pfad benutzt
wird. Lokale Pfade und CI brauchen kein s3fs.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_LOCAL_DIR = Path("data/processed")


def _lake_root() -> str:
    return os.environ.get("QUANTRACE_DATA_LAKE", "").strip()


def is_remote() -> bool:
    """True, wenn der Daten-See auf S3/R2 zeigt."""
    return _lake_root().startswith("s3://")


def _s3_storage_options() -> dict:
    """pandas/fsspec storage_options für R2 (S3-kompatibel)."""
    opts: dict = {
        "key": os.environ.get("AWS_ACCESS_KEY_ID", "").strip() or None,
        "secret": os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip() or None,
        # R2 braucht path-style Adressierung: https://<acct>.r2.../<bucket>/key.
        # Ohne das baut botocore virtual-hosted (https://<bucket>.<acct>.r2...),
        # was bei R2 — besonders EU-Jurisdiction — als NoSuchBucket scheitert.
        "config_kwargs": {"s3": {"addressing_style": "path"}},
    }
    endpoint = os.environ.get("R2_ENDPOINT_URL", "").strip()
    if endpoint:
        # R2 ist nicht AWS — der Endpoint muss explizit gesetzt werden.
        opts["client_kwargs"] = {"endpoint_url": endpoint}
    return opts


def cache_path(filename: str) -> str:
    """Vollständiger Pfad/URI für eine Cache-Datei (lokal oder s3://)."""
    root = _lake_root()
    if root:
        return f"{root.rstrip('/')}/{filename}"
    return str(DEFAULT_LOCAL_DIR / filename)


def exists(path: str) -> bool:
    if path.startswith("s3://"):
        import fsspec

        fs, _, paths = fsspec.get_fs_token_paths(path, storage_options=_s3_storage_options())
        return bool(fs.exists(paths[0]))
    return Path(path).exists()


def read_parquet(path: str) -> pd.DataFrame:
    # partitioning=None: paths like raw/symbol=TLT/data.parquet would otherwise
    # trigger pandas' automatic hive-partition inference (symbol →
    # dictionary<int32>), which collides with a leftover categorical `symbol`
    # column (dictionary<int8>) in files written by older fetch versions:
    # "ArrowTypeError: Unable to merge: Field symbol has incompatible types".
    if path.startswith("s3://"):
        return pd.read_parquet(
            path, storage_options=_s3_storage_options(), partitioning=None
        )
    return pd.read_parquet(path, partitioning=None)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    if path.startswith("s3://"):
        df.to_parquet(path, storage_options=_s3_storage_options())
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


# ---------------------------------------------------------------------------
# Per-Symbol-Partitionen (hive-style: raw/symbol=<SYM>/data.parquet)
#
# Roh-OHLCV + Corporate Actions liegen pro Symbol partitioniert. Vorteile:
#   - Inkrementelles Anhängen: nur das eine Symbol-File wird neu geschrieben.
#   - DuckDB kann mit hive_partitioning das `symbol` gratis aus dem Pfad ziehen.
#   - Datumsbereich-Änderung lädt nicht das ganze Universum neu.
# ---------------------------------------------------------------------------

_RAW_PREFIX = "raw"


def symbol_partition(symbol: str) -> str:
    return cache_path(f"{_RAW_PREFIX}/symbol={symbol}/data.parquet")


def read_symbol_raw(symbol: str) -> pd.DataFrame | None:
    """Gespeicherte Roh-Historie eines Symbols (DatetimeIndex) oder None."""
    path = symbol_partition(symbol)
    if not exists(path):
        return None
    df = read_parquet(path)
    # Ältere Fetch-Versionen haben eine redundante symbol-Spalte mitgeschrieben
    # (steckt schon im Partitionspfad). Beim Lesen droppen — der nächste
    # write_symbol_raw persistiert dann das bereinigte Schema.
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def write_symbol_raw(symbol: str, df: pd.DataFrame) -> None:
    """Schreibt die volle Roh-Historie eines Symbols. `date` wird als echte
    Spalte abgelegt, damit DuckDB sauber darauf filtern kann."""
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.index.name = "date"
    write_parquet(out.reset_index(), symbol_partition(symbol))


# Coverage-Metadaten: welche *angefragte* Kalenderspanne pro Symbol schon
# geholt wurde. Nötig, weil Handelstage < Kalendertage — ohne das würden
# leere Ränder (Wochenenden/Feiertage) bei jedem Load neu gefetcht.

def _coverage_path(symbol: str) -> str:
    return cache_path(f"{_RAW_PREFIX}/symbol={symbol}/coverage.json")


def _read_text(path: str) -> str:
    if path.startswith("s3://"):
        import fsspec

        with fsspec.open(path, "r", **_s3_storage_options()) as fh:
            return fh.read()
    return Path(path).read_text()


def _write_text(path: str, text: str) -> None:
    if path.startswith("s3://"):
        import fsspec

        with fsspec.open(path, "w", **_s3_storage_options()) as fh:
            fh.write(text)
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def read_coverage(symbol: str) -> tuple[date, date] | None:
    path = _coverage_path(symbol)
    if not exists(path):
        return None
    d = json.loads(_read_text(path))
    return date.fromisoformat(d["start"]), date.fromisoformat(d["end"])


def write_coverage(symbol: str, start: date, end: date) -> None:
    _write_text(
        _coverage_path(symbol),
        json.dumps({"start": start.isoformat(), "end": end.isoformat()}),
    )


def _duckdb_conn():  # pragma: no cover - dünner Adapter
    import duckdb

    con = duckdb.connect()
    if is_remote():
        con.execute("INSTALL httpfs; LOAD httpfs;")
        endpoint = (
            os.environ.get("R2_ENDPOINT_URL", "")
            .replace("https://", "")
            .replace("http://", "")
            .rstrip("/")
        )
        if endpoint:
            con.execute(f"SET s3_endpoint='{endpoint}';")
        con.execute("SET s3_url_style='path';")
        con.execute("SET s3_use_ssl=true;")
        con.execute("SET s3_region='auto';")
        ak = os.environ.get("AWS_ACCESS_KEY_ID", "")
        sk = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        if ak:
            con.execute(f"SET s3_access_key_id='{ak}';")
        if sk:
            con.execute(f"SET s3_secret_access_key='{sk}';")
    return con


def read_symbols(symbols: list[str], start, end) -> pd.DataFrame:
    """Liest die angeforderten Symbole im Datumsbereich aus dem Daten-See.

    Nutzt DuckDB (columnar, selektiv, lokal wie R2 via httpfs). Gibt ein
    Long-Format zurück: Spalten date, symbol, open/high/low/close/volume,
    divCash, splitFactor. Symbole ohne Partition werden still übersprungen.
    """
    paths = [symbol_partition(s) for s in symbols]
    present = [p for p in paths if exists(p)]
    if not present:
        return pd.DataFrame()

    con = _duckdb_conn()
    try:
        placeholders = ",".join(["?"] * len(present))
        sql = (
            f"SELECT * FROM read_parquet([{placeholders}], hive_partitioning=true) "
            "WHERE date >= ? AND date <= ? ORDER BY symbol, date"
        )
        return con.execute(sql, [*present, str(start), str(end)]).df()
    finally:
        con.close()
