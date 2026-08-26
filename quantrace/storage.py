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
import re
from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_LOCAL_DIR = Path("data/processed")

#: Hive-Partition ``date=YYYY-MM-DD``. Bewusst nur das Datumsformat — andere
#: Schlüssel (``symbol=``, ``isin=``) haben eigene Leser.
_DAY_PART_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")


def _lake_root() -> str:
    return os.environ.get("QUANTRACE_DATA_LAKE", "").strip()


def r2_endpoint() -> str:
    """``R2_ENDPOINT_URL`` auf **Schema + Host** normalisiert. ``""`` wenn ungesetzt.

    Der Endpoint ist der *Account-Host*, nicht der Bucket. Welcher Bucket
    gemeint ist, steht in ``QUANTRACE_DATA_LAKE`` (``s3://qlab/processed``) —
    ein trailing ``/qlab`` am Endpoint nennt ihn ein zweites Mal, an einer
    Stelle, an der ihn niemand erwartet.

    **Warum das eine gemeinsame Funktion ist und keine zwei.** Bis 2026-08-12
    normalisierte nur der s3fs-Pfad (``urlparse`` → ``scheme://netloc``); der
    DuckDB-Pfad strippte lediglich Schema und Slashes, sodass ``/qlab``
    überlebte und als ``SET s3_endpoint='…/qlab'`` landete. Ergebnis wäre
    gewesen: Schreiben, Lesen und LIST über pandas laufen, **jede
    DuckDB-Abfrage scheitert** — also Coverage grün und Backtest rot, aus
    derselben Env-Var.

    Genau die Form, die dieses Projekt schon zweimal Geld gekostet hat: ein
    Modus kaputt, die anderen heil, und niemand sieht den Zusammenhang.
    """
    raw = os.environ.get("R2_ENDPOINT_URL", "").strip()
    if not raw:
        return ""
    from urllib.parse import urlparse

    # Ohne Schema hätte `urlparse` alles in `path` gelegt und `netloc` leer
    # gelassen — der Endpoint wäre still verschwunden.
    u = urlparse(raw if "//" in raw else f"https://{raw}")
    return f"{u.scheme}://{u.netloc}" if u.netloc else ""


def is_remote() -> bool:
    """True, wenn der Daten-See auf S3/R2 zeigt."""
    return _lake_root().startswith("s3://")


def lake_description() -> tuple[str, bool]:
    """``(Beschreibung, ist_lokaler_rueckfall)`` — wohin zeigt der Lake gerade?

    **Warum das eine eigene Funktion ist.** Ohne ``QUANTRACE_DATA_LAKE`` fällt
    alles still auf ``data/processed`` zurück. Ein voller R2-Lake und eine
    ungesourcte ``.env`` sehen dann identisch aus: beide melden „0 Tage
    geladen". Am 2026-08-11 hat genau das eine Viertelstunde gekostet — der
    Loader lief gegen R2, das `--status` daneben gegen ein leeres lokales
    Verzeichnis, und die Ausgabe behauptete, es liege nichts da.

    Dieselbe Fehlerklasse wie der R2-Endpoint mit trailing ``/qlab`` (LIST
    leer, PUT/GET grün): **ein falsches Ziel sieht aus wie fehlende Daten.**
    Deshalb sagen die Skripte jetzt beim Start, mit wem sie reden.
    """
    root = _lake_root()
    if not root:
        return f"{DEFAULT_LOCAL_DIR} (lokal — QUANTRACE_DATA_LAKE ist nicht gesetzt)", True
    return root, False


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
    endpoint = r2_endpoint()
    if endpoint:
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


def list_children(relative: str = "") -> list[str]:
    """Namen der direkten Kinder unter einem Lake-relativem Pfad.

    Gibt basenames zurück (``date=2020-01-02``, ``data.parquet``), sortiert.
    Existiert der Pfad nicht, leere Liste — kein Fehler. Das ist die
    Primitive, die Coverage und der Loader-Status brauchen: **ein** Aufruf
    statt Tausender ``exists``-Checks über 30 Jahre × drei Feeds.

    Lokal und S3/R2. Relative Pfade sind ohne führenden Slash und ohne
    Lake-Wurzel: ``us_equities``, nicht ``data/processed/us_equities``.
    """
    root = _lake_root()
    if relative:
        target = cache_path(relative)
    else:
        target = root.rstrip("/") if root else str(DEFAULT_LOCAL_DIR)

    if target.startswith("s3://"):
        import fsspec

        fs, _, paths = fsspec.get_fs_token_paths(target, storage_options=_s3_storage_options())
        base = paths[0].rstrip("/")
        try:
            entries = fs.ls(base, detail=False)
        except FileNotFoundError:
            # S3/R2 hat keine leeren Verzeichnisse. Existiert der Prefix nur
            # als Key-Präfix (kein 0-Byte-Marker), schlägt ls fehl — glob
            # auf die Kinder rettet das.
            entries = fs.glob(base + "/*") or []
        names: list[str] = []
        for entry in entries:
            # fsspec liefert volle Keys; basename ohne trailing slash.
            name = str(entry).rstrip("/").rsplit("/", 1)[-1]
            if name and name != base.rsplit("/", 1)[-1]:
                names.append(name)
        return sorted(set(names))

    path = Path(target)
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir())


def list_day_partitions(prefix: str) -> list[date]:
    """Sortierte Daten, für die unter ``prefix`` eine ``date=``-Partition liegt.

    Leere Partitionen (handelsfreier Tag, null Splits) zählen mit: sie sind
    eine Auskunft, keine Lücke — siehe ``scripts/load_us_equities.py``.
    Unparsable Kinder werden still übersprungen; ein fremdes Schema darf die
    Coverage nicht kippen.
    """
    out: list[date] = []
    for name in list_children(prefix):
        m = _DAY_PART_RE.match(name)
        if not m:
            continue
        try:
            out.append(date.fromisoformat(m.group(1)))
        except ValueError:
            continue
    return sorted(out)


def contiguous_runs(days: list[date], *, max_runs: int = 8) -> list[dict[str, str]]:
    """Zusammenhängende Blöcke — die Antwort auf „985 Tage, aber wo?".

    Zusammenhängend heißt: keine *Handelstage* dazwischen ausgelassen.
    Wochenenden trennen also nicht. Ohne Kalender wird auf „höchstens vier
    Kalendertage Abstand" zurückgefallen, was Feiertagsbrücken abdeckt.

    **Warum das hier liegt und nicht in der API.** Der längste zusammenhängende
    Block ist eine Eigenschaft des Lakes, keine der Oberfläche — und es gibt
    zwei Leser: die Coverage-Ansicht und die Rekonstitution (#255), die ihren
    letzten Stichtag daran begrenzt. Zwei Implementierungen derselben Regel
    wären zwei Meinungen darüber, was „zusammenhängend" heißt.

    Die Blöcke kommen **längster zuerst**: wer die Ansicht überfliegt, soll das
    nutzbare Fenster sehen und nicht den ersten Schnipsel.
    """
    if not days:
        return []

    from quantrace.calendars import trading_sessions

    sortiert = sorted(set(days))
    sessions = trading_sessions("us_equity", sortiert[0], sortiert[-1])
    if sessions is not None:
        handelstage = [d.date() for d in sessions]
        position = {d: i for i, d in enumerate(handelstage)}

        def zusammen(a: date, b: date) -> bool:
            ia, ib = position.get(a), position.get(b)
            if ia is None or ib is None:
                return (b - a).days <= 4
            return ib - ia <= 1
    else:

        def zusammen(a: date, b: date) -> bool:
            return (b - a).days <= 4

    bloecke: list[list[date]] = [[sortiert[0]]]
    for prev, cur in zip(sortiert, sortiert[1:], strict=False):
        if zusammen(prev, cur):
            bloecke[-1].append(cur)
        else:
            bloecke.append([cur])

    bloecke.sort(key=len, reverse=True)
    return [
        {"start": b[0].isoformat(), "end": b[-1].isoformat(), "n_days": len(b)}
        for b in bloecke[:max_runs]
    ]


def longest_run(days: list[date]) -> tuple[date, date] | None:
    """Der längste zusammenhängende Block als Datumspaar — oder ``None``.

    Die bequeme Form für Rechencode. ``min``/``max`` über alle Partitionen wäre
    die falsche Antwort: das sind **Extrema**. Am 2026-08-15 bestand der Lake
    aus zwei Blöcken plus einer Streu-Partition vom Sommer 2011 aus einem
    frühen Loader-Test — die Extrema meldeten damit fünfzehn Jahre Historie,
    während mittendrin viereinhalb Jahre fehlten.
    """
    return run_bounds(contiguous_runs(days, max_runs=1000))


def run_bounds(runs: list[dict[str, str]]) -> tuple[date, date] | None:
    """Grenzen des längsten Blocks aus einer bereits berechneten Liste.

    ``contiguous_runs`` sortiert **nach Handelstagen**, längster zuerst — also
    ist ``runs[0]`` die Antwort. Ein eigenes ``max`` über die Kalenderspanne
    wäre ein zweiter Begriff von „am längsten": ein Block mit 25 Handelstagen
    über eine Feiertagsbrücke kann kalendarisch weiter reichen als einer mit
    30. Die Coverage-Ansicht zeigte dann den einen Block und das Backtest-
    Fenster nähme den anderen.
    """
    if not runs:
        return None
    return date.fromisoformat(runs[0]["start"]), date.fromisoformat(runs[0]["end"])


def delete_tree(path: str) -> int:
    """Löscht einen Pfad samt Inhalt. Gibt die Zahl entfernter Objekte zurück.

    Die einzige löschende Primitive im Modul, und sie ist bewusst schmal: ein
    Pfad, kein Glob, kein Muster. Wer ein Muster übergeben will, soll die
    Treffer vorher auflisten und einzeln übergeben — dann steht die Liste im
    Log des Aufrufers, statt in einer Wildcard zu verschwinden.

    Ein nicht existierender Pfad ist keine Ausnahme, sondern 0.
    """
    if path.startswith("s3://"):
        import fsspec

        fs, _, paths = fsspec.get_fs_token_paths(path, storage_options=_s3_storage_options())
        ziel = paths[0].rstrip("/")
        try:
            treffer = fs.find(ziel)
        except FileNotFoundError:
            return 0
        if not treffer:
            return 0
        fs.rm(treffer)
        return len(treffer)

    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        p.unlink()
        return 1
    n = sum(1 for child in p.rglob("*") if child.is_file())
    import shutil

    shutil.rmtree(p)
    return n


def read_parquet(path: str) -> pd.DataFrame:
    # partitioning=None: paths like raw/symbol=TLT/data.parquet would otherwise
    # trigger pandas' automatic hive-partition inference (symbol →
    # dictionary<int32>), which collides with a leftover categorical `symbol`
    # column (dictionary<int8>) in files written by older fetch versions:
    # "ArrowTypeError: Unable to merge: Field symbol has incompatible types".
    if path.startswith("s3://"):
        return pd.read_parquet(path, storage_options=_s3_storage_options(), partitioning=None)
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
    # Große Schreibläufe sortieren extern: `materialise()` schiebt 88 Mio.
    # Zeilen durch ein ORDER BY, das nicht in den RAM passt. Ohne Angabe legt
    # DuckDB diesen Stapel unter `.tmp` im Arbeitsverzeichnis ab — also auf der
    # Systemplatte, unabhängig davon, wo der Platz tatsächlich ist. Der Ort
    # gehört deshalb konfiguriert, nicht geerbt.
    tmp_dir = os.environ.get("QUANTRACE_DUCKDB_TEMP_DIR", "").strip()
    if tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        con.execute(f"SET temp_directory='{tmp_dir}';")
    if is_remote():
        con.execute("INSTALL httpfs; LOAD httpfs;")
        # DuckDB will den Host ohne Schema — die Normalisierung selbst steckt
        # in `r2_endpoint()`, gemeinsam mit dem s3fs-Pfad. Zwei eigene
        # Varianten waren genau der Unterschied, an dem ein trailing `/qlab`
        # den einen Pfad brach und den anderen nicht.
        endpoint = r2_endpoint()
        if endpoint:
            host = endpoint.split("://", 1)[1]
            con.execute(f"SET s3_endpoint='{host}';")
        con.execute("SET s3_url_style='path';")
        con.execute(
            f"SET s3_use_ssl={'true' if endpoint.startswith('https://') or not endpoint else 'false'};"
        )
        con.execute("SET s3_region='auto';")
        ak = os.environ.get("AWS_ACCESS_KEY_ID", "")
        sk = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        if ak:
            con.execute(f"SET s3_access_key_id='{ak}';")
        if sk:
            con.execute(f"SET s3_secret_access_key='{sk}';")
        # Der Default (= CPU-Kerne, auf Heroku Basic nur eine Hand voll) bremst
        # nichts CPU-Gebundenes hier — es sind lauter kleine R2-GETs, eins je
        # Tagespartition. Gemessen (#Actions-Read über 17 Jahre AAPL):
        # 3.452 Dateien brauchten ~60s mit dem Default und ~23s mit 64 Threads;
        # 128 hat gehangen (zu viele gleichzeitige Verbindungen). 32 ist die
        # sichere Seite von diesem Knick.
        con.execute("SET threads=32;")
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
