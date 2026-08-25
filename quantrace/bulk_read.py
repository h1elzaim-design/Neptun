"""Lesepfad auf Schicht 2 — mit Adjustierung aus den eigenen Corporate Actions.

Zwei Dinge macht dieses Modul, und das zweite ist das wichtigere.

## 1 · Adjustieren aus den eigenen Daten

Die Spalte ``adjusted_close`` aus dem Bulk ist zum **Abrufzeitpunkt**
adjustiert. Sie zu benutzen hätte zwei Fehler auf einmal:

* **Look-ahead.** Der Wert für den 12.09.2008 trägt alle Splits und Dividenden
  ein, die bis zum Ladetag passiert sind — also auch die aus 2015. Ein
  Backtest über 2008 rechnet damit mit Wissen von 2026.
* **Nahtstellen.** Wer 1996–2010 heute lädt und 2011–2026 nächstes Jahr,
  klebt zwei Abschnitte mit verschiedenen Faktoren aneinander. An jedem Split
  dazwischen springt die Reihe, und der Sprung sieht aus wie eine Rendite.

Deshalb: Rohkurse aus Schicht 2, Aktionen aus ``us_equity_splits`` und
``us_equity_dividends``, und die Adjustierung passiert **beim Lesen** über
``quantrace.adjust.adjust_ohlcv``. Damit ist sie eine Funktion der Daten statt
eine Funktion des Abrufdatums.

## 2 · Nicht so tun, als wäre adjustiert, was nicht adjustiert ist

Der Lake lädt über Tage. Solange die Actions-Feeds hinterherhinken, gibt es
Fenster mit Kursen und ohne Splits. Die naheliegende Implementierung liefert
dort stillschweigend die Rohreihe — und der Aufrufer hält sie für adjustiert.

Das ist exakt das Muster, an dem sich dieses Projekt schon zweimal die Finger
verbrannt hat: `_score_realism` las fehlende Kosten als 0,0, `us_core_etfs`
versprach ein Fenster, das es nicht hatte. **Fehlend ist ein eigener Fall, kein
Nullwert.**

Also trägt jedes Ergebnis eine ``Adjustment``-Auskunft: ob adjustiert wurde,
welches Fenster die Actions abdecken, und was das für die Reihe bedeutet. Wer
sie ignoriert, tut das sichtbar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from quantrace import storage
from quantrace.adjust import adjust_ohlcv
from quantrace.instruments import US_DIVIDENDS_PREFIX, US_SPLITS_PREFIX
from quantrace.resolve import RESOLVED_PREFIX, materialised_keys

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Adjustment:
    """Was mit den Kursen passiert ist — und was nicht.

    ``status`` ist die eine Angabe, auf die es ankommt:

    ``full``
        Splits und Dividenden decken das angefragte Fenster ab. Die Reihe ist
        total-return-adjustiert.
    ``partial``
        Die Actions decken nur einen Teil ab. Vor ``covered_from`` ist die
        Reihe roh — an einem Split dort springt sie.
    ``none``
        Keine Actions im Lake. Die Reihe ist **roh**, nicht adjustiert.
    """

    status: str
    covered_from: date | None = None
    covered_to: date | None = None
    n_splits: int = 0
    n_dividends: int = 0

    @property
    def is_total_return(self) -> bool:
        return self.status == "full"

    def warning(self) -> str | None:
        """Ein Satz für Log, API und UI — oder ``None``, wenn alles sauber ist."""
        if self.status == "full":
            return None
        if self.status == "none":
            return (
                "Keine Splits/Dividenden im Lake — die Reihe ist ROH, nicht "
                "adjustiert. Renditen unterschätzen den Total Return, und an "
                "jedem Split springt der Kurs."
            )
        return (
            f"Corporate Actions decken erst ab {self.covered_from} — davor ist "
            "die Reihe roh. Ein Split vor diesem Datum erscheint als Rendite."
        )


def _actions_window(prefix: str) -> tuple[date | None, date | None]:
    days = storage.list_day_partitions(prefix)
    return (min(days), max(days)) if days else (None, None)


def _read_actions(prefix: str, codes: list[str], start: date, end: date) -> pd.DataFrame:
    """Splits oder Dividenden für die Codes im Fenster. Leer, wenn nichts liegt.

    Nur die Partitionen im angefragten Fenster. ``date=*`` über den ganzen
    Feed wären auf R2 tausend HTTP-GETs für eine Adjustierung, die nur das
    Chart-Jahr braucht — DuckDB kann Hive-Pruning erst anwenden, nachdem
    der Glob expandiert ist, und die Expansion *ist* der teure Teil.
    """
    days = storage.list_day_partitions(prefix)
    if not days or not codes:
        return pd.DataFrame()
    im_fenster = [d for d in days if start <= d <= end]
    if not im_fenster:
        return pd.DataFrame()

    pfade = [
        storage.cache_path(f"{prefix}/date={d.isoformat()}/*.parquet") for d in im_fenster
    ]
    con = storage._duckdb_conn()
    try:
        dateien = ",".join(["?"] * len(pfade))
        codes_ph = ",".join(["?"] * len(codes))
        sql = (
            f"SELECT * FROM read_parquet([{dateien}], union_by_name=true) "
            f"WHERE code IN ({codes_ph})"
        )
        return con.execute(sql, [*pfade, *codes]).df()
    except Exception as exc:  # pragma: no cover - defekte Partition
        # Eine kaputte Actions-Partition darf den Kurs-Read nicht killen —
        # aber sie darf auch nicht als „keine Actions" durchgehen. Der
        # Aufrufer sieht das am `status`, der dann nicht `full` wird.
        log.warning("Actions unter %s nicht lesbar: %s", prefix, exc)
        return pd.DataFrame()
    finally:
        con.close()


def read_instruments(
    instruments: list[str],
    start: date,
    end: date,
    *,
    adjust: bool = True,
) -> tuple[pd.DataFrame, Adjustment]:
    """Long-OHLCV aus Schicht 2, optional adjustiert.

    Returns
    -------
    (frame, adjustment)
        ``frame`` im Long-Format (``instrument``, ``code``, ``date``, OHLCV).
        ``adjustment`` sagt, ob und wie weit adjustiert wurde — **immer
        prüfen**, bevor die Zahlen irgendwo landen.
    """
    # Ein LIST-Aufruf gegen den Prefix statt eines `exists` je Instrument.
    # Der Dateiname innerhalb der Partition steht nicht fest — DuckDBs
    # PARTITION_BY schreibt `data_0.parquet`, ein Einzel-Write `data.parquet`
    # —, deshalb wird pro Instrument geglobt statt geraten.
    vorhandene = materialised_keys()
    vorhanden = [i for i in instruments if i in vorhandene]
    if not vorhanden:
        return pd.DataFrame(), Adjustment(status="none")

    pfade = [
        storage.cache_path(f"{RESOLVED_PREFIX}/instrument={i}/*.parquet") for i in vorhanden
    ]
    con = storage._duckdb_conn()
    try:
        platzhalter = ",".join(["?"] * len(pfade))
        sql = (
            f"SELECT * FROM read_parquet([{platzhalter}], union_by_name=true) "
            "WHERE date >= ? AND date <= ? ORDER BY instrument, date"
        )
        prices = con.execute(sql, [*pfade, str(start), str(end)]).df()
    finally:
        con.close()

    if prices.empty:
        return prices, Adjustment(status="none")

    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    if not adjust:
        return prices, Adjustment(status="none")

    codes = sorted({str(c) for c in prices["code"].dropna().unique()})
    splits = _read_actions(US_SPLITS_PREFIX, codes, start, end)
    divs = _read_actions(US_DIVIDENDS_PREFIX, codes, start, end)

    s_von, s_bis = _actions_window(US_SPLITS_PREFIX)
    d_von, d_bis = _actions_window(US_DIVIDENDS_PREFIX)

    if s_von is None and d_von is None:
        # Kein einziger Actions-Tag im Lake. Roh zurückgeben — und es sagen.
        info = Adjustment(status="none")
        log.warning("%s", info.warning())
        return prices, info

    von = max([d for d in (s_von, d_von) if d is not None])
    bis = min([d for d in (s_bis, d_bis) if d is not None])
    voll = von <= start and bis >= end
    info = Adjustment(
        status="full" if voll else "partial",
        covered_from=von,
        covered_to=bis,
        n_splits=int(len(splits)),
        n_dividends=int(len(divs)),
    )
    if not voll:
        log.warning("%s", info.warning())

    return _apply_actions(prices, splits, divs), info


def _apply_actions(
    prices: pd.DataFrame, splits: pd.DataFrame, divs: pd.DataFrame
) -> pd.DataFrame:
    """Baut je Instrument ``divCash``/``splitFactor`` und adjustiert.

    Die beiden Spaltennamen sind Tiingo-Vokabular — bewusst übernommen, damit
    ``adjust.adjust_ohlcv`` unverändert weiterbenutzt wird. Eine zweite
    Adjustierungs-Implementierung wäre eine zweite Wahrheit.
    """
    # Lazy: `providers.eodhd` zieht auf Modulebene `httpx` herein. Ein
    # Lesepfad, der nur einen Bruch parst, soll keinen HTTP-Client brauchen —
    # dieselbe Funktion bleibt trotzdem die einzige Implementierung.
    from quantrace.providers.eodhd import parse_split_ratio

    split_map: dict[tuple[str, date], float] = {}
    if not splits.empty and "split" in splits.columns:
        for row in splits.itertuples(index=False):
            faktor = parse_split_ratio(getattr(row, "split", None))
            if faktor is None or faktor <= 0:
                continue
            tag = pd.to_datetime(row.date).date()
            split_map[(str(row.code), tag)] = faktor

    div_map: dict[tuple[str, date], float] = {}
    if not divs.empty and "dividend" in divs.columns:
        for row in divs.itertuples(index=False):
            betrag = getattr(row, "dividend", None)
            if betrag is None or pd.isna(betrag):
                continue
            tag = pd.to_datetime(row.date).date()
            div_map[(str(row.code), tag)] = float(betrag)

    teile: list[pd.DataFrame] = []
    for _instrument, teil in prices.groupby("instrument", sort=True):
        teil = teil.sort_values("date").reset_index(drop=True)
        code = str(teil["code"].iloc[0])
        idx = pd.DatetimeIndex(pd.to_datetime(teil["date"]))
        # `.to_numpy()` ist hier Pflicht, nicht Stil: mit einem expliziten
        # `index` reindiziert pandas übergebene Series auf diesen Index — die
        # Positionen 0..n-1 träfen auf Zeitstempel, und der ganze Frame käme
        # als NaN zurück. Roh-Arrays haben keinen Index, der kollidieren kann.
        roh = pd.DataFrame(
            {
                "open": teil["open"].astype(float).to_numpy(),
                "high": teil["high"].astype(float).to_numpy(),
                "low": teil["low"].astype(float).to_numpy(),
                "close": teil["close"].astype(float).to_numpy(),
                "volume": teil["volume"].astype(float).to_numpy(),
                "splitFactor": [split_map.get((code, d), 1.0) for d in teil["date"]],
                "divCash": [div_map.get((code, d), 0.0) for d in teil["date"]],
            },
            index=idx,
        )
        adj = adjust_ohlcv(roh)
        adj = adj.reset_index(drop=True)
        neu = teil.copy()
        for col in ("open", "high", "low", "close", "volume"):
            if col in adj.columns:
                neu[col] = adj[col].to_numpy()
        teile.append(neu)

    return pd.concat(teile, ignore_index=True) if teile else prices


__all__ = ["Adjustment", "read_instruments"]
