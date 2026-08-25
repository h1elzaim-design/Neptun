"""Data Agent — lädt, normalisiert, versioniert und qualitätssichert Marktdaten.

**Der Backtest-Pfad ist ``provider='eodhd'``** und liest den US-Bulk: rohe
Tagesquerschnitte (Schicht 1), über Identitäten aufgelöst (Schicht 2),
adjustiert beim Lesen. Das ist seit dem 2026-08-13 der einzige Weg zu Kursen.

**Warum Tiingo weg ist.** Es war der bequemere Weg und der falsche: eine
Symbolliste, die nur Überlebende führt. Ein Backtest über 2008 sah dort nie
LEH, WM oder BSC — nicht weil sie fehlten, sondern weil eine Liste der heute
handelbaren Papiere per Konstruktion keine Toten enthält. Der EODHD-Bulk ist
survivorship-frei ab Werk: der Querschnitt vom 2008-09-12 enthält alle vier.

Der Preis dafür steht in ``docs/STATUS.md``: bis die Historie geladen ist,
reicht der Lake nicht weit zurück. Ein Backtest, der an fehlenden Daten
scheitert, ist trotzdem der bessere Zustand als einer, der auf Überlebenden
rechnet und eine Zahl liefert.

Legacy-Pfad (yfinance/fmp/… via openbb): kombinierter Universum-Cache,
adjustiert beim Fetch. Optionales Extra `.[data]`, für Backtests nicht
vorgesehen.

Storage entscheidet lokal vs. R2 allein über Env (QUANTRACE_DATA_LAKE).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from quantrace import quality, storage
from quantrace.calendars import DEFAULT_CALENDAR, validate_universe_calendar
from quantrace.data_providers import bootstrap_credentials, default_provider
from quantrace.membership import Membership
from quantrace.models import MarketData, Timeframe

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = storage.DEFAULT_LOCAL_DIR
_OHLCV = ["open", "high", "low", "close", "volume"]

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
    cost_class: str | None = None,
    membership: Membership | None = None,
) -> MarketData:
    """Lädt OHLCV für eine Symbolliste und gibt ein normalisiertes MarketData zurück.

    Frame-Layout:
        MultiIndex columns: (symbol, field) mit field ∈ {open, high, low, close, volume}
        DatetimeIndex tz-naiv, sortiert.

    ``cost_class`` kommt aus dem Universe-YAML und gilt nur für **konstruierte**
    Universen: dort bestimmt eine Regel die Mitglieder, also auch ihre
    Kostenklasse. Handverlesene Listen klassifizieren weiter jedes Symbol
    einzeln in `config/costs.yaml`.

    ``membership`` ist die zeitvariable Mitgliedschaft rekonstituierter
    Universen (#255). ``symbols`` ist dann die **Vereinigung** über alle
    Perioden — also das, was geladen werden muss —, und die Mitgliedschaft
    schneidet daraus je Tag heraus, was an diesem Tag dazugehörte. Ohne diesen
    Schritt hätte ein rekonstituiertes Universum den umgekehrten Fehler des
    Schnappschusses: es handelte Papiere Jahre bevor die Regel sie wählte.

    **Die Reihenfolge ist Absicht:** erst laden und die Datenqualität prüfen,
    dann beschneiden. Andersherum meldete die Qualitätsprüfung jedes
    Nichtmitglied als fehlende Bars — ein Fehler, der keiner ist, und der die
    echten überdeckt.
    """
    # Ein Universum, ein Kalender (#184) — vor jedem I/O, damit ein gemischtes
    # Universum nicht erst Daten zieht und dann scheitert.
    validate_universe_calendar(symbols, calendar, universe=universe, cost_class=cost_class)

    provider = provider or default_provider()
    if provider == "eodhd":
        md = _load_via_bulk(universe, symbols, start, end, timeframe, adjusted, calendar)
    elif provider == "tiingo":
        # Eigener Zweig statt „unbekannter Provider": ein Universe-YAML mit
        # `provider: tiingo` ist kein Tippfehler, sondern eine Datei von vor
        # dem 2026-08-13. Der Unterschied gehört in die Fehlermeldung.
        raise ValueError(
            f"Universum {universe!r} nennt provider='tiingo' — der Pfad ist entfernt. "
            "Der Backtest-Pfad ist der EODHD-Bulk: `provider: eodhd` in der "
            "Universe-YAML setzen. Tiingo war eine Survivor-Liste; ein Backtest "
            "über 2008 sah dort nie LEH, WM oder BSC."
        )
    else:
        md = _load_via_openbb_cache(
            universe, symbols, start, end, timeframe, provider, adjusted, cache_dir,
            force_refresh, calendar,
        )

    # Nachträglich statt durch drei Ladepfade gereicht: die Klasse ist eine
    # Eigenschaft des Universums, keine der Quelle. `model_copy` lässt
    # `content_hash` unangetastet — der Hash beschreibt die Daten, und die
    # ändern sich hier nicht.
    if cost_class:
        md = md.model_copy(update={"cost_class": cost_class})

    # Die Mitgliedschaft ändert die Daten sehr wohl, deshalb baut `apply` ein
    # neues MarketData statt zu kopieren: der content_hash muss den
    # beschnittenen Rahmen beschreiben, nicht den geladenen.
    return membership.apply(md) if membership is not None else md


# ---------------------------------------------------------------------------
# Bulk-Pfad (EODHD): Schicht 2, survivorship-frei, adjust-on-read
# ---------------------------------------------------------------------------


def _load_via_bulk(
    universe: str,
    symbols: list[str],
    start: date,
    end: date,
    timeframe: Timeframe,
    adjusted: bool,
    calendar: str | None = None,
) -> MarketData:
    """Universum aus dem EODHD-Bulk laden — über Schicht 2, nicht über Ticker.

    Der Unterschied zum Tiingo-Pfad ist nicht die Quelle, sondern die
    **Identität**: dort ist der Ticker der Schlüssel, hier wird er zum
    Zeitpunkt der Anfrage auf ein Instrument aufgelöst. Ein Code, der im
    Fenster den Besitzer gewechselt hat, führt zu einem Abbruch statt zu einer
    stillschweigend verketteten Reihe (`resolve.AmbiguousCodeError`).

    **Adjustiert wird aus den eigenen Corporate Actions.** Fehlen sie für das
    Fenster, ist die Reihe roh — und dieser Loader sagt es, statt sie als
    Total Return durchgehen zu lassen. Bei ``adjusted=True`` ist eine nicht
    adjustierbare Reihe ein **Fehler**: wer Total Return anfordert und Rohkurse
    bekommt, rechnet mit einem Vorzeichenfehler weiter.
    """
    from quantrace import bulk_read, resolve

    if timeframe is not Timeframe.DAILY:
        raise ValueError("Der Bulk-Lake führt ausschließlich Tagesdaten.")

    aufgeloest, fehlend = resolve.resolve_symbols(symbols, start, end)
    if not aufgeloest:
        raise RuntimeError(
            f"Kein Symbol aus {universe} ist im Bulk-Lake für {start}..{end} aufgelöst. "
            "Schicht 2 gebaut? → python scripts/build_resolved.py --manifest-only"
        )
    if fehlend:
        # Kein Abbruch: ein Universum darf Symbole enthalten, die im Fenster
        # noch nicht (oder nicht mehr) handelten — genau das ist der Sinn eines
        # survivorship-freien Bestands. Aber es wird gesagt.
        log.warning(
            "%s: %d Symbole ohne Daten in %s..%s: %s",
            universe, len(fehlend), start, end, ", ".join(sorted(fehlend)),
        )

    rueckwaerts = {v: k for k, v in aufgeloest.items()}
    long_df, info = bulk_read.read_instruments(
        sorted(aufgeloest.values()), start, end, adjust=adjusted
    )
    if long_df.empty:
        raise RuntimeError(
            f"Schicht 2 kennt {len(aufgeloest)} Instrumente für {universe}, aber keine "
            "Kursdateien liegen. → python scripts/build_resolved.py --top 500"
        )

    if adjusted and not info.is_total_return:
        raise RuntimeError(
            f"{universe}: adjusted=True angefordert, aber die Reihen sind nicht "
            f"adjustierbar — {info.warning()} "
            "Entweder Splits/Dividenden für das Fenster laden oder adjusted=False "
            "setzen und die Rohreihe bewusst verwenden."
        )

    per_out: dict[str, pd.DataFrame] = {}
    for instrument, teil in long_df.groupby("instrument"):
        name = rueckwaerts.get(str(instrument), str(instrument))
        f = teil.copy()
        f["date"] = pd.to_datetime(f["date"])
        f = f.set_index("date").sort_index()
        per_out[name] = f[[c for c in _OHLCV if c in f.columns]]

    report = quality.check_universe(
        per_out, calendar=calendar, expected_start=start, expected_end=end
    )
    report.log()
    if not report.ok:
        errs = [(i.symbol, i.kind, i.detail) for i in report.issues if i.severity == "error"]
        raise ValueError(f"Data-Quality-Fehler im Universum {universe}: {errs}")

    combined = pd.concat(per_out, axis=1)
    combined.columns.names = ["symbol", "field"]
    combined = combined.sort_index().dropna(how="all")

    return MarketData(
        universe=universe,
        symbols=sorted(per_out),
        timeframe=timeframe,
        start=start,
        end=end,
        provider="eodhd",
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
            "Nutze provider='eodhd' oder installiere `pip install -e \".[data]\"`."
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


def equal_weight_benchmark(close: pd.DataFrame) -> pd.Series:
    """Gleichgewichteter Verlauf — über **Renditen**, nicht über Kurse.

    Der Mittelwert einer Kursmatrix ist kein gleichgewichteter Index, sondern
    ein **kurs**gewichteter (Dow-Logik): ein Papier zu 500 $ zählt zehnmal so
    viel wie eines zu 50 $. Solange die Zusammensetzung fest ist, ist das eine
    Ungenauigkeit im Namen.

    Sobald sie sich ändert, wird ein Fehler daraus. Tritt ein Papier zu 50 $
    einem Korb bei, der bei 100 $ steht, fällt der Mittelwert von 100 auf 75 —
    **ein Sprung von −25 % ohne dass sich ein einziger Kurs bewegt hat.** Für
    einen HMM-Fit ist das nicht Rauschen, sondern das stärkste Signal weit und
    breit: er lernt an jedem Rekonstitutionsdatum einen Regimewechsel, den es
    nie gab (#255).

    Über Renditen gibt es den Sprung nicht: gemittelt wird, was jedes Mitglied
    an diesem Tag *getan* hat, und ein Neuzugang trägt am Eintrittstag keine
    Rendite bei. ``NaN`` heißt „an diesem Tag kein Mitglied (oder kein Kurs)"
    und fällt aus dem Mittel heraus statt ihn zu verschieben.
    """
    returns = close.pct_change(fill_method=None)
    mittel = returns.mean(axis=1, skipna=True)

    # **Kein Mitglied ist nicht null Prozent.** `fillna(0.0)` machte aus „an
    # diesem Tag hatte kein Papier einen Kurs" eine Aussage über den Markt:
    # eine Strecke mit exakt 0 % Tagesrendite, also null Varianz, die der HMM
    # als besonders ruhiges Regime liest. Solche Tage haben keinen Wert, und
    # der eigene Fall dafür ist „nicht vorhanden", nicht „null".
    hat_mitglied = returns.notna().any(axis=1)
    mittel = mittel[hat_mitglied]
    if mittel.empty:
        return mittel.astype(float)
    return (1.0 + mittel.fillna(0.0)).cumprod()
