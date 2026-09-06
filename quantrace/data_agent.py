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
from collections.abc import Mapping
from datetime import date, timedelta
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

    explizit = provider is not None
    provider = provider or default_provider()
    if provider == "eodhd":
        md, membership = _load_via_bulk(
            universe, symbols, start, end, timeframe, adjusted, calendar, membership
        )
    elif provider == "tiingo":
        # Eigener Zweig statt „unbekannter Provider": `tiingo` ist kein
        # Tippfehler, sondern ein Rest von vor dem 2026-08-13.
        #
        # **Und woher der Wert kam, gehört genauso hinein.** Die Meldung zeigte
        # bis zum 2026-08-27 auf die Universe-YAML — über `sweep`, `backtest`
        # und `walk-forward` kann sie von dort aber gar nicht stammen:
        # `_universe_data` reicht `provider` nicht durch, der Wert kommt also
        # aus `default_provider()` und damit aus `QUANTRACE_DATA_PROVIDER`.
        # Genau der Fall trat ein: alle zwölf YAMLs sagten längst `eodhd`, die
        # `.env` trug den toten Provider, und gesucht wurde in den Dateien, die
        # recht hatten.
        woher = (
            "als Argument übergeben"
            if explizit
            else "aus QUANTRACE_DATA_PROVIDER (Umgebung oder .env im Repo-Verzeichnis)"
        )
        raise ValueError(
            f"provider='tiingo' ist entfernt — hier {woher}, Universum "
            f"{universe!r}. Der Backtest-Pfad ist der EODHD-Bulk: den Wert auf "
            "`eodhd` setzen oder die Zeile löschen, `eodhd` ist der Default. "
            "Tiingo war eine Survivor-Liste; ein Backtest über 2008 sah dort "
            "nie LEH, WM oder BSC."
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


def _unbrauchbare_symbole(report: quality.QualityReport) -> dict[str, str]:
    """Symbole mit ``error``-Befund → Grund. Sie fliegen raus, statt alles zu stoppen.

    **Warum nicht mehr abgebrochen wird** (#322). Ein Data-Quality-Fehler ist
    eine Aussage über *ein* Symbol; der Abbruch machte daraus eine über das
    ganze Universum. Am 2026-09-05 kostete das genau einen Bar: `PNLYY` trägt
    am 2011-06-08 ``high=9,74`` bei ``low=10,00`` — high und low vertauscht,
    ein schlechter Tick von EODHD. Damit war **jeder** Backtest über
    `us_top500_liquid` unmöglich, dessen Fenster dieses Datum enthält, für
    alle 1.934 Kürzel und alle Strategien.

    Das skaliert falsch herum: ein survivorship-freies Universum über zwanzig
    Jahre enthält mit Sicherheit weitere solche Ticks, und je länger das
    Fenster, desto sicherer der Abbruch. Der Querschnitts-Backtest aus #300
    wäre daran nie angekommen.

    **Die Strenge bleibt, die Reichweite nicht.** ``high < low`` bleibt ein
    Fehler und kein Warnhinweis — wer das durchlässt, rechnet jeden
    Range-Indikator still auf Unsinn. Das Symbol wird deshalb entfernt und
    nicht repariert: welcher der vier Werte der falsche ist, weiss niemand,
    und ein geratener Kurs ist teurer als ein fehlender.

    **Und der Rauswurf reist mit.** Er steht in ``MarketData.unusable_symbols``
    und damit im Ergebnis, dieselbe Bauform wie ``missing_symbols`` (#307).
    Ein Ausschluss, der nur im Log steht, ist ein Ausschluss, den beim Lesen
    der Kennzahlen niemand mehr vor Augen hat.
    """
    aus: dict[str, str] = {}
    for i in report.issues:
        if i.severity != "error":
            continue
        grund = f"{i.kind}: {i.detail}"
        aus[i.symbol] = f"{aus[i.symbol]} | {grund}" if i.symbol in aus else grund
    return aus


def _erloes_annahmen(
    combined: pd.DataFrame, aufgeloest: Mapping[str, str], end: date
) -> dict[str, float]:
    """Wo der Katalog dem Rahmen widerspricht: „aufgehört" ist nicht „gestorben".

    Endet die Reihe eines Symbols vor dem letzten Bar des Rahmens, verkauft der
    Backtest zwangsweise und bucht dabei den Delisting-Abschlag
    (`backtest_runner._close_untradable`). Für die meisten dieser Symbole ist
    das richtig — der Lake ist survivorship-frei, und ein Papier, das aufhört,
    hat aufgehört.

    Für einige ist es das nicht: die Karte führt für ihr Instrument Kurse
    **nach** dem Fenster. Dann endete nicht das Papier, sondern die Abdeckung —
    eine fehlende Partition, eine lange Handelsaussetzung, ein Kürzel, das im
    Fenster ruhte. Diesen Symbolen einen Insolvenz-Abschlag aufzubuchen, wäre
    dieselbe Sorte Fehler wie der, den `delisting_return` behebt, nur mit
    umgekehrtem Vorzeichen.

    **Nur der widerlegte Fall wird eingetragen.** Wo die Karte schweigt oder
    zustimmt, bleibt es bei der Vorgabe aus `BacktestConfig` — ein Eintrag hier
    ist eine Aussage über ein Papier, kein Platzhalter für alle.
    """
    from quantrace import resolve

    if combined.empty or "close" not in set(combined.columns.get_level_values("field")):
        return {}
    close = combined.xs("close", level="field", axis=1)
    letzter_bar = close.index[-1]
    abbricht = [
        str(sym)
        for sym in close.columns
        if (ende := close[sym].last_valid_index()) is not None and ende < letzter_bar
    ]
    if not abbricht:
        return {}

    # Anzeigename → Instrument, aber nur für die Symbole, bei denen es eine
    # Rolle spielt. Ein LIST über die ganze Karte für 1.900 Symbole zu filtern,
    # von denen zwanzig abbrechen, wäre Arbeit ohne Frage.
    kandidaten = {aufgeloest[s]: s for s in abbricht if s in aufgeloest}
    if not kandidaten:
        return {}
    lebend = resolve.mit_kursen_nach(kandidaten, end)
    if lebend:
        log.info(
            "%d Symbole enden im Rahmen, führen laut Karte aber nach %s noch Kurse — "
            "für sie gilt kein Delisting-Abschlag: %s",
            len(lebend),
            end,
            ", ".join(sorted(kandidaten[i] for i in lebend)[:10]),
        )
    return {kandidaten[i]: 0.0 for i in lebend}


def _load_via_bulk(
    universe: str,
    symbols: list[str],
    start: date,
    end: date,
    timeframe: Timeframe,
    adjusted: bool,
    calendar: str | None = None,
    membership: Membership | None = None,
) -> tuple[MarketData, Membership | None]:
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

    if membership is not None:
        # **Je Mitgliedschaftsperiode auflösen, nicht über das ganze Fenster.**
        # Über zwanzig Jahre hat ein survivorship-freies Universum mit
        # Sicherheit recycelte Kürzel: am 2026-09-01 trugen 75 der 1.934 Ticker
        # in `us_top500_liquid` mehr als ein Segment, und ein einziges davon
        # (ACL — Alcon bis 2011, danach ein fremdes Papier) brach den gesamten
        # Load ab. Das Universum weiss aber, *wann* ein Ticker Mitglied war,
        # und innerhalb einer Periode ist er eindeutig.
        aufloesung = resolve.resolve_membership(
            [
                (p.start, (p.end - timedelta(days=1)) if p.end else end, sorted(p.symbols))
                for p in membership.periods
            ]
        )
        aufgeloest = aufloesung.zuordnung
        fehlend = aufloesung.fehlend
        membership = membership.mit_symbolen(aufloesung.perioden_symbole)
        if aufloesung.epochen:
            # Kein Fehler, aber eine Aussage über die Daten: unter diesem
            # Kürzel handelten verschiedene Papiere, und sie bleiben getrennt.
            log.warning(
                "%s: %d Kürzel zeigen über die Perioden auf verschiedene Papiere und "
                "werden getrennt geführt: %s",
                universe,
                len(aufloesung.epochen),
                ", ".join(f"{k} → {'/'.join(v)}" for k, v in list(aufloesung.epochen.items())[:5]),
            )
        if aufloesung.mehrdeutig:
            log.warning(
                "%s: %d Kürzel blieben innerhalb ihrer Periode mehrdeutig und fehlen dort: %s",
                universe,
                len(aufloesung.mehrdeutig),
                ", ".join(sorted(aufloesung.mehrdeutig)[:5]),
            )
    else:
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
    unbrauchbar = _unbrauchbare_symbole(report)
    # **Der Adjustierungs-Ausschluss gehört in dieselbe Liste** (#312). Ein
    # Instrument, dessen Aktionseintrag keinen positiven Preisfaktor ergibt,
    # fällt in `bulk_read._apply_actions` heraus — bis hierher kam davon nur
    # eine Log-Zeile an, und im Ergebnis sah das Symbol aus wie eines, das im
    # Fenster nie gehandelt hat. Es ist das Gegenteil: es hat gehandelt, und
    # die Reihe ist unbrauchbar. Zurückübersetzt auf den Universumsnamen,
    # weil `unadjustable` auf Instrumente schlüsselt.
    for instrument, grund in info.unadjustable.items():
        unbrauchbar.setdefault(rueckwaerts.get(str(instrument), str(instrument)), grund)
    if unbrauchbar:
        for sym in unbrauchbar:
            per_out.pop(sym, None)
        log.error(
            "%s: %d Symbole wegen Data-Quality-Fehlern ausgeschlossen — gerechnet "
            "wird ohne sie: %s",
            universe,
            len(unbrauchbar),
            "; ".join(f"{s} ({g})" for s, g in sorted(unbrauchbar.items())),
        )
    if not per_out:
        raise ValueError(
            f"Kein Symbol aus {universe} ist im Fenster {start}..{end} brauchbar. "
            f"Data-Quality-Fehler: {unbrauchbar}"
        )

    combined = pd.concat(per_out, axis=1)
    combined.columns.names = ["symbol", "field"]
    combined = combined.sort_index().dropna(how="all")

    md = MarketData(
        universe=universe,
        symbols=sorted(per_out),
        # Bis hierher stand die Lücke nur im Log (`log.warning` oben). Von dort
        # kam sie nie bis zum Ergebnis, und ein Backtest über 15 statt 16
        # Symbole sah aus wie einer über das ganze Universum (#307).
        missing_symbols=sorted(fehlend),
        # Getrennt von `missing_symbols`, weil es etwas anderes heisst: dort
        # lag nichts, hier lag etwas Kaputtes. Wer die Zahlen später liest,
        # muss „nie geladen" von „geladen und verworfen" unterscheiden können.
        unusable_symbols=unbrauchbar,
        timeframe=timeframe,
        start=start,
        end=end,
        provider="eodhd",
        adjusted=adjusted,
        calendar=calendar or DEFAULT_CALENDAR,
        frame=combined,
        delisting_returns=_erloes_annahmen(combined, aufgeloest, end),
    )
    # Die Mitgliedschaft kommt mit zurück: sie kann oben auf Anzeigenamen
    # umgeschrieben worden sein, und der Aufrufer maskiert damit.
    return md, membership


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
