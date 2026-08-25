"""ISIN als Lake-Schlüssel für den US-Aktienmarkt.

**Warum nicht der Ticker.** Am 2026-08-10 gegen EODHD geprüft:

    tot     BBBY_old   Bed Bath & Beyond Inc     ISIN US0758961009
    tot     BBBYQ      Bed Bath & Beyond Inc.    ISIN US0758961009
    aktiv   BBBY       Bed Bath & Beyond, Inc.   ISIN US6903701018   ← andere Firma

Overstock.com kaufte die Marke aus der Insolvenzmasse und benannte sich um.
Ticker **und** Name gingen über; die drei Einträge unterscheiden sich um ein
Komma. Ein Lake, der nach ``symbol=BBBY`` partitioniert, legt zwei Firmen in
eine Datei — und der Totalverlust der einen wird durch den Kursverlauf der
anderen ersetzt. Ein Backtest verbucht dort einen Gewinn, wo Anleger alles
verloren, und nichts wirft eine Fehlermeldung.

**Warum das hier nur ISINs kennt.** Crypto läuft als eigener Track mit eigener
Ablage; BTCUSD hat keine ISIN und braucht auch keine. Einen gemeinsamen
Namensraum für beide zu bauen hieße, eine Vereinigung vorzubereiten, die
bewusst nicht stattfindet. Die Lake-Präfixe bleiben deshalb getrennt::

    us_equities/isin=US78462F1030/data.parquet    ← dieses Modul
    raw/symbol=BTCUSD/data.parquet                ← Crypto, unverändert

Getrennte Präfixe heißen auch getrennte Hive-Schemata: DuckDB kann jedes für
sich lesen, ohne über eine gemischte Partitionierung zu stolpern.

**Was dieses Modul NICHT tut.** Es löst keine Ticker auf. Das Nachschlagen
``SPY → welche ISIN?`` ist eine Entscheidung mit Annahme — einmal getroffen,
überprüft und in ``data/instruments.yaml`` festgeschrieben, nicht zur Laufzeit
geraten. Zur Laufzeit zu raten wäre genau die Bequemlichkeit, die `BBBY` in den
Lake gelassen hätte.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

#: Zwei Buchstaben Land, neun alphanumerische Zeichen, eine Prüfziffer.
_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

#: Lake-Präfix des US-Aktien-Tracks. Getrennt von `raw/`, wo Tiingo-Daten und
#: Crypto nach Symbol liegen — zwei Partitionsschemata unter einer Wurzel
#: würden jede Hive-Abfrage brechen.
US_EQUITY_PREFIX = "us_equities"

#: Corporate Actions, je eine eigene Wurzel aus demselben Grund. Sie liegen
#: neben den Kursen und nicht darin, weil sie eine andere Kardinalität haben:
#: an einem Handelstag gibt es 26.000 Kurse, aber nur eine Handvoll Splits.
US_SPLITS_PREFIX = "us_equity_splits"
US_DIVIDENDS_PREFIX = "us_equity_dividends"


def isin_checksum_ok(isin: str) -> bool:
    """Luhn-Prüfziffer einer ISIN.

    Fängt Tippfehler und abgeschnittene Werte ab — nicht erfundene. Eine ISIN
    mit gültiger Prüfziffer kann trotzdem zu keinem Papier gehören; eine mit
    ungültiger gehört garantiert zu keinem. Das ist billige Sicherheit an einer
    Stelle, wo ein stiller Fehler eine ganze Kursreihe falsch zuordnet.
    """
    if not _ISIN_RE.match(isin):
        return False
    # Buchstaben zu Zahlen (A=10 … Z=35), dann Luhn über die Ziffernfolge.
    digits = "".join(str(int(c, 36)) for c in isin[:-1])
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10 == int(isin[-1])


def normalise_isin(raw: str) -> str:
    """Rohwert → geprüfte ISIN in Großbuchstaben. Wirft mit Klartext.

    Streng mit Absicht: eine durchgerutschte Falscheingabe erzeugt eine
    Partition, die aussieht wie ein Wertpapier und keins ist.
    """
    isin = str(raw).strip().upper().replace(" ", "")
    if not _ISIN_RE.match(isin):
        raise ValueError(
            f"{raw!r} hat nicht die Form einer ISIN "
            "(2 Buchstaben Land + 9 alphanumerisch + 1 Prüfziffer)"
        )
    if not isin_checksum_ok(isin):
        raise ValueError(f"ISIN {isin}: Prüfziffer stimmt nicht — Tippfehler?")
    return isin


def isin_partition(isin: str) -> str:
    """Partitionsname im Lake, z. B. ``isin=US78462F1030``."""
    return f"isin={normalise_isin(isin)}"


def isin_from_partition(name: str) -> str:
    """Umkehrung von `isin_partition` — fürs Einlesen bestehender Ablagen."""
    bare = name.removeprefix("isin=")
    if bare == name:
        raise ValueError(f"{name!r} ist kein ISIN-Partitionsname (erwartet 'isin=…')")
    return normalise_isin(bare)


#: Die geprüfte Ticker→ISIN-Karte, erzeugt von `scripts/build_instrument_map.py`
#: und im Repo versioniert. Siehe dort, warum sie nicht zur Laufzeit entsteht.
INSTRUMENT_MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "instruments.yaml"


@lru_cache(maxsize=1)
def load_instrument_map() -> dict[str, str]:
    """Ticker → ISIN aus `data/instruments.yaml`.

    Nur Einträge **ohne** offene `review`-Markierung. Ein ungeprüfter Fall ist
    keine Zuordnung, sondern eine Frage — ihn hier durchzulassen hieße, die
    Prüfung zur Zierde zu machen.

    Leeres dict, wenn die Datei fehlt. Der Aufrufer merkt das an einem leeren
    Ergebnis; eine Ausnahme wäre hier zu scharf, weil der US-Track optional ist
    und Crypto ohne diese Karte läuft.

    **Dasselbe gilt für eine kaputte Datei**, und das war bis 2026-08-12 nicht
    so. Der fehlende Fall war abgefangen, der defekte nicht — ein YAML-Fehler
    flog als Ausnahme durch, und zwar aus ``_isin_candidate`` heraus, also
    **nach** dem 20-Minuten-Read von 19,6 Mio Code-Tagen. Die ganze Arbeit weg,
    wegen einer Datei, ohne die der Lauf ohnehin weiterlaufen sollte.

    Die Absicht stand im Docstring darüber („eine Ausnahme wäre hier zu
    scharf"); gebaut war nur die Hälfte davon.
    """
    if not INSTRUMENT_MAP_PATH.exists():
        return {}

    import yaml

    try:
        data = yaml.safe_load(INSTRUMENT_MAP_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 — YAML wirft ein halbes Dutzend Typen
        log.warning(
            "Instrument-Karte %s ist nicht lesbar (%s). Es gibt damit KEINE "
            "belegten ISINs — die Schlüssel werden synthetisch. Der Lauf geht "
            "weiter; die Datei gehört repariert.",
            INSTRUMENT_MAP_PATH,
            exc,
        )
        return {}

    if not isinstance(data, dict):
        # Eine YAML-Liste oder ein nackter Skalar parst fehlerfrei und hat
        # trotzdem kein `instruments:`. `.get` würde hier mit AttributeError
        # abbrechen — ein Formatfehler, der wie ein Programmfehler aussieht.
        log.warning(
            "Instrument-Karte %s hat kein Mapping auf oberster Ebene (%s). "
            "Ohne `instruments:` gibt es keine belegten ISINs.",
            INSTRUMENT_MAP_PATH,
            type(data).__name__,
        )
        return {}

    out: dict[str, str] = {}
    for symbol, meta in (data.get("instruments") or {}).items():
        if not isinstance(meta, dict) or meta.get("review") or not meta.get("isin"):
            continue
        try:
            out[str(symbol).strip().upper()] = normalise_isin(str(meta["isin"]))
        except ValueError:
            # Eine kaputte ISIN in der Karte ist ein Fehler, der auffallen
            # muss — aber nicht durch einen Absturz beim Import. Der `--check`
            # des Skripts ist die Stelle, die ihn meldet.
            continue
    return out


def isin_for(symbol: str) -> str | None:
    """ISIN eines Tickers laut geprüfter Karte, oder ``None``.

    ``None`` heißt „nicht zugeordnet oder nicht geprüft" — nie „gibt es nicht".
    Wer daraufhin auf den Ticker zurückfällt, hat die Trennung aufgegeben.
    """
    return load_instrument_map().get(symbol.strip().upper())


__all__ = [
    "INSTRUMENT_MAP_PATH",
    "US_DIVIDENDS_PREFIX",
    "US_EQUITY_PREFIX",
    "US_SPLITS_PREFIX",
    "isin_for",
    "isin_checksum_ok",
    "isin_from_partition",
    "isin_partition",
    "load_instrument_map",
    "normalise_isin",
]
