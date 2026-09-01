"""Mitgliedschaft über die Zeit — ein Universum, das sich fortschreibt (#255).

Der offene Rest von ADR-015. Ein Regel-Universum ist bis hierher ein
**Schnappschuss**: die Regel lief einmal, zu einem Stichtag, und das Ergebnis
steht als flache ``symbols:``-Liste in der YAML. Korrekt für diesen Tag — und
danach zunehmend falsch, weil Neuemissionen nie hinzukommen. Wer 2000 screent
und bis 2009 backtestet, handelt neun Jahre lang einen Korb, den es nach 2000
nie wieder gab.

Das ist eine **andere** Verzerrung als Survivorship, aber eine. Survivorship
lässt die Toten weg; ein Schnappschuss lässt die Nachgeborenen weg. Beide
machen ein Universum unehrlich, und beide sieht man dem Ergebnis nicht an.

## Was dieses Modul ist

Die Leseseite. ``quantrace.screen.reconstitute`` erzeugt die Mitgliedschaft
(Regel alle N Monate neu auswerten), hier wird sie **gelesen und durchgesetzt**:

* ``from_universe_config`` — ``membership:`` aus der Universe-YAML
* ``symbols_on`` — wer gehört an Tag X dazu
* ``apply`` — der Kursrahmen, aber ohne Nichtmitglieder

## Die eine Regel

**Ein Papier ist an einem Tag handelbar oder es ist es nicht** — dazwischen
gibt es nichts. ``apply`` setzt alles außerhalb der Mitgliedschaft auf ``NaN``,
und zwar für *alle* OHLCV-Felder gemeinsam. Das ist bewusst dieselbe Form, die
der survivorship-freie Lake ohnehin liefert: ein Papier, das 2008 aufhört zu
handeln, hat danach keine Zeilen. Der Rechenpfad musste damit schon vorher
umgehen können — Mitgliedschaft erzeugt keinen neuen Sonderfall, sie benutzt
den vorhandenen.

Die Alternative wäre gewesen, je Periode einen eigenen Rahmen zu laden und die
Ergebnisse zu verketten. Das hätte an jeder Periodengrenze einen Bruch erzeugt
(Indikator-Rückblick, Positionsübergang, Kostenzuordnung) und für jeden dieser
Brüche eine Konvention gebraucht. Ein Rahmen mit Löchern hat diese Brüche
nicht.

## Was mit einem Titel passiert, der mitten in der Periode ausscheidet

**Nichts Besonderes, und das ist die Antwort, nicht die Ausrede.** Ein Papier,
das im Mai delistet wird, während seine Periode bis September läuft, hat ab Mai
keine Kurse mehr — genau wie im Lake. Die Strategie sieht ``NaN``, hält keine
Position, und die nächste Rekonstitution wählt es nicht mehr aus, weil der
Screen es zum Stichtag nicht mehr findet.

Wichtig ist, was **nicht** passiert: die Mitgliedschaft holt es nicht zurück
und schreibt ihm keinen Schlusskurs. Wer aus einem Delisting einen Erlös machen
will, braucht Corporate Actions, keine Universumsdefinition — und still eine
Null oder den letzten Kurs zu unterstellen, wäre die teure Variante: sie sieht
aus wie ein Ergebnis.

## Warum die flache ``symbols:``-Liste bleibt

Sie ist die **Vereinigung** über alle Perioden und damit die Liste dessen, was
geladen werden muss. Jeder bestehende Leser (Katalog, Kostenprüfung,
Fenster-Audit, ``quantrace fetch``) arbeitet unverändert weiter. Nur der
Backtest-Lesepfad kennt zusätzlich ``membership:`` — und das ist genau die
Stelle, an der der Unterschied zählt.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - nur für Typannotationen
    import pandas as pd

    from quantrace.models import MarketData

log = logging.getLogger(__name__)


class MembershipError(ValueError):
    """Fachlicher Fehler in einer Mitgliedschaft. Der Router macht daraus 409."""


def _norm(symbol: Any) -> str:
    """Ein Symbol, eine Schreibweise. Die Mitgliedschaft führt Großbuchstaben."""
    return str(symbol).strip().upper()


def _als_datum(wert: Any, feld: str) -> date:
    """YAML liefert ``date`` für unquoted Datumsangaben, ``str`` für quoted."""
    if isinstance(wert, date):
        return wert
    try:
        return date.fromisoformat(str(wert))
    except ValueError as exc:
        raise MembershipError(f"{feld}: '{wert}' ist kein Datum (YYYY-MM-DD).") from exc


@dataclass(frozen=True)
class MembershipPeriod:
    """Wer von wann bis wann dazugehört.

    ``end`` ist **exklusiv** und ``None`` für die letzte, offene Periode. Die
    Konvention ist nicht beliebig: mit inklusivem Ende gehörte der Rekonsti-
    tutionstag zu zwei Perioden, und ein Papier, das genau dort ausscheidet,
    wäre an diesem Tag gleichzeitig drin und draußen.
    """

    start: date
    end: date | None
    symbols: frozenset[str]

    def contains(self, day: date) -> bool:
        if day < self.start:
            return False
        return self.end is None or day < self.end


@dataclass(frozen=True)
class Membership:
    """Die vollständige Mitgliedschaft eines Universums über die Zeit.

    Die Perioden sind **lückenlos und überschneidungsfrei** — geprüft, nicht
    zugesagt. Eine Lücke wäre ein Zeitraum, in dem das Universum leer ist; das
    ist kein Backtest-Ergebnis von null Rendite, sondern eine kaputte Datei,
    und der Unterschied gehört gesagt, bevor jemand die Kurve anschaut.
    """

    periods: tuple[MembershipPeriod, ...]
    #: Wie oft neu geschirmt wurde, als Text fürs YAML (``"6M"``). Rein
    #: dokumentarisch — durchgesetzt wird ``periods``, nicht die Frequenz.
    frequency: str | None = None

    def __post_init__(self) -> None:
        if not self.periods:
            raise MembershipError("Eine Mitgliedschaft ohne Perioden ist keine.")
        vorher: MembershipPeriod | None = None
        for p in self.periods:
            if not p.symbols:
                raise MembershipError(
                    f"Periode ab {p.start} enthält kein Symbol. Ein leeres "
                    "Universum ist eine Aussage, die niemand treffen wollte — "
                    "eher ist die Regel zu eng oder der Lake zu dünn."
                )
            if p.end is not None and p.end <= p.start:
                raise MembershipError(
                    f"Periode ab {p.start} endet am {p.end} — das Ende liegt "
                    "nicht nach dem Anfang."
                )
            if vorher is not None:
                if vorher.end is None:
                    raise MembershipError(
                        "Nur die letzte Periode darf offen enden; hier folgt "
                        f"auf eine offene Periode noch eine ab {p.start}."
                    )
                if vorher.end != p.start:
                    raise MembershipError(
                        f"Lücke oder Überschneidung zwischen {vorher.end} und "
                        f"{p.start}. Die Mitgliedschaft muss lückenlos sein — "
                        "sonst gäbe es Tage ohne Universum, und ein leerer Tag "
                        "sieht im Ergebnis aus wie ein ruhiger."
                    )
            vorher = p

    # -- Lesen ---------------------------------------------------------------

    @property
    def start(self) -> date:
        return self.periods[0].start

    @property
    def end(self) -> date | None:
        return self.periods[-1].end

    @property
    def union(self) -> list[str]:
        """Alle je enthaltenen Symbole — das, was geladen werden muss.

        Sortiert, nicht nach Rang: über mehrere Stichtage hinweg hat ein
        Liquiditätsrang keine gemeinsame Bedeutung mehr.
        """
        alle: set[str] = set()
        for p in self.periods:
            alle |= set(p.symbols)
        return sorted(alle)

    def symbols_on(self, day: date) -> set[str]:
        """Wer gehört an diesem Tag dazu? Vor der ersten Periode: niemand."""
        for p in self.periods:
            if p.contains(day):
                return set(p.symbols)
        return set()

    # -- Durchsetzen ---------------------------------------------------------

    def tradable_frame(self, index: Any, symbols: list[str]) -> pd.DataFrame:
        """Boolesche Maske (Bar × Symbol): durfte gehalten werden?

        Flach, ein Eintrag je Symbol — nicht je OHLCV-Feld. Der Backtest fragt
        „gehört das Papier heute dazu", und diese Frage hat keine fünf
        Antworten.
        """
        import numpy as np
        import pandas as pd

        tage = pd.Index([ts.date() for ts in index])
        normiert = [_norm(s) for s in symbols]
        out = np.zeros((len(tage), len(symbols)), dtype=bool)
        for p in self.periods:
            zeile = tage >= p.start
            if p.end is not None:
                zeile &= tage < p.end
            mitglied = np.asarray([n in p.symbols for n in normiert])
            out |= np.asarray(zeile)[:, None] & mitglied[None, :]
        return pd.DataFrame(out, index=index, columns=symbols)

    def mask_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Kursrahmen mit Löchern: außerhalb der Mitgliedschaft ``NaN``.

        Nimmt **beide** Spaltenlayouts: den MultiIndex ``(symbol, field)`` aus
        ``MarketData`` und eine flache Spalte je Symbol, wie sie beim Pivot für
        den Regime-Benchmark entsteht. Der Unterschied ist keiner: in beiden
        Fällen ist die oberste Spaltenebene das Symbol, und genau die wird
        gefragt.

        **Verglichen wird normalisiert.** ``from_universe_config`` schreibt die
        Mitglieder in Großbuchstaben, die Spalten tragen dagegen den String aus
        ``symbols:`` wie er dasteht. Eine handgeschriebene YAML mit
        ``symbols: [spy, qqq]`` hätte sonst *nirgends* ein Mitglied, und der
        Fehler danach hieße „das Backtest-Fenster liegt neben der
        Mitgliedschaft" — eine Meldung, die auf die Datumsgrenzen zeigt,
        während der Fehler in der Schreibweise steckt.

        Symbole ohne jede Mitgliedschaft im Rahmen bleiben hier stehen — sie
        zu entfernen ist Sache von ``apply``.
        """
        import numpy as np
        import pandas as pd

        if frame.empty:
            return frame

        tage = pd.Index([ts.date() for ts in frame.index])
        zeilen: list[np.ndarray] = []
        for p in self.periods:
            m = tage >= p.start
            if p.end is not None:
                m &= tage < p.end
            zeilen.append(np.asarray(m))

        # Eine Maske über den ganzen Rahmen statt ein `.loc`-Schreibzugriff je
        # Symbol: bei 500 Symbolen × 20 Perioden × 5.000 Bars kostet die
        # Schleifenvariante Sekunden und zerlegt den Block-Manager in hunderte
        # Fragmente, die danach jeder `.xs`/`.mean` mitbezahlt.
        spalten = frame.columns.get_level_values(0)
        # Einmal normalisieren, nicht je Periode: bei 60 Perioden × 2.500
        # Spalten wären das 150.000 strip/upper-Aufrufe für 2.500 Antworten.
        normiert = [_norm(c) for c in spalten]
        erlaubt = np.zeros((len(frame.index), len(spalten)), dtype=bool)
        for p, m in zip(self.periods, zeilen, strict=True):
            mitglied = np.asarray([n in p.symbols for n in normiert])
            erlaubt |= m[:, None] & mitglied[None, :]
        return frame.where(pd.DataFrame(erlaubt, index=frame.index, columns=frame.columns))

    def mit_symbolen(self, je_periode: Sequence[Sequence[str]]) -> Membership:
        """Dieselben Perioden, andere Symbolnamen (#311).

        Gebraucht, sobald ein Kürzel über die Perioden hinweg auf verschiedene
        Papiere zeigt: ``resolve.resolve_membership`` macht daraus ``ACL__S1``
        und ``ACL__S2``, und die Mitgliedschaft muss dieselben Namen benutzen —
        sonst maskiert sie Spalten, die es nicht mehr gibt, und der Korb wäre
        an jedem Tag leer.

        Die Zeitgrenzen bleiben unangetastet; nur die Namen ändern sich.
        """
        if len(je_periode) != len(self.periods):
            raise MembershipError(
                f"{len(je_periode)} Symbollisten für {len(self.periods)} Perioden — "
                "die Umbenennung muss Periode für Periode passen."
            )
        neu = tuple(
            MembershipPeriod(start=p.start, end=p.end, symbols=frozenset(_norm(s) for s in syms))
            for p, syms in zip(self.periods, je_periode, strict=True)
        )
        return Membership(periods=neu, frequency=self.frequency)

    def apply(self, md: MarketData) -> MarketData:
        """``MarketData`` auf die Mitgliedschaft beschneiden.

        Drei Dinge passieren, und alle drei sind sichtbar:

        1. Kurse außerhalb der Mitgliedschaft werden ``NaN``.
        2. Symbole, die im geladenen Fenster **nie** Mitglied sind, fliegen
           samt Spalten raus — sonst stünden sie in ``md.symbols`` und jede
           Gleichgewichtung teilte Kapital auf Papiere auf, die nicht im
           Universum sind. **Nur diese**: eine leere Spalte kann auch ein
           Mitglied ohne Lake-Daten sein, und das ist eine Datenlücke, keine
           Aussage über die Mitgliedschaft. Beides in einen Topf zu werfen
           hieße, eine Lücke als Absicht zu protokollieren.
        3. ``content_hash`` wird neu gerechnet, und zwar **erzwungen**:
           ``model_post_init`` hasht Form, ersten und letzten Index und die
           letzte Zeile. Wächst der Korb monoton (die letzte Periode enthält
           alles), sind die alle vier identisch — der maskierte Rahmen trüge
           die Kennung des unmaskierten, und zwei verschiedene Läufe wären im
           Vault nicht auseinanderzuhalten.
        """
        import hashlib

        from quantrace.models import MarketData as _MarketData

        maskiert = self.mask_frame(md.frame)
        if maskiert.empty:
            return md

        spalten = list(dict.fromkeys(maskiert.columns.get_level_values(0)))
        nie_mitglied = {
            sym for sym in spalten if not any(_norm(sym) in p.symbols for p in self.periods)
        }
        ohne_daten = {
            sym
            for sym in spalten
            if sym not in nie_mitglied and bool(maskiert[sym].isna().to_numpy().all())
        }
        if ohne_daten:
            log.warning(
                "%s: %d Mitglieder tragen im Fenster %s..%s keine Kurse: %s",
                md.universe, len(ohne_daten), md.start, md.end, ", ".join(sorted(ohne_daten)),
            )
        leer = sorted(nie_mitglied | ohne_daten)
        if nie_mitglied:
            log.info(
                "%s: %d Symbole sind im Fenster nie Mitglied: %s",
                md.universe, len(nie_mitglied), ", ".join(sorted(nie_mitglied)),
            )
        if leer:
            maskiert = maskiert.drop(columns=leer, level=0)

        entfernt = set(leer)
        uebrig = [s for s in md.symbols if s not in entfernt]
        if not uebrig:
            raise MembershipError(
                f"Kein Symbol aus {md.universe} ist zwischen {md.start} und "
                f"{md.end} Mitglied. Die Mitgliedschaft läuft von "
                f"{self.start} bis {self.end or 'offen'} — das Backtest-Fenster "
                "liegt daneben."
            )

        # Der Hash über den beschnittenen Inhalt — nicht der geerbte. Das
        # Standardverfahren aus `model_post_init` sieht die Maske im Inneren
        # des Rahmens nicht, deshalb hier explizit über die NaN-Struktur.
        digest = hashlib.sha256(md.content_hash.encode())
        digest.update(maskiert.isna().to_numpy().tobytes())
        digest.update(repr([(p.start, p.end, sorted(p.symbols)) for p in self.periods]).encode())

        # Die Mitgliedschaft als eigene Maske, nicht aus NaN erschlossen: der
        # Backtest muss ein Ausscheiden (verkaufen) von einer Handelsaussetzung
        # (durchhalten) unterscheiden können, und in den Kursen sehen beide
        # gleich aus. Siehe `backtest_runner._close_untradable`.
        handelbar = _tradable_frame(self, maskiert)

        return _MarketData(
            universe=md.universe,
            symbols=uebrig,
            timeframe=md.timeframe,
            start=md.start,
            end=md.end,
            provider=md.provider,
            adjusted=md.adjusted,
            calendar=md.calendar,
            cost_class=md.cost_class,
            frame=maskiert,
            tradable=handelbar,
            content_hash=digest.hexdigest()[:16],
        )

def _tradable_frame(m: Membership, maskiert: pd.DataFrame) -> pd.DataFrame:
    """Die Handelbarkeitsmaske zum beschnittenen Rahmen."""
    symbole = list(dict.fromkeys(maskiert.columns.get_level_values(0)))
    return m.tradable_frame(maskiert.index, symbole)


def from_universe_config(cfg: dict[str, Any]) -> Membership | None:
    """``membership:`` aus einer Universe-YAML lesen — oder ``None``.

    ``None`` heißt: flaches Universum, jedes Symbol gilt über den ganzen
    Zeitraum. Das ist der Normalfall und kein Mangel.

    **Geprüft wird gegen die flache Liste.** Ein Mitglied, das nicht unter
    ``symbols:`` steht, wird nie geladen — es fehlte dann im Backtest, ohne
    dass irgendwo etwas rot würde. Die Vereinigung muss deshalb eine Teilmenge
    von ``symbols:`` sein, sonst ist die Datei kaputt und sagt es.
    """
    roh = cfg.get("membership")
    if not roh:
        return None
    if not isinstance(roh, list):
        raise MembershipError("`membership:` muss eine Liste von Perioden sein.")

    perioden: list[MembershipPeriod] = []
    for i, eintrag in enumerate(roh):
        if not isinstance(eintrag, dict):
            raise MembershipError(f"membership[{i}] ist kein Abschnitt mit from/to/symbols.")
        if "from" not in eintrag:
            raise MembershipError(f"membership[{i}] hat kein `from`.")
        bis = eintrag.get("to")
        symbole = eintrag.get("symbols") or []
        if not isinstance(symbole, list):
            raise MembershipError(f"membership[{i}].symbols ist keine Liste.")
        perioden.append(
            MembershipPeriod(
                start=_als_datum(eintrag["from"], f"membership[{i}].from"),
                end=None if bis in (None, "null", "") else _als_datum(bis, f"membership[{i}].to"),
                symbols=frozenset(_norm(s) for s in symbole if str(s).strip()),
            )
        )

    frequenz = ((cfg.get("construction") or {}).get("rebalance")) or None
    m = Membership(tuple(perioden), frequency=str(frequenz) if frequenz else None)

    flach = {_norm(s) for s in (cfg.get("symbols") or [])}
    if flach:
        fehlend = sorted(set(m.union) - flach)
        if fehlend:
            raise MembershipError(
                f"{len(fehlend)} Mitglieder stehen nicht unter `symbols:`: "
                f"{', '.join(fehlend[:8])}{' …' if len(fehlend) > 8 else ''}. "
                "Die flache Liste ist die Ladeliste — was dort fehlt, wird nie "
                "geladen und fehlt im Backtest, ohne dass etwas rot wird."
            )
    return m


__all__ = [
    "Membership",
    "MembershipError",
    "MembershipPeriod",
    "from_universe_config",
]
