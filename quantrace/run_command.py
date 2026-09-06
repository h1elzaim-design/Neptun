"""Ein Lauf → ein CLI-Aufruf. Die einzige Übersetzung im Repo (#317).

**Warum das hier liegt und nicht bei einem der beiden Aufrufer.** Es gab diese
Funktion zweimal: in ``api/services/pipeline_runner.py`` für die Validierung und
den lokalen Lauf, und in ``worker/job_runner.py`` für den Lauf, der tatsächlich
rechnet. Die Worker-Fassung trug den Kommentar *„Kept in sync … both sides
change together"* — und war es nicht. Ihr fehlte die gesamte Behandlung von
``cost_model``.

Die Folge war nicht ein fehlendes Feature, sondern eine **stille Abweichung
zwischen protokollierter und gerechneter Annahme**: die API prüfte
``cost_model`` gegen ``COST_MODELS``, baute das Kommando — und verwarf es
(``# validate only``). Der Worker baute es neu, ohne ``--cost-model``, die CLI
nahm ihren Default ``flat``, und im Ergebnis-JSON stand ``per_asset_class``.
Ein Backtest, dessen Kostenannahme nicht die ist, die danebensteht, ist nicht
ungenau — er ist unbelegt.

Zwei Fassungen sind deshalb keine Redundanz, sondern eine Fehlerquelle mit
Anlauf: die zweite driftet, und der Unterschied fällt erst an einem Ergebnis
auf, das nicht zu dem passt, was der Run behauptet gerechnet zu haben.

**Diese Funktion ist rein.** Kein Lake-Zugriff, kein Vault, kein Environment —
was sie über die Welt wissen muss, bekommt sie übergeben:

* ``start``/``end`` **müssen** in ``params`` stehen. Wer ein Fenster erst
  auflösen muss, tut das vorher und schickt das Ergebnis mit. Vorher löste die
  API das Fenster auf, warf das Kommando weg und schickte die *rohen* Params —
  der Worker fiel dann auf sein eigenes ``QUANTRACE_DEFAULT_START`` zurück oder
  scheiterte an einem leeren Enddatum, obwohl die API erfolgreich validiert hatte.
* ``graph_has_param_space`` liest den Vault und kommt deshalb als Callback.
  Fehlt es, entfällt die Prüfung — der Aufrufer ohne Vault-Zugriff soll den
  Lauf nicht an einer Frage scheitern lassen, die er nicht beantworten kann.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

#: Präfix, mit dem eine Graph-Spec aus dem Vault statt einer Registry-Strategie
#: adressiert wird — z.B. "graph:my_breakout" (#188).
GRAPH_PREFIX = "graph:"

#: Strategien ohne Parameter. Ein Sweep über ein leeres Grid ist kein Sweep.
PARAMLESS_STRATEGIES: frozenset[str] = frozenset({"buy_and_hold"})

#: Die Modi, die diese Übersetzung kennt.
MODES: tuple[str, ...] = ("single", "sweep", "wf")


def command_for(
    strategy: str,
    universe: str,
    mode: str,
    params: dict[str, Any] | None,
    *,
    graph_has_param_space: Callable[[str], bool] | None = None,
) -> list[str]:
    """``{mode, strategy, universe, params}`` → argv für ``python -m quantrace``.

    ``params`` darf Steuer-Keys (``start``, ``end``, ``folds``, ``rank_by``,
    ``param_space``, ``cost_model``, ``delisting_return``) und — im
    Single-Mode — Strategie-Params enthalten. Steuer-Keys werden extrahiert,
    der Rest geht als ``--params`` JSON an die CLI.

    Als argv, nicht als Zeile: Quoting ist Sache dessen, der es anzeigt
    (``shlex.join``), nicht dieser Funktion.
    """
    p = dict(params or {})
    start = str(p.pop("start", None) or "")
    end = str(p.pop("end", None) or "")
    if not start or not end:
        raise ValueError(
            "Kein Zeitraum: start und end müssen im Lauf stehen. Wer sie aus dem "
            "Lake ableiten will, tut das vor dem Aufruf und schickt das Ergebnis "
            "mit — ein Fallback an dieser Stelle wäre eine zweite Wahrheit über "
            "das Fenster, in dem gerechnet wurde."
        )
    folds = str(p.pop("folds", 4))
    rank_by = p.pop("rank_by", None)
    param_space = p.pop("param_space", None)

    cost_model = p.pop("cost_model", None)
    if cost_model is not None:
        from quantrace.models import COST_MODELS

        if cost_model not in COST_MODELS:
            raise ValueError(
                f"Unbekanntes cost_model '{cost_model}' — erlaubt: {', '.join(COST_MODELS)}."
            )

    # Dieselbe Bauform wie ``cost_model``, und aus demselben Grund: eine
    # Annahme, die im Ergebnis-JSON steht, muss die CLI auch erreichen. Ohne
    # diesen Zweig fiele sie in ``--params`` und würde von ``StrategySpec``
    # wortlos geschluckt — genau der Weg, auf dem #317 entstanden ist.
    delisting_return = p.pop("delisting_return", None)
    if delisting_return is not None:
        try:
            delisting_return = float(delisting_return)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"delisting_return '{delisting_return}' ist keine Zahl."
            ) from exc
        if not -1.0 < delisting_return <= 0.0:
            raise ValueError(
                f"delisting_return {delisting_return} liegt ausserhalb von (-1, 0] — "
                "ein Delisting bringt weder mehr als den letzten Kurs noch weniger "
                "als nichts, und die -1 selbst ist ausgeschlossen: sie fuehrt zu "
                "einer Order zum Preis null. Totalverlust ist -0.999."
            )

    base = ["python", "-m", "quantrace"]
    # "graph:<slug>" → Graph-Spec aus dem Vault (#188), sonst Registry-Strategie.
    if strategy.startswith(GRAPH_PREFIX):
        selector = ["--graph-spec", strategy[len(GRAPH_PREFIX) :]]
    else:
        selector = ["--strategy", strategy]
    common = [*selector, "--universe", universe, "--start", start, "--end", end]
    annahmen = ["--cost-model", str(cost_model)] if cost_model else []
    if delisting_return is not None:
        annahmen += ["--delisting-return", str(delisting_return)]

    if mode == "single":
        cmd = [*base, "backtest", *common, *annahmen]
        if p:  # verbleibende Keys = Strategie-Params
            cmd += ["--params", json.dumps(p)]
        return cmd

    if mode in ("sweep", "wf"):
        if not strategy.startswith(GRAPH_PREFIX) and strategy in PARAMLESS_STRATEGIES:
            raise ValueError(
                f"Strategie '{strategy}' hat keine Parameter — {mode} ist sinnlos. "
                "Nutze Mode 'single'."
            )
        # Dieselbe Falle für Graph-Specs. Ihr Grid steht im Frontmatter der
        # Note, nicht in der Registry — fehlt es und kommt auch kein
        # `param_space` mit, bricht die CLI mit `typer.BadParameter` ab, also
        # mit **Exit-Code 2 und ohne verwertbare Meldung** im Run. Genau so sind
        # reihenweise `graph:*`-Sweeps gescheitert, zuletzt von einem
        # Agent-Lauf, der die Strategie selbst gebaut hatte.
        if (
            graph_has_param_space is not None
            and strategy.startswith(GRAPH_PREFIX)
            and not param_space
        ):
            slug = strategy[len(GRAPH_PREFIX) :]
            if not graph_has_param_space(slug):
                raise ValueError(
                    f"Graph-Spec '{slug}' hat kein `param_space` im Frontmatter — "
                    f"{mode} braucht ein Grid. Auf der Strategie-Seite unter "
                    f"'Sweep-Grid' ableiten oder setzen (/strategies/{slug}), "
                    "hier ein `param_space` mitgeben, oder für einen einzelnen "
                    "Lauf Mode 'single'."
                )
        sub = "sweep" if mode == "sweep" else "walkforward"
        cmd = [*base, sub, *common, *annahmen]
        if mode == "wf":
            cmd += ["--folds", folds]
        if param_space:  # explizites Grid (dict von Listen); sonst Registry-Default
            cmd += ["--params", json.dumps(param_space)]
        if rank_by:
            cmd += ["--rank-by", str(rank_by)]
        return cmd

    raise ValueError(f"Unknown mode: {mode}")
