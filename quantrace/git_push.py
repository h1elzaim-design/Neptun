"""Was beide Push-Pfade brauchen — und was einer davon am 2026-09-07 nicht hatte.

Zwei Stellen im Projekt pushen nach GitHub: `api/services/vault_writer.py`
(Notes aus der Webapp) und `worker/job_runner.py` (Backtest-Ergebnisse aus dem
Azure-Container). Beide klonen beim Start, beide pushen Minuten bis Stunden
später auf `main`, und beide treffen deshalb denselben Normalfall: **`main` ist
inzwischen weitergelaufen.**

Der Vault-Writer hat das seit dem 2026-08-30 gelernt und zieht nach. Der Worker
hatte es nicht — mit der Folge, dass am 2026-09-07 zwei Walk-Forward-Läufe
rechneten, ihre Notes schrieben, committeten und der Push abprallte, weil in
derselben Minute drei andere Läufe auf `main` landeten. Der Container setzt sich
beim nächsten Start auf `main` zurück; die Ergebnisse waren damit weg.

Zwei Funktionen, weil genau zwei Dinge geteilt gehören:

* **`ist_abgelehnt`** — die Frage, ob ein Push-Fehler ein Nachziehen verdient.
  Ein Token- oder Netzwerkfehler darf hier nicht treffen: er wird durch ein
  Rebase nicht besser, und das Rebase verschleierte nur die Ursache.
* **`redigiere`** — der Token aus der Fehlermeldung. Git schreibt die
  Remote-URL in jede Fehlerzeile, und die URL trägt das `GITHUB_TOKEN`. Beim
  Worker landete diese Zeile über `JobOutcome.error` in der Spalte `runs.error`
  und von dort in die Laufansicht der Webapp — ein gültiges Push-Token, in der
  Datenbank und im Browser, lesbar für jeden, der den Lauf öffnet.
"""

from __future__ import annotations

import re

__all__ = ["ist_abgelehnt", "redigiere"]


# `https://x-access-token:ghp_…@github.com/…` — der Teil zwischen Schema und
# `@` ist das Geheimnis. Bewusst über *jedes* Schema und *jeden* Host, nicht
# nur github.com: eine Meldung, die einen Token durchlässt, weil der Host
# unbekannt war, ist kein Schutz.
_MIT_ZUGANGSDATEN = re.compile(r"(?P<schema>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s@]+@")

# Ein loser Token, der ohne URL in einer Meldung steht (git nennt ihn in
# manchen Versionen im Klartext). Die Präfixe sind GitHubs eigene.
_LOSER_TOKEN = re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,})")


def redigiere(text: str) -> str:
    """Zugangsdaten aus einer Meldung entfernen, bevor sie irgendwo landet.

    Angewandt wird sie **an der Quelle** — dort, wo aus einem Git-Fehler ein
    `RuntimeError` wird —, nicht erst beim Anzeigen. Sonst gibt es weiter einen
    Pfad, auf dem der Token in ein Log, eine Datenbankspalte oder eine
    Fehler-Mail rutscht, und der wird beim nächsten Umbau vergessen.
    """
    ohne_url = _MIT_ZUGANGSDATEN.sub(r"\g<schema>***@", text)
    return _LOSER_TOKEN.sub("***", ohne_url)


def ist_abgelehnt(meldung: str) -> bool:
    """Ob eine Push-Ablehnung daher kommt, dass der Remote weiter ist.

    Git sagt das in Prosa und je nach Version verschieden; geprüft werden
    deshalb die Wendungen, die alle Varianten teilen.
    """
    m = meldung.lower()
    return any(
        hinweis in m
        for hinweis in ("non-fast-forward", "fetch first", "rejected", "behind its remote")
    )
