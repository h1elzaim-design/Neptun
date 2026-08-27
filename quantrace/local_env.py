"""Die lokale ``.env`` laden — für alles, was man von Hand aufruft.

**Warum das nötig ist, obwohl es nach Kleinigkeit aussieht.** Keine Zeile in
``.env.example`` trägt ein ``export``. Ein ``source .env`` setzt damit
*Shell*-Variablen, keine *Umgebungs*variablen, und Python sieht in
``os.environ`` nichts. Richtig wäre ``set -a; source .env; set +a`` — das muss
man aber wissen.

Der Fehler, den das produziert, ist der teure: nicht „Token fehlt", sondern
ein **leerer Lake**. ``storage._lake_root()`` liest ``QUANTRACE_DATA_LAKE``,
findet nichts, fällt auf das lokale Verzeichnis zurück, und jeder Coverage-Read
meldet wahrheitsgemäß null Partitionen. Am 2026-08-13 genau so passiert.

**Warum das hier im Paket liegt und nicht mehr nur unter ``scripts/``.** Die
Datenskripte machten es seit jeher, die CLI nicht — und `run-local` scheiterte
deshalb mit „Kein Token", während der Wert die ganze Zeit in der ``.env``
stand. Zwei Aufrufwege in dasselbe Repo, von denen nur einer die Datei kennt,
sind genau die Sorte Unterschied, die niemand vermutet. ``scripts/_env.py``
re-exportiert von hier, damit es nur eine Fassung gibt.

Bereits gesetzte Umgebungsvariablen gewinnen (``override=False`` ist der
Default von ``load_dotenv``): wer explizit ``EODHD_API_KEY=… python …`` davor
schreibt, meint das auch so.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_local_env(path: Path | None = None) -> bool:
    """Lädt ``<repo>/.env``. Gibt zurück, ob etwas geladen wurde.

    Fehlt ``python-dotenv`` oder die Datei, passiert nichts — beides ist ein
    zulässiger Zustand (CI hat echte Env-Vars, kein ``.env``), und eine
    Ausnahme hier würde einen Aufruf blockieren, der ohne die Datei laufen
    kann.

    Gesucht wird zuerst neben dem Repo, dann im Arbeitsverzeichnis: bei einer
    nicht-editierbaren Installation zeigt ``__file__`` in ``site-packages``,
    und dort liegt keine ``.env``.
    """
    kandidaten = [path] if path else [REPO_ROOT / ".env", Path.cwd() / ".env"]
    for ziel in kandidaten:
        if ziel is None or not ziel.exists():
            continue
        try:
            from dotenv import load_dotenv
        except ImportError:  # pragma: no cover - dotenv ist optional
            return False
        if load_dotenv(ziel):
            return True
    return False
