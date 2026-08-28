"""Der Transport — MCP über stdio. Alles Inhaltliche steht in ``tools.py``.

Diese Datei ist absichtlich dünn. Sie kennt keine Tool-Namen, keine Grenzen und
keine Handler-Logik: sie fragt die Registry, was es gibt, und reicht Aufrufe
durch ``tools.call`` — also durch Manifest-Prüfung und Deckel. Wer die
Oberfläche ändern will, ändert ``callable_manifest.yaml``, nicht diese Datei.

Der SDK-Import steht **in** den Funktionen: ohne ``pip install -e ".[mcp]"``
bleibt der Rest des Pakets benutzbar, und die Tests der Grenze laufen ohne das
Extra. Eine Abhängigkeit, die nur der Server braucht, gehört nicht in den
Importpfad des Backtest-Workers.

**Woher das Wire-Schema kommt.** Das SDK leitet es aus der *Signatur* der
registrierten Funktion ab. Deshalb wird je Tool ein Wrapper mit
``functools.wraps`` registriert: ``inspect.signature`` folgt ``__wrapped__``
und sieht damit den echten Handler, während der Wrapper Manifest und Deckel
durchsetzt. Die Schemata in ``tools.py`` bleiben die lesbare, SDK-freie
Fassung derselben Zusage — dass beide dieselben Parameter nennen, prüft
``tests/test_mcp_server.py``. Zwei Fassungen ohne Gleichheitsbeweis wären zwei
Wahrheiten.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

from quantrace.mcp_server.tools import (
    CallableTool,
    ResponseTooLargeError,
    ToolNotAllowedError,
    call,
    registry,
)

log = logging.getLogger(__name__)

SERVER_NAME = "neptun"
SERVER_INSTRUCTIONS = (
    "Neptun — die aufrufbare Oberfläche eines Quant-Research-Frameworks. "
    "Frag zuerst platform_capabilities: es sagt, was heute überhaupt "
    "rechenbar ist. Dann assess_feasibility für eine konkrete Idee, bevor du "
    "etwas entwirfst, das die Daten nicht tragen."
)


def _fehlende_abhaengigkeit() -> RuntimeError:
    return RuntimeError(
        "Das MCP-SDK fehlt. Es ist ein Extra, weil nur dieser Server es "
        'braucht: pip install -e ".[mcp]"'
    )


def _wrapper(tool: CallableTool, path: Path | None) -> Any:
    """Der Handler, wie ihn das SDK sieht — mit Gate und Deckel davor.

    ``functools.wraps`` ist hier nicht Kosmetik: es setzt ``__wrapped__``, und
    daran liest das SDK die Signatur ab. Ohne das bekäme jedes Tool ein leeres
    Schema, und ein aufrufendes Modell müsste raten, was es übergeben darf.
    """

    @functools.wraps(tool.handler)
    def _run(**kwargs: Any) -> str:
        # Fehler gehen als *Ergebnis* zurück, nicht als Ausnahme: das aufrufende
        # Modell soll lesen können, warum etwas nicht ging, und danach etwas
        # anderes versuchen. Ein abgebrochener Aufruf lehrt es nichts.
        try:
            return call(tool.name, kwargs, path=path)
        except (ToolNotAllowedError, ResponseTooLargeError) as exc:
            return f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 — siehe oben
            log.exception("Tool %s ist gescheitert", tool.name)
            return f"{type(exc).__name__}: {exc}"

    return _run


def build_server(path: Path | None = None) -> Any:
    """Einen MCP-Server bauen, dessen Toolliste aus dem Manifest kommt."""
    try:
        from mcp.server import MCPServer
    except ModuleNotFoundError as exc:  # pragma: no cover — Abhängigkeit fehlt
        raise _fehlende_abhaengigkeit() from exc

    server: Any = MCPServer(name=SERVER_NAME, instructions=SERVER_INSTRUCTIONS)
    for tool in registry(path).values():
        server.add_tool(_wrapper(tool, path), name=tool.name, description=tool.description)
    return server


def run(path: Path | None = None) -> None:
    """Über stdio bedienen, bis der Aufrufer die Verbindung schließt."""
    import asyncio

    asyncio.run(build_server(path).run_stdio_async())
