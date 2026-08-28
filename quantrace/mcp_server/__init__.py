"""Neptun MCP — die aufrufbare Oberfläche des Frameworks.

Die Maschine ist öffentlich, die Daten dahinter nicht. Dieser Server macht die
öffentliche Hälfte für fremde Agenten ansteuerbar, ohne dass dabei etwas aus
dem privaten Labor herausläuft: er ruft ausschließlich Code aus ``quantrace/``
auf, importiert nichts aus ``agents/``, und was er anbietet, steht in
``callable_manifest.yaml`` — nicht in diesem Paket.

Starten (braucht das Extra ``mcp``)::

    pip install -e ".[mcp]"
    python -m quantrace.mcp_server

Die Grenze und ihre Begründung: ``docs/MCP_BOUNDARY.md``.
"""

from quantrace.mcp_server.tools import (
    CallableTool,
    ManifestMissingError,
    ResponseTooLargeError,
    ToolNotAllowedError,
    call,
    load_manifest,
    registry,
)

__all__ = [
    "CallableTool",
    "ManifestMissingError",
    "ResponseTooLargeError",
    "ToolNotAllowedError",
    "call",
    "load_manifest",
    "registry",
]
