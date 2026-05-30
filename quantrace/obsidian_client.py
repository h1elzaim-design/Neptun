"""Obsidian Local REST API Client.

Setzt das Plugin `coddingtonbear/obsidian-local-rest-api` voraus.
Selbstsignierte Zertifikate sind Standard — `verify=False` ist hier
absichtlich (loopback only).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from quantrace.models import KnowledgeNote

log = logging.getLogger(__name__)


class ObsidianClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("OBSIDIAN_API_URL", "https://127.0.0.1:27124")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OBSIDIAN_API_KEY", "")
        if not self.api_key:
            log.warning("OBSIDIAN_API_KEY ist leer — REST-Calls werden 401 zurückgeben.")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            verify=False,
            timeout=timeout,
        )

    # -- low-level ------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        r = self._client.request(method, path, **kwargs)
        if r.status_code >= 400:
            log.error("Obsidian %s %s -> %s: %s", method, path, r.status_code, r.text[:200])
        r.raise_for_status()
        return r

    # -- notes ----------------------------------------------------------------

    def get_note(self, vault_path: str) -> str:
        r = self._request("GET", f"/vault/{vault_path}", headers={"Accept": "text/markdown"})
        return r.text

    def upsert_note(self, vault_path: str, content: str) -> None:
        self._request(
            "PUT",
            f"/vault/{vault_path}",
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/markdown"},
        )

    def append_to_note(self, vault_path: str, content: str) -> None:
        self._request(
            "POST",
            f"/vault/{vault_path}",
            content=content.encode("utf-8"),
            headers={"Content-Type": "text/markdown"},
        )

    def patch_section(
        self,
        vault_path: str,
        heading: str,
        content: str,
        operation: str = "append",
    ) -> None:
        self._request(
            "PATCH",
            f"/vault/{vault_path}",
            content=content.encode("utf-8"),
            headers={
                "Content-Type": "text/markdown",
                "Operation": operation,
                "Target-Type": "heading",
                "Target": heading,
            },
        )

    def delete_note(self, vault_path: str) -> None:
        self._request("DELETE", f"/vault/{vault_path}")

    # -- commands -------------------------------------------------------------

    def list_commands(self) -> list[dict[str, Any]]:
        r = self._request("GET", "/commands/")
        return r.json().get("commands", [])

    def execute_command(self, command_id: str) -> None:
        self._request("POST", f"/commands/{command_id}/")

    # -- high-level -----------------------------------------------------------

    def publish(self, note: KnowledgeNote) -> str:
        """Schreibt einen KnowledgeNote in den Vault. Gibt den Vault-Pfad zurück."""
        self.upsert_note(note.path, note.to_markdown())
        log.info("Obsidian: %s", note.path)
        return note.path

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ObsidianClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
