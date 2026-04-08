"""Exa MCP search provider with strict tool-description sanitization.

The Exa MCP server exposes semantic/neural web search as a set of named
tools (``exa_search``, ``exa_find_similar``, ...). We deliberately treat
the server as untrusted: tool descriptions advertised via ``list_tools``
are read once, used only to compute the intersection with our version
pin's allowlist, and then dropped. They never pass into any log,
exception message, stored attribute, or returned ``SearchResult`` — which
is the mitigation for the MCP tool-poisoning class of attacks documented
by Invariant Labs in April 2025.

The provider accepts an injected :class:`ExaMcpClient` (Protocol) rather
than instantiating an MCP SDK client itself. Tests pass in-memory stubs;
wiring a real stdio transport into the CLI is a follow-up and is not
required for this module to be correct.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from researcher.config import McpServerPin
from researcher.search.base import SearchResult


class ExaMcpError(RuntimeError):
    """Raised when the MCP server's tool surface fails pin verification.

    The message is intentionally generic and MUST NOT include any portion
    of the server-advertised tool descriptions.
    """


@runtime_checkable
class ExaMcpClient(Protocol):
    """Narrow transport contract the provider depends on.

    Real implementations wrap an MCP stdio session; tests pass an
    in-memory stub. Keeping this a Protocol means this module has zero
    runtime dependency on an MCP SDK.
    """

    async def list_tools(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class ExaMcpProvider:
    """SearchProvider backed by the Exa MCP server.

    First ``search()`` call triggers a one-shot verification against the
    version-pinned allowlist; the result is cached on the instance so
    steady-state traffic pays exactly one ``list_tools`` round trip.
    Verification failure is sticky and fail-closed — every subsequent
    ``search()`` returns ``[]`` without contacting the server.
    """

    name = "exa_mcp"

    def __init__(
        self,
        client: ExaMcpClient,
        pin: McpServerPin,
        max_results: int = 10,
    ) -> None:
        self._client = client
        self._pin = pin
        self._max_results = max_results
        # None = unchecked, True = verified, False = sticky fail-closed.
        self._verified: bool | None = None

    async def _verify_tools(self) -> None:
        """Verify that ``exa_search`` is in the server's allowlisted tools.

        Reads the tool list once. Tool *descriptions* are deliberately
        not bound to any local name — we iterate, pull only the ``name``
        field, and drop the rest. Raises :class:`ExaMcpError` with a
        description-free message if the required tool is absent.
        """
        tools = await self._client.list_tools()
        available: set[str] = set()
        for t in tools:
            if not isinstance(t, dict):
                continue
            tname = t.get("name", "")
            # Sanitization invariant: do NOT read t["description"] into
            # any local, log line, or exception. The membership test is
            # all we need.
            if isinstance(tname, str) and tname in self._pin.allowed_tools:
                available.add(tname)
        if "exa_search" not in available:
            raise ExaMcpError("exa_search tool unavailable or not allowlisted")

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        if self._verified is None:
            try:
                await self._verify_tools()
                self._verified = True
            except Exception:
                # Fail-closed and sticky. The exception is swallowed
                # rather than re-raised so the outer orchestrator treats
                # this provider like any other search failure.
                self._verified = False
        if not self._verified:
            return []

        try:
            resp = await self._client.call_tool(
                "exa_search",
                {
                    "query": query,
                    "num_results": min(max_results, self._max_results),
                },
            )
        except Exception:
            return []

        raw = resp.get("results") if isinstance(resp, dict) else None
        if not isinstance(raw, list):
            return []

        out: list[SearchResult] = []
        for i, r in enumerate(raw):
            if not isinstance(r, dict):
                continue
            title = str(r.get("title", ""))
            url = str(r.get("url", ""))
            text = str(r.get("text", ""))
            out.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=text[:500],
                    rank=i,
                )
            )
        return out


__all__ = ["ExaMcpClient", "ExaMcpError", "ExaMcpProvider"]
