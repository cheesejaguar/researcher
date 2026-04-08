"""MCP server exposure of the researcher KnowledgeStore (v1.3 #5).

This package inverts the v1.3 integration story: previous Tier 3 items
added tools INTO researcher (PDF extraction, Exa MCP client, OTEL);
this one exposes researcher ITSELF as a tool so external agents
(Claude Desktop, Cursor, Gemini) can query the accumulated research,
fetch entity details, and trigger new runs.

The :class:`ResearcherMcpServer` in :mod:`researcher.mcp.server` is the
in-process, fully-testable core. The :mod:`researcher.mcp.stdio`
transport is a ten-line wrapper that reads newline-delimited JSON-RPC
from stdin and writes responses to stdout — exercised manually rather
than via unit tests.
"""

from researcher.mcp.server import McpTool, ResearcherMcpServer

__all__ = ["McpTool", "ResearcherMcpServer"]
