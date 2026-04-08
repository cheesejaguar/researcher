"""Newline-delimited JSON stdio transport for :class:`ResearcherMcpServer`.

This is intentionally a tiny hand-rolled wrapper — no external MCP SDK
dependency. Claude Desktop / Cursor / Gemini all spawn an MCP server as
a subprocess and exchange JSON-RPC messages line-by-line on stdin/stdout,
which is exactly what :func:`serve_stdio` implements.

There is no unit test for this module; it is exercised manually via
``researcher mcp --db <path>``. Keeping it at ~20 lines of pure I/O glue
is deliberate so the blast radius of "untested" stays minimal.
"""

from __future__ import annotations

import asyncio
import json
import sys

from researcher.mcp.server import ResearcherMcpServer


async def serve_stdio(server: ResearcherMcpServer) -> None:
    """Read JSON-RPC messages from stdin and write responses to stdout.

    One message per line. Malformed JSON is silently dropped so a single
    bad line can't kill the transport loop. Notifications (messages
    without an ``id`` field) don't produce a response.
    """
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            message = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        response = await server.handle_jsonrpc(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
