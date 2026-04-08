"""In-process MCP server that exposes the researcher KnowledgeStore.

The server implements a minimal subset of the Model Context Protocol tool
interface plus JSON-RPC 2.0 envelope handling. No external MCP SDK
dependency — the envelope is hand-rolled so operators can wire the
server into Claude Desktop / Cursor / Gemini via a trivial stdio
transport (:mod:`researcher.mcp.stdio`).

Exposed tools:

* ``query_entities(entity_type?, limit=50)`` — list entity ids in the
  store, optionally filtered by type.
* ``get_entity(entity_id)`` — fetch a single entity by id.
* ``list_runs(limit=20)`` — list prior runs recorded in the
  ``run_summary`` table.
* ``get_run_status(run_id)`` — return the summary row for a given run.
* ``start_run(spec_path, run_id?)`` — trigger a new researcher run via
  an injected async launcher callable. Disabled by default.

Every tool handler is a thin async wrapper around an existing
:class:`~researcher.storage.store.KnowledgeStore` method; the server
performs no business logic of its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable

from researcher.storage.store import Entity, KnowledgeStore

# JSON-RPC 2.0 error codes we care about.
_METHOD_NOT_FOUND = -32601
_INTERNAL_ERROR = -32000


@dataclass
class McpTool:
    """A single MCP tool registration.

    The handler is an async callable that receives the raw ``arguments``
    dict from ``tools/call`` and returns a JSON-serializable result dict.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _serialize_entity(entity: Entity) -> dict[str, Any]:
    """Render an Entity as a JSON-safe dict for the MCP tool response."""
    fields: dict[str, Any] = {}
    for name, cell in entity.fields.items():
        fields[name] = {
            "value": cell.value,
            "confidence": float(cell.confidence),
            "provenance_ids": list(cell.provenance_ids),
        }
    return {
        "entity_id": entity.id,
        "entity_type": entity.type,
        "fields": fields,
    }


class ResearcherMcpServer:
    """In-process MCP server exposing the researcher KnowledgeStore.

    Construct with an opened :class:`KnowledgeStore` and (optionally) a
    ``run_launcher`` callable. When no launcher is provided, the
    ``start_run`` tool returns a disabled-error response so the server
    is safe to expose read-only.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        run_launcher: Callable[[str, str | None], Awaitable[str]] | None = None,
    ) -> None:
        self._store = store
        self._run_launcher = run_launcher
        self._tools: list[McpTool] = self._build_tools()
        self._tool_index: dict[str, McpTool] = {t.name: t for t in self._tools}

    # -------------------------------------------------- tool registration

    def _build_tools(self) -> list[McpTool]:
        return [
            McpTool(
                name="query_entities",
                description=(
                    "List entity ids in the researcher KnowledgeStore, "
                    "optionally filtered by entity_type and capped by limit."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "entity_type": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                },
                handler=self._handle_query_entities,
            ),
            McpTool(
                name="get_entity",
                description=(
                    "Fetch one entity by id. Returns {error: 'not_found'} "
                    "when the entity does not exist."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"entity_id": {"type": "string"}},
                    "required": ["entity_id"],
                },
                handler=self._handle_get_entity,
            ),
            McpTool(
                name="list_runs",
                description=("List recent runs recorded in the run_summary table, newest first."),
                input_schema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1},
                    },
                },
                handler=self._handle_list_runs,
            ),
            McpTool(
                name="get_run_status",
                description=(
                    "Return the stored summary for a specific run_id, or "
                    "{error: 'not_found'} if no summary is recorded."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
                handler=self._handle_get_run_status,
            ),
            McpTool(
                name="start_run",
                description=(
                    "Trigger a new researcher run from a spec file. "
                    "Gated behind an injected launcher; disabled by default."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "spec_path": {"type": "string"},
                        "run_id": {"type": "string"},
                    },
                    "required": ["spec_path"],
                },
                handler=self._handle_start_run,
            ),
        ]

    def tools(self) -> list[McpTool]:
        """Return the internal list of :class:`McpTool` dataclasses."""
        return self._tools

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the MCP-style wire representation of the tool registry."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in self._tools
        ]

    # -------------------------------------------------- tool dispatch

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch to a tool handler by name."""
        tool = self._tool_index.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        return await tool.handler(arguments or {})

    # -------------------------------------------------- tool handlers

    async def _handle_query_entities(self, arguments: dict[str, Any]) -> dict[str, Any]:
        entity_type = arguments.get("entity_type")
        limit_raw = arguments.get("limit", 50)
        try:
            limit = int(limit_raw)
        except (TypeError, ValueError):
            limit = 50
        if limit < 1:
            limit = 1

        entity_ids = await self._list_entity_ids(entity_type, limit)
        entities: list[dict[str, Any]] = []
        for eid in entity_ids:
            ent = await self._store.get_entity(eid)
            if ent is not None:
                entities.append(_serialize_entity(ent))
        return {"entities": entities}

    async def _list_entity_ids(self, entity_type: str | None, limit: int) -> list[str]:
        """List entity ids, preferring SQL and falling back to in-memory scan.

        The DuckDB-backed store answers via :meth:`KnowledgeStore.query`;
        the in-memory test stub raises ``NotImplementedError`` from
        ``query``, so we fall back to scanning the stub's ``_entities``
        mapping. A catch-all ``Exception`` guard also handles the case
        where the ``entities`` table is missing (e.g. a freshly opened
        store without ``init_schema`` having been called).
        """
        try:
            if entity_type is not None:
                rows = await self._store.query(
                    "SELECT id FROM entities WHERE entity_type = ? LIMIT ?",
                    (entity_type, limit),
                )
            else:
                rows = await self._store.query(
                    "SELECT id FROM entities LIMIT ?",
                    (limit,),
                )
        except NotImplementedError:
            return self._in_memory_entity_ids(entity_type, limit)
        except Exception:
            return []

        ids: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            eid = row.get("id") or row.get("entity_id")
            if eid is not None:
                ids.append(str(eid))
        return ids

    def _in_memory_entity_ids(self, entity_type: str | None, limit: int) -> list[str]:
        """In-memory fallback for stores whose ``query`` is unimplemented.

        Used exclusively by :class:`tests.stubs.store.StubKnowledgeStore`
        in the unit-test suite. Real users always hit the DuckDB path.
        """
        internal = getattr(self._store, "_entities", None)
        if not isinstance(internal, dict):
            return []
        ids: list[str] = []
        for eid, ent in internal.items():
            if entity_type is not None and getattr(ent, "type", None) != entity_type:
                continue
            ids.append(eid)
            if len(ids) >= limit:
                break
        return ids

    async def _handle_get_entity(self, arguments: dict[str, Any]) -> dict[str, Any]:
        entity_id = arguments.get("entity_id")
        if not entity_id:
            return {"error": "not_found"}
        ent = await self._store.get_entity(str(entity_id))
        if ent is None:
            return {"error": "not_found"}
        return {"entity": _serialize_entity(ent)}

    async def _handle_list_runs(self, arguments: dict[str, Any]) -> dict[str, Any]:
        limit_raw = arguments.get("limit", 20)
        try:
            limit = int(limit_raw)
        except (TypeError, ValueError):
            limit = 20
        if limit < 1:
            limit = 1

        try:
            rows = await self._store.query(
                "SELECT run_id, started_at, summary_json "
                "FROM run_summary ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
        except NotImplementedError:
            return {"runs": []}
        except Exception:
            # Fresh store without a run_summary table yet.
            return {"runs": []}

        runs: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            summary_json = row.get("summary_json")
            summary: Any
            try:
                summary = (
                    json.loads(summary_json) if isinstance(summary_json, str) else summary_json
                )
            except (TypeError, ValueError):
                summary = None
            started_at = row.get("started_at")
            if isinstance(started_at, datetime):
                started_at = started_at.isoformat()
            runs.append(
                {
                    "run_id": row.get("run_id"),
                    "started_at": started_at,
                    "summary": summary,
                }
            )
        return {"runs": runs}

    async def _handle_get_run_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        run_id = arguments.get("run_id")
        if not run_id:
            return {"error": "not_found"}
        try:
            rows = await self._store.query(
                "SELECT run_id, started_at, summary_json "
                "FROM run_summary WHERE run_id = ? "
                "ORDER BY started_at DESC LIMIT 1",
                (run_id,),
            )
        except NotImplementedError:
            return {"error": "not_found"}
        except Exception:
            return {"error": "not_found"}

        if not rows:
            return {"error": "not_found"}
        row = rows[0]
        if not isinstance(row, dict):
            return {"error": "not_found"}
        summary_json = row.get("summary_json")
        summary: dict[str, Any] = {}
        try:
            parsed = json.loads(summary_json) if isinstance(summary_json, str) else summary_json
            if isinstance(parsed, dict):
                summary = parsed
        except (TypeError, ValueError):
            summary = {}
        started_at = row.get("started_at")
        if isinstance(started_at, datetime):
            started_at = started_at.isoformat()
        merged: dict[str, Any] = {"run_id": str(row.get("run_id")), **summary}
        if started_at is not None:
            merged.setdefault("started_at", started_at)
        return merged

    async def _handle_start_run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._run_launcher is None:
            return {"error": "start_run not enabled on this server"}
        spec_path = arguments.get("spec_path")
        if not spec_path:
            return {"started": False, "error": "spec_path is required"}
        run_id = arguments.get("run_id")
        run_id_arg = str(run_id) if run_id is not None else None
        try:
            launched_id = await self._run_launcher(str(spec_path), run_id_arg)
        except Exception as exc:
            return {"started": False, "error": str(exc)}
        return {"run_id": launched_id, "started": True}

    # -------------------------------------------------- JSON-RPC envelope

    async def handle_jsonrpc(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle a single JSON-RPC 2.0 request.

        Returns ``None`` for notifications (messages without an ``id``
        field) so the stdio transport knows not to emit a response.
        Requests always get an envelope back — either a ``result`` on
        success or an ``error`` on failure. Tool exceptions are wrapped
        as internal errors with code -32000 so a misbehaving tool can't
        take down the transport loop.
        """
        msg_id = message.get("id") if isinstance(message, dict) else None
        is_notification = isinstance(message, dict) and "id" not in message

        method = message.get("method") if isinstance(message, dict) else None
        params = message.get("params") or {}

        if is_notification:
            return None

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": self.list_tools()},
            }

        if method == "tools/call":
            tool_name = params.get("name") if isinstance(params, dict) else None
            arguments = params.get("arguments", {}) if isinstance(params, dict) else {}
            if not isinstance(arguments, dict):
                arguments = {}
            if not tool_name:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": _METHOD_NOT_FOUND,
                        "message": "tools/call requires a name",
                    },
                }
            try:
                result = await self.call_tool(str(tool_name), arguments)
            except KeyError as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": _METHOD_NOT_FOUND,
                        "message": str(exc),
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": _INTERNAL_ERROR,
                        "message": str(exc),
                    },
                }
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": _METHOD_NOT_FOUND, "message": "method not found"},
        }
