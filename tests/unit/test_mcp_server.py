"""Unit tests for the ResearcherMcpServer (v1.3 #5).

The MCP server exposes the researcher KnowledgeStore to external agents
over a minimal MCP tool interface. The server is pure-Python and
in-process testable: the JSON-RPC envelope is hand-rolled and no stdio
transport is touched here (that's a 10-line wrapper exercised manually).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from researcher.mcp.server import McpTool, ResearcherMcpServer
from researcher.storage.store import FieldCell
from tests.stubs.store import StubKnowledgeStore

# ---------------------------------------------------------------- helpers


def _cell(value: Any, confidence: float = 0.9) -> FieldCell:
    return FieldCell(
        value=value,
        confidence=confidence,
        provenance_ids=[],
        updated_at=datetime.now(UTC),
    )


async def _populate(
    store: StubKnowledgeStore,
    entity_type: str,
    names: list[str],
) -> list[str]:
    ids: list[str] = []
    for name in names:
        eid = await store.upsert_entity(
            entity_type,
            name,
            {"detail": _cell(f"{name}-detail")},
        )
        ids.append(eid)
    return ids


# ---------------------------------------------------------------- tool discovery


@pytest.mark.asyncio
async def test_list_tools_exposes_five_tools() -> None:
    store = StubKnowledgeStore()
    server = ResearcherMcpServer(store=store)

    tools = server.list_tools()

    names = {t["name"] for t in tools}
    assert names == {
        "query_entities",
        "get_entity",
        "list_runs",
        "get_run_status",
        "start_run",
    }
    for tool in tools:
        assert tool["description"]
        assert isinstance(tool["inputSchema"], dict)

    # The internal tool objects should be McpTool dataclasses.
    internal = server.tools()
    assert len(internal) == 5
    for t in internal:
        assert isinstance(t, McpTool)
        assert callable(t.handler)


@pytest.mark.asyncio
async def test_tools_list_jsonrpc() -> None:
    store = StubKnowledgeStore()
    server = ResearcherMcpServer(store=store)

    response = await server.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    )

    assert response is not None
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    tool_names = {t["name"] for t in response["result"]["tools"]}
    assert "query_entities" in tool_names


# ---------------------------------------------------------------- query_entities


@pytest.mark.asyncio
async def test_query_entities_empty_store() -> None:
    store = StubKnowledgeStore()
    server = ResearcherMcpServer(store=store)

    result = await server.call_tool("query_entities", {})

    assert result == {"entities": []}


@pytest.mark.asyncio
async def test_query_entities_returns_populated() -> None:
    store = StubKnowledgeStore()
    await _populate(store, "War", ["WWI", "WWII", "Korea"])
    await _populate(store, "Person", ["Ada", "Grace"])
    server = ResearcherMcpServer(store=store)

    wars = await server.call_tool("query_entities", {"entity_type": "War"})
    assert len(wars["entities"]) == 3
    for ent in wars["entities"]:
        assert ent["entity_type"] == "War"

    everything = await server.call_tool("query_entities", {})
    assert len(everything["entities"]) == 5


@pytest.mark.asyncio
async def test_query_entities_respects_limit() -> None:
    store = StubKnowledgeStore()
    await _populate(store, "War", [f"war-{i}" for i in range(10)])
    server = ResearcherMcpServer(store=store)

    result = await server.call_tool("query_entities", {"entity_type": "War", "limit": 3})

    assert len(result["entities"]) == 3


# ---------------------------------------------------------------- get_entity


@pytest.mark.asyncio
async def test_get_entity_existing() -> None:
    store = StubKnowledgeStore()
    ids = await _populate(store, "War", ["WWI"])
    server = ResearcherMcpServer(store=store)

    result = await server.call_tool("get_entity", {"entity_id": ids[0]})

    assert "entity" in result
    assert result["entity"]["entity_id"] == ids[0]
    assert result["entity"]["entity_type"] == "War"
    assert "fields" in result["entity"]


@pytest.mark.asyncio
async def test_get_entity_missing() -> None:
    store = StubKnowledgeStore()
    server = ResearcherMcpServer(store=store)

    result = await server.call_tool("get_entity", {"entity_id": "does-not-exist"})

    assert result == {"error": "not_found"}


# ---------------------------------------------------------------- list_runs / status


@pytest.mark.asyncio
async def test_list_runs_empty() -> None:
    store = StubKnowledgeStore()
    server = ResearcherMcpServer(store=store)

    result = await server.call_tool("list_runs", {})

    assert result == {"runs": []}


@pytest.mark.asyncio
async def test_get_run_status_missing() -> None:
    store = StubKnowledgeStore()
    server = ResearcherMcpServer(store=store)

    result = await server.call_tool("get_run_status", {"run_id": "nope"})

    assert result == {"error": "not_found"}


# ---------------------------------------------------------------- start_run


@pytest.mark.asyncio
async def test_start_run_disabled_without_launcher() -> None:
    store = StubKnowledgeStore()
    server = ResearcherMcpServer(store=store)

    result = await server.call_tool("start_run", {"spec_path": "specs/wars.yaml"})

    assert result == {"error": "start_run not enabled on this server"}


@pytest.mark.asyncio
async def test_start_run_calls_launcher() -> None:
    store = StubKnowledgeStore()
    calls: list[tuple[str, str | None]] = []

    async def launcher(spec_path: str, run_id: str | None) -> str:
        calls.append((spec_path, run_id))
        return "run-123"

    server = ResearcherMcpServer(store=store, run_launcher=launcher)

    result = await server.call_tool("start_run", {"spec_path": "specs/wars.yaml"})

    assert calls == [("specs/wars.yaml", None)]
    assert result == {"run_id": "run-123", "started": True}


@pytest.mark.asyncio
async def test_start_run_launcher_exception() -> None:
    store = StubKnowledgeStore()

    async def launcher(spec_path: str, run_id: str | None) -> str:
        raise RuntimeError("boom")

    server = ResearcherMcpServer(store=store, run_launcher=launcher)

    result = await server.call_tool("start_run", {"spec_path": "specs/wars.yaml"})

    assert result["started"] is False
    assert "boom" in result["error"]


# ---------------------------------------------------------------- JSON-RPC envelope


@pytest.mark.asyncio
async def test_jsonrpc_tools_call_success() -> None:
    store = StubKnowledgeStore()
    server = ResearcherMcpServer(store=store)

    response = await server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "query_entities", "arguments": {}},
        }
    )

    assert response is not None
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 2
    assert "result" in response
    assert response["result"] == {"entities": []}


@pytest.mark.asyncio
async def test_jsonrpc_unknown_method() -> None:
    store = StubKnowledgeStore()
    server = ResearcherMcpServer(store=store)

    response = await server.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 7, "method": "banana", "params": {}}
    )

    assert response is not None
    assert response["id"] == 7
    assert "error" in response
    assert response["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_jsonrpc_tool_exception_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    store = StubKnowledgeStore()
    server = ResearcherMcpServer(store=store)

    async def exploding_handler(arguments: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("kaboom")

    # Swap the query_entities tool handler with one that raises.
    for tool in server.tools():
        if tool.name == "query_entities":
            tool.handler = exploding_handler
            break

    response = await server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "query_entities", "arguments": {}},
        }
    )

    assert response is not None
    assert response["id"] == 9
    assert "error" in response
    assert response["error"]["code"] == -32000
    assert "kaboom" in response["error"]["message"]


@pytest.mark.asyncio
async def test_jsonrpc_notification_returns_none() -> None:
    store = StubKnowledgeStore()
    server = ResearcherMcpServer(store=store)

    response = await server.handle_jsonrpc({"jsonrpc": "2.0", "method": "tools/list", "params": {}})

    assert response is None
