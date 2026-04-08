"""Tests for researcher.observability.otel — OTLP trace export (v1.3 #4).

The OtelEventAdapter subscribes to EventBus events and translates them into a
hierarchical OTLP span tree:

    run                       (opened on start_run)
    +--- cycle                (CycleStart -> CycleEnd)
          +--- agent.<kind>   (AgentSpawn -> AgentStateChange done/failed)

Key invariants exercised here:
  1. run -> cycle -> agent parent/child nesting via stable span IDs.
  2. AgentLog events buffer into the agent span's attributes and are flushed
     into end_span(attributes=...) when the agent closes.
  3. include_facts toggles whether FactWritten emits a brief point span.
  4. flush() closes any still-open span with status="error" + reason="flush".
  5. trace_id is stably derived from run_id — same run_id, same trace_id.
  6. EventBus.add_subscriber receives every emitted event; subscriber errors
     are swallowed + counted via bus._subscriber_errors, never re-raised.
  7. build_otlp_exporter raises a clear RuntimeError mentioning
     "opentelemetry-sdk" when the SDK is not importable.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from researcher.events import (
    AgentLog,
    AgentLogPayload,
    AgentSpawn,
    AgentSpawnPayload,
    AgentStateChange,
    AgentStateChangePayload,
    CycleEnd,
    CycleEndPayload,
    CycleStart,
    CycleStartPayload,
    EventBus,
    FactWritten,
    FactWrittenPayload,
)
from researcher.observability.otel import (
    OtelEventAdapter,
    build_otlp_exporter,
)

RUN_ID = "run-otel-test"


def _ts() -> datetime:
    return datetime(2026, 4, 8, 12, 0, tzinfo=UTC)


# ---------- Stub exporter ----------


class StubExporter:
    """In-memory SpanExporter that records all start/end calls."""

    def __init__(self) -> None:
        self.started: list[dict[str, Any]] = []
        self.ended: list[dict[str, Any]] = []

    def start_span(
        self,
        name: str,
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        start_time_unix_nano: int,
        attributes: dict[str, Any],
    ) -> None:
        self.started.append(
            {
                "name": name,
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "start_ns": start_time_unix_nano,
                "attributes": dict(attributes),
            }
        )

    def end_span(
        self,
        *,
        span_id: str,
        end_time_unix_nano: int,
        status: str = "ok",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.ended.append(
            {
                "span_id": span_id,
                "end_ns": end_time_unix_nano,
                "status": status,
                "attributes": dict(attributes or {}),
            }
        )


# ---------- Event factories ----------


def _cycle_start(cycle: int = 1) -> CycleStart:
    return CycleStart(
        seq=0,
        ts=_ts(),
        run_id=RUN_ID,
        payload=CycleStartPayload(cycle=cycle, pending_tasks=3),
    )


def _cycle_end(cycle: int = 1) -> CycleEnd:
    return CycleEnd(
        seq=0,
        ts=_ts(),
        run_id=RUN_ID,
        payload=CycleEndPayload(
            cycle=cycle,
            claims_written=5,
            entities_total=2,
            cost_usd=0.12,
        ),
    )


def _agent_spawn(agent_id: str = "a1", kind: str = "discover") -> AgentSpawn:
    return AgentSpawn(
        seq=0,
        ts=_ts(),
        run_id=RUN_ID,
        payload=AgentSpawnPayload(agent_id=agent_id, task_id="t1", kind=kind),
    )


def _agent_state_change(agent_id: str, new: str) -> AgentStateChange:
    return AgentStateChange(
        seq=0,
        ts=_ts(),
        run_id=RUN_ID,
        payload=AgentStateChangePayload(agent_id=agent_id, old="running", new=new),
    )


def _agent_log(agent_id: str, msg: str = "hello", level: str = "warn") -> AgentLog:
    return AgentLog(
        seq=0,
        ts=_ts(),
        run_id=RUN_ID,
        payload=AgentLogPayload(agent_id=agent_id, level=level, msg=msg),  # type: ignore[arg-type]
    )


def _fact_written() -> FactWritten:
    return FactWritten(
        seq=0,
        ts=_ts(),
        run_id=RUN_ID,
        payload=FactWrittenPayload(
            entity_id="ent-1",
            entity_type="War",
            field="start_year",
            value=1939,
            confidence=0.9,
            source_url="https://example.com",
        ),
    )


# ---------- OtelEventAdapter ----------


def test_start_run_creates_root_span():
    exp = StubExporter()
    adapter = OtelEventAdapter(exp, run_id=RUN_ID)
    adapter.start_run()

    assert len(exp.started) == 1
    root = exp.started[0]
    assert root["name"] == "run"
    assert root["parent_span_id"] is None
    assert root["trace_id"]  # non-empty


def test_cycle_span_is_child_of_run():
    exp = StubExporter()
    adapter = OtelEventAdapter(exp, run_id=RUN_ID)
    adapter.start_run()
    adapter.handle(_cycle_start(1))

    assert len(exp.started) == 2
    run_span = exp.started[0]
    cycle_span = exp.started[1]
    assert cycle_span["name"] == "cycle"
    assert cycle_span["parent_span_id"] == run_span["span_id"]
    assert cycle_span["trace_id"] == run_span["trace_id"]


def test_cycle_end_closes_cycle_span():
    exp = StubExporter()
    adapter = OtelEventAdapter(exp, run_id=RUN_ID)
    adapter.start_run()
    adapter.handle(_cycle_start(1))
    cycle_span_id = exp.started[-1]["span_id"]
    adapter.handle(_cycle_end(1))

    matches = [e for e in exp.ended if e["span_id"] == cycle_span_id]
    assert len(matches) == 1
    assert matches[0]["status"] == "ok"


def test_agent_spawn_creates_agent_span():
    exp = StubExporter()
    adapter = OtelEventAdapter(exp, run_id=RUN_ID)
    adapter.start_run()
    adapter.handle(_cycle_start(1))
    cycle_span_id = exp.started[-1]["span_id"]
    adapter.handle(_agent_spawn("a1", "discover"))

    agent_span = exp.started[-1]
    assert agent_span["name"] == "agent.discover"
    assert agent_span["parent_span_id"] == cycle_span_id


def test_agent_done_closes_agent_span():
    exp = StubExporter()
    adapter = OtelEventAdapter(exp, run_id=RUN_ID)
    adapter.start_run()
    adapter.handle(_cycle_start(1))
    adapter.handle(_agent_spawn("a1", "enrich"))
    agent_span_id = exp.started[-1]["span_id"]
    adapter.handle(_agent_state_change("a1", "done"))

    matches = [e for e in exp.ended if e["span_id"] == agent_span_id]
    assert len(matches) == 1
    assert matches[0]["status"] == "ok"


def test_agent_failed_closes_agent_span_with_error_status():
    exp = StubExporter()
    adapter = OtelEventAdapter(exp, run_id=RUN_ID)
    adapter.start_run()
    adapter.handle(_cycle_start(1))
    adapter.handle(_agent_spawn("a1", "enrich"))
    agent_span_id = exp.started[-1]["span_id"]
    adapter.handle(_agent_state_change("a1", "failed"))

    matches = [e for e in exp.ended if e["span_id"] == agent_span_id]
    assert len(matches) == 1
    assert matches[0]["status"] == "error"


def test_agent_log_appends_attribute():
    exp = StubExporter()
    adapter = OtelEventAdapter(exp, run_id=RUN_ID)
    adapter.start_run()
    adapter.handle(_cycle_start(1))
    adapter.handle(_agent_spawn("a1", "discover"))
    adapter.handle(_agent_log("a1", msg="warning 1", level="warn"))
    adapter.handle(_agent_log("a1", msg="warning 2", level="warn"))
    adapter.handle(_agent_state_change("a1", "done"))

    # Find the agent's end_span call
    agent_end = [e for e in exp.ended if e["attributes"].get("logs")]
    assert len(agent_end) == 1
    logs = agent_end[0]["attributes"]["logs"]
    assert isinstance(logs, list)
    assert len(logs) >= 2


def test_fact_written_creates_point_span_when_include_facts():
    exp = StubExporter()
    adapter = OtelEventAdapter(exp, run_id=RUN_ID, include_facts=True)
    adapter.start_run()
    adapter.handle(_cycle_start(1))
    start_count_before = len(exp.started)
    end_count_before = len(exp.ended)

    adapter.handle(_fact_written())

    # A new "fact" span both started and ended immediately.
    new_started = exp.started[start_count_before:]
    new_ended = exp.ended[end_count_before:]
    fact_starts = [s for s in new_started if s["name"] == "fact"]
    assert len(fact_starts) == 1
    fact_span_id = fact_starts[0]["span_id"]
    fact_ends = [e for e in new_ended if e["span_id"] == fact_span_id]
    assert len(fact_ends) == 1


def test_fact_written_ignored_when_include_facts_false():
    exp = StubExporter()
    adapter = OtelEventAdapter(exp, run_id=RUN_ID)  # default include_facts=False
    adapter.start_run()
    adapter.handle(_cycle_start(1))
    start_count_before = len(exp.started)

    adapter.handle(_fact_written())

    new_started = exp.started[start_count_before:]
    fact_starts = [s for s in new_started if s["name"] == "fact"]
    assert fact_starts == []


def test_flush_ends_open_spans_with_error_status():
    exp = StubExporter()
    adapter = OtelEventAdapter(exp, run_id=RUN_ID)
    adapter.start_run()
    adapter.handle(_cycle_start(1))
    cycle_span_id = exp.started[-1]["span_id"]

    adapter.flush()

    matches = [e for e in exp.ended if e["span_id"] == cycle_span_id]
    assert len(matches) == 1
    assert matches[0]["status"] == "error"
    assert matches[0]["attributes"].get("reason") == "flush"


def test_trace_id_is_stable_for_run_id():
    exp1 = StubExporter()
    exp2 = StubExporter()
    a1 = OtelEventAdapter(exp1, run_id="stable-run-id-42")
    a2 = OtelEventAdapter(exp2, run_id="stable-run-id-42")
    a1.start_run()
    a2.start_run()

    assert exp1.started[0]["trace_id"] == exp2.started[0]["trace_id"]


def test_handle_ignores_unknown_event_types():
    exp = StubExporter()
    adapter = OtelEventAdapter(exp, run_id=RUN_ID)
    adapter.start_run()

    class NotAnEvent:
        pass

    # Should not raise.
    adapter.handle(NotAnEvent())  # type: ignore[arg-type]


# ---------- EventBus subscribers ----------


@pytest.mark.asyncio
async def test_event_bus_add_subscriber_receives_events(tmp_path: Path):
    bus = EventBus(jsonl_path=tmp_path / "events.jsonl")
    await bus.start()
    received: list[Any] = []
    bus.add_subscriber(lambda e: received.append(e))
    try:
        await bus.emit(_cycle_start(1))
        await bus.emit(_cycle_end(1))
    finally:
        await bus.stop()

    assert len(received) == 2
    assert received[0].type == "cycle_start"
    assert received[1].type == "cycle_end"


@pytest.mark.asyncio
async def test_event_bus_subscriber_exception_counted_not_raised(tmp_path: Path):
    bus = EventBus(jsonl_path=tmp_path / "events.jsonl")
    await bus.start()

    def bad(_ev: Any) -> None:
        raise RuntimeError("boom")

    bus.add_subscriber(bad)
    try:
        await bus.emit(_cycle_start(1))
    finally:
        await bus.stop()

    assert bus._subscriber_errors == 1


@pytest.mark.asyncio
async def test_event_bus_multiple_subscribers_all_called(tmp_path: Path):
    bus = EventBus(jsonl_path=tmp_path / "events.jsonl")
    await bus.start()
    calls_a: list[Any] = []
    calls_b: list[Any] = []
    calls_c: list[Any] = []
    bus.add_subscriber(lambda e: calls_a.append(e))
    bus.add_subscriber(lambda e: calls_b.append(e))
    bus.add_subscriber(lambda e: calls_c.append(e))
    try:
        await bus.emit(_cycle_start(1))
    finally:
        await bus.stop()

    assert len(calls_a) == 1
    assert len(calls_b) == 1
    assert len(calls_c) == 1


# ---------- Real OTLP exporter soft-import ----------


def test_build_otlp_exporter_raises_when_sdk_missing(monkeypatch: pytest.MonkeyPatch):
    # Poison the import so importlib raises ImportError.
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
        None,
    )

    with pytest.raises(RuntimeError, match="opentelemetry-sdk"):
        build_otlp_exporter("localhost:4317")
