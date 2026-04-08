"""Tests for researcher.events — Event union + EventBus.

Load-bearing invariants for the event bus:
  1. JSONL on disk is AUTHORITATIVE: every emit appears in the file, fsync'd.
  2. Socket queue is best-effort bounded: when full, emit drops oldest and increments counter,
     but never blocks the core.
  3. seq is monotonically assigned by the bus, starting from 0.
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from researcher.events import (
    AgentLog,
    AgentLogPayload,
    CostUpdate,
    CostUpdatePayload,
    CycleStart,
    CycleStartPayload,
    EventBus,
    FactWritten,
    FactWrittenPayload,
    RunComplete,
    RunCompletePayload,
    parse_event,
)

RUN_ID = "run-test"


def _ts() -> datetime:
    return datetime(2026, 4, 8, 12, 0, tzinfo=UTC)


# ---------- Event union ----------

def test_cycle_start_has_type_discriminator():
    e = CycleStart(
        seq=0,
        ts=_ts(),
        run_id=RUN_ID,
        payload=CycleStartPayload(cycle=1, pending_tasks=3),
    )
    assert e.type == "cycle_start"


def test_event_roundtrip_through_json():
    e = FactWritten(
        seq=5,
        ts=_ts(),
        run_id=RUN_ID,
        payload=FactWrittenPayload(
            entity_id="ent-1",
            entity_type="War",
            field="start_year",
            value=1939,
            confidence=0.95,
            source_url="https://example.com",
        ),
    )
    data = e.model_dump_json()
    parsed = parse_event(json.loads(data))
    assert isinstance(parsed, FactWritten)
    assert parsed.payload.value == 1939
    assert parsed.seq == 5


def test_parse_event_dispatches_by_type():
    # A raw dict comes off the wire; parse_event must route to the right subclass.
    raw = {
        "type": "agent_log",
        "seq": 2,
        "ts": _ts().isoformat(),
        "run_id": RUN_ID,
        "payload": {"agent_id": "a1", "level": "info", "msg": "hello"},
    }
    e = parse_event(raw)
    assert isinstance(e, AgentLog)
    assert e.payload.msg == "hello"


# ---------- EventBus: JSONL fsync ----------

@pytest.mark.asyncio
async def test_emit_writes_jsonl(tmp_path: Path):
    bus = EventBus(jsonl_path=tmp_path / "events.jsonl")
    await bus.start()
    try:
        await bus.emit(
            CycleStart(
                seq=0, ts=_ts(), run_id=RUN_ID,
                payload=CycleStartPayload(cycle=1, pending_tasks=3),
            )
        )
        await bus.emit(
            AgentLog(
                seq=0, ts=_ts(), run_id=RUN_ID,
                payload=AgentLogPayload(agent_id="a1", level="info", msg="hi"),
            )
        )
    finally:
        await bus.stop()

    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["type"] == "cycle_start"
    assert second["type"] == "agent_log"


@pytest.mark.asyncio
async def test_emit_assigns_monotonic_seq(tmp_path: Path):
    # Callers pass seq=0; the bus overwrites with its own monotonic counter.
    bus = EventBus(jsonl_path=tmp_path / "events.jsonl")
    await bus.start()
    try:
        for i in range(5):
            await bus.emit(
                AgentLog(
                    seq=0, ts=_ts(), run_id=RUN_ID,
                    payload=AgentLogPayload(agent_id="a", level="info", msg=f"{i}"),
                )
            )
    finally:
        await bus.stop()

    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    seqs = [json.loads(line)["seq"] for line in lines]
    assert seqs == [0, 1, 2, 3, 4]


# ---------- EventBus: socket queue backpressure ----------

@pytest.mark.asyncio
async def test_socket_queue_drops_oldest_when_full(tmp_path: Path):
    # Max queue = 3. Emit 10. Expect 7 drops and the queue to contain the LAST 3.
    bus = EventBus(jsonl_path=tmp_path / "events.jsonl", max_queue=3)
    await bus.start()
    try:
        for i in range(10):
            await bus.emit(
                AgentLog(
                    seq=0, ts=_ts(), run_id=RUN_ID,
                    payload=AgentLogPayload(agent_id="a", level="info", msg=f"{i}"),
                )
            )
        # emit never blocks; queue holds 3; dropped counter = 7
        assert bus.dropped_socket_events == 7
        assert bus.socket_queue.qsize() == 3
        remaining_msgs = []
        while not bus.socket_queue.empty():
            ev = bus.socket_queue.get_nowait()
            remaining_msgs.append(ev.payload.msg)
        # last 3 emitted were msgs 7,8,9
        assert remaining_msgs == ["7", "8", "9"]
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_jsonl_is_authoritative_even_when_socket_overflows(tmp_path: Path):
    # Even if socket drops 7 of 10, the JSONL file must contain all 10.
    bus = EventBus(jsonl_path=tmp_path / "events.jsonl", max_queue=3)
    await bus.start()
    try:
        for i in range(10):
            await bus.emit(
                AgentLog(
                    seq=0, ts=_ts(), run_id=RUN_ID,
                    payload=AgentLogPayload(agent_id="a", level="info", msg=f"{i}"),
                )
            )
    finally:
        await bus.stop()

    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 10
    assert bus.dropped_socket_events == 7


@pytest.mark.asyncio
async def test_emit_never_blocks_core(tmp_path: Path):
    # Timeout guard: emitting many events with a full socket queue must return promptly.
    bus = EventBus(jsonl_path=tmp_path / "events.jsonl", max_queue=1)
    await bus.start()
    try:
        async def burst():
            for i in range(500):
                await bus.emit(
                    CostUpdate(
                        seq=0, ts=_ts(), run_id=RUN_ID,
                        payload=CostUpdatePayload(
                            cost_usd_total=0.01 * i,
                            tokens_in_total=100 * i,
                            tokens_out_total=50 * i,
                        ),
                    )
                )
        # If emit blocks, this will time out.
        await asyncio.wait_for(burst(), timeout=2.0)
        assert bus.dropped_socket_events == 499
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_run_complete_event(tmp_path: Path):
    # Sanity: the final event in a run is a RunComplete with a reason.
    bus = EventBus(jsonl_path=tmp_path / "events.jsonl")
    await bus.start()
    try:
        await bus.emit(
            RunComplete(
                seq=0, ts=_ts(), run_id=RUN_ID,
                payload=RunCompletePayload(
                    reason="plateau",
                    entities=120,
                    cost_usd=1.42,
                    wall_s=312.0,
                    db_path="/tmp/store.duckdb",
                ),
            )
        )
    finally:
        await bus.stop()

    line = (tmp_path / "events.jsonl").read_text().strip()
    data = json.loads(line)
    assert data["type"] == "run_complete"
    assert data["payload"]["reason"] == "plateau"
