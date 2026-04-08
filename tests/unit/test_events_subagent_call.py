"""Tests for the SubagentCall event."""

import json
from datetime import datetime, timezone

from researcher.events import (
    SubagentCall,
    SubagentCallPayload,
    parse_event,
)


def _ts() -> datetime:
    return datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)


def test_subagent_call_has_type_discriminator():
    e = SubagentCall(
        seq=0,
        ts=_ts(),
        run_id="run-x",
        payload=SubagentCallPayload(
            agent_id="a1",
            task_id="t1",
            cli_kind="claude_code",
            wall_ms=4321,
            exit_code=0,
            claims_emitted=4,
        ),
    )
    assert e.type == "subagent_call"


def test_subagent_call_roundtrip_through_json():
    e = SubagentCall(
        seq=7,
        ts=_ts(),
        run_id="run-x",
        payload=SubagentCallPayload(
            agent_id="a1",
            task_id="t1",
            cli_kind="codex",
            wall_ms=1200,
            exit_code=0,
            claims_emitted=2,
        ),
    )
    data = json.loads(e.model_dump_json())
    parsed = parse_event(data)
    assert isinstance(parsed, SubagentCall)
    assert parsed.payload.cli_kind == "codex"
    assert parsed.payload.claims_emitted == 2


def test_parse_event_dispatches_subagent_call_from_wire():
    raw = {
        "type": "subagent_call",
        "seq": 9,
        "ts": _ts().isoformat(),
        "run_id": "run-x",
        "payload": {
            "agent_id": "a1",
            "task_id": "t1",
            "cli_kind": "claude_code",
            "wall_ms": 999,
            "exit_code": 1,
            "claims_emitted": 0,
        },
    }
    e = parse_event(raw)
    assert isinstance(e, SubagentCall)
    assert e.payload.exit_code == 1
