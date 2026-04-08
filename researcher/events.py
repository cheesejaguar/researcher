"""Event schema + EventBus.

The event bus is the core's only outward communication channel to observers (TUI, CLI,
replay tools). JSONL on disk is the authoritative record; a bounded socket queue is a
best-effort live tail. On queue overflow, oldest events are dropped (never newest, which
are most relevant) and a counter is incremented. Emit must NEVER block the core.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------- Payloads ----------


class CycleStartPayload(BaseModel):
    cycle: int
    pending_tasks: int


class CycleEndPayload(BaseModel):
    cycle: int
    claims_written: int
    entities_total: int
    cost_usd: float


class AgentSpawnPayload(BaseModel):
    agent_id: str
    task_id: str
    kind: str


class AgentStateChangePayload(BaseModel):
    agent_id: str
    old: str
    new: str


class AgentLogPayload(BaseModel):
    agent_id: str
    level: Literal["debug", "info", "warn", "error"]
    msg: str


class FactWrittenPayload(BaseModel):
    entity_id: str
    entity_type: str
    field: str
    value: Any
    confidence: float
    source_url: str


class ConflictPayload(BaseModel):
    entity_id: str
    field: str
    winning_value: Any = None
    losing_value: Any = None
    reason: str


class CostUpdatePayload(BaseModel):
    cost_usd_total: float
    tokens_in_total: int
    tokens_out_total: int


class BudgetWarningPayload(BaseModel):
    cost_usd_total: float
    budget_usd: float
    fraction: float  # 0..1


class RunCompletePayload(BaseModel):
    reason: Literal[
        "budget", "plateau", "ctrl_c", "error", "deadline", "no_tasks", "subagent_cap"
    ]
    entities: int
    cost_usd: float
    wall_s: float
    db_path: str


class SubagentCallPayload(BaseModel):
    agent_id: str
    task_id: str
    cli_kind: Literal["claude_code", "codex"]
    wall_ms: int
    exit_code: int | None
    claims_emitted: int


# ---------- Event envelope ----------


class _EventBase(BaseModel):
    seq: int
    ts: datetime
    run_id: str


class CycleStart(_EventBase):
    type: Literal["cycle_start"] = "cycle_start"
    payload: CycleStartPayload


class CycleEnd(_EventBase):
    type: Literal["cycle_end"] = "cycle_end"
    payload: CycleEndPayload


class AgentSpawn(_EventBase):
    type: Literal["agent_spawn"] = "agent_spawn"
    payload: AgentSpawnPayload


class AgentStateChange(_EventBase):
    type: Literal["agent_state_change"] = "agent_state_change"
    payload: AgentStateChangePayload


class AgentLog(_EventBase):
    type: Literal["agent_log"] = "agent_log"
    payload: AgentLogPayload


class FactWritten(_EventBase):
    type: Literal["fact_written"] = "fact_written"
    payload: FactWrittenPayload


class ConflictDetected(_EventBase):
    type: Literal["conflict_detected"] = "conflict_detected"
    payload: ConflictPayload


class ConflictResolved(_EventBase):
    type: Literal["conflict_resolved"] = "conflict_resolved"
    payload: ConflictPayload


class CostUpdate(_EventBase):
    type: Literal["cost_update"] = "cost_update"
    payload: CostUpdatePayload


class BudgetWarning(_EventBase):
    type: Literal["budget_warning"] = "budget_warning"
    payload: BudgetWarningPayload


class RunComplete(_EventBase):
    type: Literal["run_complete"] = "run_complete"
    payload: RunCompletePayload


class SubagentCall(_EventBase):
    type: Literal["subagent_call"] = "subagent_call"
    payload: SubagentCallPayload


Event = Annotated[
    CycleStart | CycleEnd | AgentSpawn | AgentStateChange | AgentLog | FactWritten | ConflictDetected | ConflictResolved | CostUpdate | BudgetWarning | RunComplete | SubagentCall,
    Field(discriminator="type"),
]


# A small adapter-backed parser for raw dicts coming off the wire or a file.
class _EventEnvelope(BaseModel):
    event: Event


def parse_event(data: dict) -> Event:
    """Parse a raw dict into the correct Event subclass by discriminator."""
    return _EventEnvelope(event=data).event


# ---------- EventBus ----------


class EventBus:
    """Emits events to a JSONL file (authoritative) and an in-process bounded socket queue.

    The socket queue is drained by a transport (Unix socket server, stdout tailer, etc.)
    living elsewhere; the bus itself is transport-agnostic. On queue overflow, oldest
    events are dropped and `dropped_socket_events` is incremented — emit never blocks.
    """

    def __init__(
        self,
        jsonl_path: Path | str,
        socket_path: Optional[str] = None,
        max_queue: int = 1024,
    ) -> None:
        self._jsonl_path = Path(jsonl_path)
        self._socket_path = socket_path
        self._max_queue = max_queue
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue)
        self._seq = 0
        self._dropped = 0
        self._file = None  # type: ignore[assignment]
        self._started = False

    async def start(self) -> None:
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._jsonl_path, "a", buffering=1, encoding="utf-8")
        self._started = True

    async def stop(self) -> None:
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None
        self._started = False

    async def emit(self, event: Event) -> None:
        """Write to JSONL, then best-effort enqueue to the socket queue.

        Never blocks: if the queue is full, drops the oldest event and increments
        `dropped_socket_events`, then enqueues the new one.
        """
        if not self._started or self._file is None:
            raise RuntimeError("EventBus.emit called before start()")
        # Assign seq monotonically (overrides whatever the caller passed in).
        event = event.model_copy(update={"seq": self._seq})
        self._seq += 1

        # 1) JSONL (authoritative).
        line = event.model_dump_json() + "\n"
        self._file.write(line)
        # NOTE: for v1 we flush on every emit. fsync is expensive; defer batched
        # fsync behind a perf flag once benchmarks show it matters.
        self._file.flush()

        # 2) Socket queue (best effort, bounded, drop oldest).
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._dropped += 1
            except asyncio.QueueEmpty:
                pass
            # Retry put — should succeed since we just removed one.
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                # Extremely degenerate case (concurrent producers) — drop and move on.
                self._dropped += 1

    @property
    def socket_queue(self) -> asyncio.Queue:
        """The bounded in-process queue that a socket transport should drain."""
        return self._queue

    @property
    def dropped_socket_events(self) -> int:
        return self._dropped

    @property
    def next_seq(self) -> int:
        return self._seq
