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
from typing import Annotated, Any, Callable, Literal, Optional

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


class InterruptRequestedPayload(BaseModel):
    point: str
    context: dict[str, Any] = Field(default_factory=dict)


class InterruptResolvedPayload(BaseModel):
    point: str
    decision: Literal["continue", "abort"]


class CoverageReportPayload(BaseModel):
    """Structured coverage signal emitted after every cycle_end (v1.2 #8).

    Externalizes the plateau / coverage signal the orchestrator already
    computes internally so external tools can ingest a machine-readable
    snapshot per cycle: which entity types are sparse, which fields are
    underconfident, how many conflicts are still open, what kinds of
    sources are dominating, and a short heuristic suggestion list of
    "what to do next".
    """

    cycle: int
    entities_by_type: dict[str, int]
    fields_below_confidence: dict[str, int]
    confidence_threshold: float
    conflicts_open: int
    source_type_breakdown: dict[str, int]
    next_recommended_seeds: list[str]


class SourcePackPayload(BaseModel):
    sources: int
    chunks: int


class VerificationVotePayload(BaseModel):
    entity_id: str | None = None
    field: str
    model: str
    vote: Any
    confidence: float
    disagreement: bool = False


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


class InterruptRequested(_EventBase):
    type: Literal["interrupt_requested"] = "interrupt_requested"
    payload: InterruptRequestedPayload


class InterruptResolved(_EventBase):
    type: Literal["interrupt_resolved"] = "interrupt_resolved"
    payload: InterruptResolvedPayload


class CoverageReport(_EventBase):
    type: Literal["coverage_report"] = "coverage_report"
    payload: CoverageReportPayload


class SourcePackLoaded(_EventBase):
    type: Literal["source_pack_loaded"] = "source_pack_loaded"
    payload: SourcePackPayload


class VerificationVote(_EventBase):
    type: Literal["verification_vote"] = "verification_vote"
    payload: VerificationVotePayload


Event = Annotated[
    CycleStart | CycleEnd | AgentSpawn | AgentStateChange | AgentLog | FactWritten | ConflictDetected | ConflictResolved | CostUpdate | BudgetWarning | RunComplete | SubagentCall | InterruptRequested | InterruptResolved | CoverageReport | SourcePackLoaded | VerificationVote,
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
        # v1.3 #4: synchronous sideband subscribers (e.g. OTEL adapter). Each
        # callback receives the emitted event AFTER the JSONL write and the
        # socket-queue put. Exceptions inside a subscriber are swallowed and
        # counted — a buggy sink must NEVER block or crash the core emit path.
        self._subscribers: list[Callable[[Event], None]] = []
        self._subscriber_errors: int = 0

    def add_subscriber(self, callback: Callable[[Event], None]) -> None:
        """Register a synchronous callback invoked after every emit.

        Subscribers are intended for non-critical observers (OTLP
        exporter, live TUI, etc.). The callback is called synchronously
        inside :meth:`emit`, so it must be cheap and non-blocking. Any
        exception raised by a subscriber is caught and counted on
        ``_subscriber_errors``; it never bubbles to the core.
        """
        self._subscribers.append(callback)

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

        # 3) Sideband subscribers (v1.3 #4). Exceptions swallowed + counted.
        for sub in self._subscribers:
            try:
                sub(event)
            except Exception:
                self._subscriber_errors += 1

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


class EventSocketServer:
    """Unix-socket broadcaster for live TUI attach.

    The EventBus remains authoritative via JSONL. This server is a best-effort
    live view over the bus' bounded socket queue: every queued event is written
    as one JSON line to all currently connected clients. Slow or broken clients
    are dropped so they cannot block the orchestrator.
    """

    def __init__(self, bus: EventBus, socket_path: Path | str) -> None:
        self._bus = bus
        self._socket_path = Path(socket_path)
        self._server: asyncio.AbstractServer | None = None
        self._pump_task: asyncio.Task | None = None
        self._clients: set[asyncio.StreamWriter] = set()

    async def start(self) -> None:
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._socket_path.unlink()
        except FileNotFoundError:
            pass
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._socket_path),
        )
        self._pump_task = asyncio.create_task(self._pump())

    async def stop(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
            self._pump_task = None
        for writer in list(self._clients):
            await self._close_writer(writer)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        try:
            self._socket_path.unlink()
        except FileNotFoundError:
            pass

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._clients.add(writer)
        try:
            await reader.read()
        finally:
            self._clients.discard(writer)
            writer.close()

    async def _pump(self) -> None:
        while True:
            event = await self._bus.socket_queue.get()
            while not self._clients:
                await asyncio.sleep(0.01)
            line = (event.model_dump_json() + "\n").encode("utf-8")
            dead: list[asyncio.StreamWriter] = []
            for writer in list(self._clients):
                try:
                    writer.write(line)
                    await writer.drain()
                except Exception:
                    dead.append(writer)
            for writer in dead:
                await self._close_writer(writer)
            self._bus.socket_queue.task_done()

    async def _close_writer(self, writer: asyncio.StreamWriter) -> None:
        self._clients.discard(writer)
        try:
            writer.close()
            await asyncio.wait_for(writer.wait_closed(), timeout=0.5)
        except Exception:
            pass
