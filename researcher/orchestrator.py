"""Orchestrator — the cycle loop that coordinates map (agents) → reduce (writer).

Wave 0 provides a skeleton cycle loop that compiles and runs against stub
components. Wave 1-E replaces the agent-spawn code path with real agent
dispatch and tunes the graceful-drain protocol.

Cycle loop:
  1. scheduler.next_batch(max_n)           — plan
  2. emit cycle_start
  3. spawn agents (bounded, parallel map)  — execute
  4. await all agents
  5. writer.drain()                        — serial reduce
  6. scheduler.update_from_metrics(...)
  7. emit cycle_end
  8. check budget / plateau / deadline     — maybe break
  9. on stop: _graceful_stop(reason)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

from researcher.budget import Budget, BudgetExceededError
from researcher.events import (
    CycleEnd,
    CycleEndPayload,
    CycleStart,
    CycleStartPayload,
    EventBus,
    RunComplete,
    RunCompletePayload,
)
from researcher.llm.client import LLMClient
from researcher.models import AgentResult, Task
from researcher.scheduler import Scheduler
from researcher.spec import RunSpec
from researcher.storage.store import KnowledgeStore
from researcher.storage.writer import FactWriter
from researcher.backends.cli_runner import CliRunner
from researcher.backends.models import BackendChoice, CliKind
from researcher.backends.resolver import BackendResolver


class StopReason(str, Enum):
    BUDGET = "budget"
    PLATEAU = "plateau"
    DEADLINE = "deadline"
    CTRL_C = "ctrl_c"
    ERROR = "error"
    NO_TASKS = "no_tasks"
    SUBAGENT_CAP = "subagent_cap"


class Orchestrator:
    def __init__(
        self,
        spec: RunSpec,
        store: KnowledgeStore,
        llm: LLMClient,
        bus: EventBus,
        writer: FactWriter,
        scheduler: Scheduler,
        budget: Budget,
        run_id: str,
        max_parallel_agents: int = 6,
    ) -> None:
        self._spec = spec
        self._store = store
        self._llm = llm
        self._bus = bus
        self._writer = writer
        self._scheduler = scheduler
        self._budget = budget
        self._run_id = run_id
        self._sem = asyncio.Semaphore(max_parallel_agents)
        self._cycle = 0
        self._started_at: Optional[datetime] = None
        self._backend_resolver: Optional[BackendResolver] = None
        self._cli_runner_factory: Optional[Callable[[CliKind], CliRunner]] = None
        self._cycle_subagent_failures: int = 0
        self._circuit_break_threshold: int = 3

    def set_backend_resolver(self, resolver: BackendResolver) -> None:
        self._backend_resolver = resolver

    def set_cli_runner_factory(
        self, factory: Callable[[CliKind], CliRunner]
    ) -> None:
        self._cli_runner_factory = factory

    def _resolver_pick(self, task: Task) -> BackendChoice:
        if self._backend_resolver is None:
            return BackendChoice(kind=None, reason="no resolver configured")
        return self._backend_resolver.pick(policy=self._spec.backend_policy)  # type: ignore[arg-type]

    def _record_subagent_failure(self, error: str) -> None:
        """Update the per-cycle failure counter and trip circuit break if warranted."""
        if self._backend_resolver is None:
            return
        # Immediate trip on auth / usage errors.
        if error == "auth_required":
            self._backend_resolver.clear_detected("auth_required")
            return
        if error == "usage_limit_reached":
            self._backend_resolver.clear_detected("usage_limit_reached")
            return
        # Count other failures; trip after threshold.
        self._cycle_subagent_failures += 1
        if self._cycle_subagent_failures >= self._circuit_break_threshold:
            self._backend_resolver.clear_detected(
                f"{self._cycle_subagent_failures} failures in cycle"
            )

    def _reset_cycle_failure_counters(self) -> None:
        self._cycle_subagent_failures = 0

    async def run(self) -> StopReason:
        self._started_at = datetime.now(timezone.utc)
        self._budget.start_wall_clock()
        await self._scheduler.seed()

        reason: StopReason = StopReason.NO_TASKS
        try:
            while self._cycle < self._spec.max_cycles:
                self._cycle += 1
                batch = await self._scheduler.next_batch(self._spec.max_entities_per_cycle)
                if not batch:
                    reason = StopReason.NO_TASKS
                    break

                await self._emit_cycle_start(len(batch))

                # Parallel map.
                results = await asyncio.gather(
                    *(self._spawn_agent(t) for t in batch), return_exceptions=True
                )
                for t, r in zip(batch, results):
                    if isinstance(r, Exception):
                        continue
                    await self._scheduler.mark_done(t.id, r)

                # Serial reduce: drain writer queue.
                # NOTE: Wave 1-E replaces this with a per-cycle drain that keeps
                # the writer running across cycles via a sentinel boundary.
                # For Wave 0 this is a placeholder.

                metrics = await self._store.snapshot_metrics()
                await self._scheduler.update_from_metrics(metrics)
                await self._emit_cycle_end(metrics)

                # Stop checks.
                if self._budget.exceeded():
                    reason = StopReason.BUDGET
                    break
                if self._budget.wall_exceeded():
                    reason = StopReason.DEADLINE
                    break
                if self._scheduler.plateau:
                    reason = StopReason.PLATEAU
                    break
            else:
                # Exited via max_cycles limit — treat as plateau.
                reason = StopReason.PLATEAU
        except BudgetExceededError:
            reason = StopReason.BUDGET
        except Exception:
            reason = StopReason.ERROR
            raise
        finally:
            await self._graceful_stop(reason)

        return reason

    async def _spawn_agent(self, task: Task) -> AgentResult:
        """Wave 0 placeholder: actual agent dispatch lands in Wave 1-E."""
        async with self._sem:
            # Wave 1-E: look up agent class by TaskKind, instantiate, call .run(task).
            from researcher.models import AgentState

            return AgentResult(task_id=task.id, agent_id="stub", state=AgentState.DONE)

    async def _emit_cycle_start(self, pending: int) -> None:
        await self._bus.emit(
            CycleStart(
                seq=0,
                ts=datetime.now(timezone.utc),
                run_id=self._run_id,
                payload=CycleStartPayload(cycle=self._cycle, pending_tasks=pending),
            )
        )

    async def _emit_cycle_end(self, metrics) -> None:
        await self._bus.emit(
            CycleEnd(
                seq=0,
                ts=datetime.now(timezone.utc),
                run_id=self._run_id,
                payload=CycleEndPayload(
                    cycle=self._cycle,
                    claims_written=self._writer.metrics.get("written", 0),
                    entities_total=metrics.entities_total,
                    cost_usd=metrics.cost_usd_total,
                ),
            )
        )

    async def _graceful_stop(self, reason: StopReason) -> None:
        """Drain the writer, write run_summary, emit run_complete.

        Wave 1-E adds the per-task cancellation grace window and the
        provisional-claims discard policy. Wave 0 keeps it simple.
        """
        try:
            metrics = await self._store.snapshot_metrics()
        except Exception:
            metrics = None

        wall_s = 0.0
        if self._started_at is not None:
            wall_s = (datetime.now(timezone.utc) - self._started_at).total_seconds()

        await self._bus.emit(
            RunComplete(
                seq=0,
                ts=datetime.now(timezone.utc),
                run_id=self._run_id,
                payload=RunCompletePayload(
                    reason=reason.value,  # type: ignore[arg-type]
                    entities=(metrics.entities_total if metrics else 0),
                    cost_usd=(metrics.cost_usd_total if metrics else self._budget.total_spent()),
                    wall_s=wall_s,
                    db_path="",
                ),
            )
        )
