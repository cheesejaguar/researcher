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
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from researcher.agents.native_deps import NativeAgentDeps
from researcher.agents.subagent import SubagentResearcher
from researcher.backends.cli_runner import CliRunner
from researcher.backends.models import BackendChoice, CliKind
from researcher.backends.resolver import BackendResolver
from researcher.budget import Budget, BudgetExceededError
from researcher.events import (
    AgentSpawn,
    AgentSpawnPayload,
    BudgetWarning,
    BudgetWarningPayload,
    CostUpdate,
    CostUpdatePayload,
    CoverageReport,
    CoverageReportPayload,
    CycleEnd,
    CycleEndPayload,
    CycleStart,
    CycleStartPayload,
    EventBus,
    InterruptRequested,
    InterruptRequestedPayload,
    InterruptResolved,
    InterruptResolvedPayload,
    RunComplete,
    RunCompletePayload,
    VerificationVote,
    VerificationVotePayload,
)
from researcher.interrupts import InterruptHandler
from researcher.llm.client import LLMClient
from researcher.models import AgentResult, AgentState, Task, TaskKind
from researcher.scheduler import Scheduler
from researcher.skills.registry import SkillRegistry
from researcher.spec import RunSpec
from researcher.storage.store import CoverageSnapshot, KnowledgeStore, StoreMetrics
from researcher.storage.writer import FactWriter


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
        obsidian_writer: Any = None,  # Optional[ObsidianWriter]; avoid import cycle
        resume_from: Optional[int] = None,
        coverage_confidence_threshold: float = 0.5,
        db_path: str = "",
        stop_file: str = "",
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
        self._cycle = resume_from if resume_from is not None else 0
        self._started_at: Optional[datetime] = None
        self._backend_resolver: Optional[BackendResolver] = None
        self._cli_runner_factory: Optional[Callable[[CliKind], CliRunner]] = None
        self._cycle_subagent_failures: int = 0
        self._circuit_break_threshold: int = 3
        self._obsidian_writer = obsidian_writer
        self._native_deps: Optional[NativeAgentDeps] = None
        self._skill_registry: SkillRegistry | None = None
        self._resume_from = resume_from
        self._interrupt_handler: Optional[InterruptHandler] = None
        # v1.2: optional difficulty-aware compute gate (opt-in).
        self._difficulty_gate: Any = None
        # v1.2 #8: confidence threshold below which a field cell counts
        # as "low confidence" in the structured CoverageReport event.
        self._coverage_confidence_threshold = float(coverage_confidence_threshold)
        self._db_path = db_path
        self._stop_file = Path(stop_file) if stop_file else None

    def set_backend_resolver(self, resolver: BackendResolver) -> None:
        self._backend_resolver = resolver

    def set_cli_runner_factory(
        self, factory: Callable[[CliKind], CliRunner]
    ) -> None:
        self._cli_runner_factory = factory

    def set_difficulty_gate(self, gate: Any) -> None:
        """Register a difficulty-aware compute gate (v1.2, opt-in).

        When set, :meth:`_spawn_agent` consults the gate before dispatching
        each task. The gate rates the task 1-5 and the orchestrator uses
        :meth:`Budget.allows_task_with_difficulty` to enforce a
        difficulty-scaled per-task cap. Gate exceptions fail open — the task
        dispatches normally.

        Per CODA (arXiv 2603.08659) and TALE-EP.
        """
        self._difficulty_gate = gate

    def set_interrupt_handler(self, handler: InterruptHandler) -> None:
        """Register a human-in-the-loop handler.

        The handler is consulted at each point listed in
        ``spec.interrupt_points``. Without a handler (or when the spec
        declares no points) the orchestrator runs to completion exactly
        as before — the feature is strictly opt-in.
        """
        self._interrupt_handler = handler

    async def _maybe_interrupt(
        self, point: str, context: dict[str, Any]
    ) -> bool:
        """Consult the interrupt handler at a named point.

        Returns ``True`` if the run should continue, ``False`` to abort.
        Emits ``InterruptRequested`` and ``InterruptResolved`` events on the
        bus so observers (TUI, replay tools) can see the pause.
        """
        if self._interrupt_handler is None:
            return True
        spec_points = getattr(self._spec, "interrupt_points", [])
        if point not in spec_points:
            return True
        await self._bus.emit(
            InterruptRequested(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=InterruptRequestedPayload(
                    point=point, context=context
                ),
            )
        )
        try:
            decision = await self._interrupt_handler.handle(point, context)
        except Exception:
            decision_str = "continue"
        else:
            decision_str = (
                decision.value if hasattr(decision, "value") else str(decision)
            )
        await self._bus.emit(
            InterruptResolved(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=InterruptResolvedPayload(
                    point=point,
                    decision=decision_str,  # type: ignore[arg-type]
                ),
            )
        )
        return decision_str == "continue"

    def set_native_deps(self, deps: NativeAgentDeps) -> None:
        """Opt in to the native-agent path by providing shared deps.

        Without this call, `_spawn_agent` still raises NotImplementedError
        on the native branch — existing tests that never set deps keep their
        old behavior, while new code can opt in explicitly.
        """
        self._native_deps = deps
        self._skill_registry = deps.skill_registry

    def set_skill_registry(self, registry: SkillRegistry | None) -> None:
        """Register domain skill cards for the CLI subagent path."""
        self._skill_registry = registry

    def _stop_requested(self) -> bool:
        return bool(self._stop_file and self._stop_file.exists())

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
        self._started_at = datetime.now(UTC)
        self._budget.start_wall_clock()
        if self._resume_from is None:
            # Fresh run — seed the scheduler from spec.
            await self._scheduler.seed()
        # On resume the scheduler + budget were restored by the caller
        # before run() was invoked, so skip seeding to avoid duplicating
        # the initial DISCOVER tasks.

        # Start the writer drain loop for the whole run.
        self._writer.start()

        if self._obsidian_writer is not None:
            await self._obsidian_writer.start()

        reason: StopReason = StopReason.NO_TASKS
        try:
            # Optional HITL pause after seeding, before cycle 1. Checked even
            # on resume so a restarted run can re-confirm before proceeding.
            seed_ok = await self._maybe_interrupt(
                "after_initial_seed",
                {"pending_tasks": self._scheduler.pending},
            )
            if not seed_ok:
                reason = StopReason.CTRL_C
                # Skip the cycle loop entirely; the finally block below still
                # runs drain + graceful_stop.
                return reason
            while self._cycle < self._spec.max_cycles:
                if self._stop_requested():
                    reason = StopReason.CTRL_C
                    break
                self._cycle += 1
                self._reset_cycle_failure_counters()

                batch = await self._scheduler.next_batch(self._spec.max_entities_per_cycle)
                if not batch:
                    reason = StopReason.NO_TASKS
                    break

                await self._emit_cycle_start(len(batch))

                # Parallel map.
                results = await asyncio.gather(
                    *(self._spawn_agent(t) for t in batch), return_exceptions=True
                )

                subagent_cap_hit = False
                saw_real_exception = False
                to_revise: list[Task] = []
                for t, r in zip(batch, results, strict=False):
                    if isinstance(r, Exception):
                        saw_real_exception = True
                        continue
                    await self._record_result_cost(r)
                    # Submit successful claims to the writer.
                    for claim in r.claims:
                        try:
                            await self._writer.submit(claim)
                        except asyncio.QueueFull:
                            pass  # drops counted in writer.metrics
                        # Secondary sink — Obsidian. Never let failures here
                        # disrupt the primary path.
                        if self._obsidian_writer is not None:
                            try:
                                await self._obsidian_writer.on_fact(
                                    claim, run_id=self._run_id
                                )
                            except Exception:
                                pass
                    # v1.2: collect tasks tagged for bounded conditional
                    # revision. Opt-in via spec flag; capped at one retry
                    # per task by requiring ``attempt == 0``.
                    if (
                        self._spec.enable_conditional_revision
                        and getattr(r, "needs_revision", False)
                        and t.attempt == 0
                    ):
                        self._budget.record_revision()
                        to_revise.append(
                            t.model_copy(update={"attempt": t.attempt + 1})
                        )
                    # Record failures for circuit-break counter + detect subagent_cap.
                    if r.state == AgentState.FAILED and r.error:
                        if r.error == "subagent_cap":
                            subagent_cap_hit = True
                        else:
                            self._record_subagent_failure(r.error)
                    await self._scheduler.mark_done(t.id, r)

                if saw_real_exception:
                    reason = StopReason.ERROR
                    break

                # v1.2: bounded conditional revision.
                # Re-dispatch tagged tasks exactly once within the same
                # cycle. Cost is bounded by construction: at most one
                # extra dispatch per task per run.
                if to_revise:
                    revision_results = await asyncio.gather(
                        *(self._spawn_agent(t) for t in to_revise),
                        return_exceptions=True,
                    )
                    for r in revision_results:
                        if isinstance(r, Exception):
                            continue
                        await self._record_result_cost(r)
                        for claim in r.claims:
                            try:
                                await self._writer.submit(claim)
                            except asyncio.QueueFull:
                                pass
                            if self._obsidian_writer is not None:
                                try:
                                    await self._obsidian_writer.on_fact(
                                        claim, run_id=self._run_id
                                    )
                                except Exception:
                                    pass
                        if r.state == AgentState.FAILED and r.error:
                            if r.error == "subagent_cap":
                                subagent_cap_hit = True
                            else:
                                self._record_subagent_failure(r.error)

                # Serial reduce: wait for this cycle's claims to process.
                await self._writer.quiesce()

                if self._obsidian_writer is not None:
                    try:
                        await self._obsidian_writer.flush()
                    except Exception:
                        pass

                metrics = await self._store.snapshot_metrics()
                await self._sync_budget_from_metrics(metrics)
                await self._scheduler.update_from_metrics(metrics)
                await self._emit_cycle_end(metrics)
                await self._maybe_emit_coverage_report(metrics)

                # Crash-safe checkpoint: snapshot scheduler + budget state
                # so a crash here resumes from the next cycle, not cycle 0.
                await self._save_checkpoint()

                # Optional HITL pause at cycle boundary. Runs before budget /
                # plateau checks so an explicit user abort takes precedence
                # over graceful termination reasons.
                cycle_ok = await self._maybe_interrupt(
                    "after_cycle_end",
                    {
                        "cycle": self._cycle,
                        "entities_total": metrics.entities_total,
                        "cost_usd": metrics.cost_usd_total,
                    },
                )
                if not cycle_ok:
                    reason = StopReason.CTRL_C
                    break

                # Stop checks — subagent_cap is highest priority because it's
                # a hard stop regardless of other state.
                if self._stop_requested():
                    reason = StopReason.CTRL_C
                    break
                if subagent_cap_hit:
                    reason = StopReason.SUBAGENT_CAP
                    break
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
            # Don't re-raise: _graceful_stop still needs to run.
        finally:
            # Drain the writer before emitting run_complete.
            try:
                await self._writer.drain()
            except Exception:
                pass
            if self._obsidian_writer is not None:
                # Post-flush surface-up hooks: wrap known-entity names in
                # [[wikilinks]] and emit per-type _index.md Dataview notes.
                # Best-effort: a stub store that doesn't implement `query`
                # (NotImplementedError) simply skips wikilink injection.
                try:
                    rows = await self._store.query(  # type: ignore[attr-defined]
                        "SELECT name FROM entities"
                    )
                    known = {r["name"] for r in rows if r.get("name")}
                    if known:
                        await self._obsidian_writer.inject_wikilinks(known)
                except Exception:
                    pass
                try:
                    await self._obsidian_writer.write_index_notes()
                except Exception:
                    pass
                try:
                    await self._obsidian_writer.stop()
                except Exception:
                    pass
            await self._graceful_stop(reason)

        return reason

    async def _spawn_agent(self, task: Task) -> AgentResult:
        """Dispatch a task to either the subagent path or the native path.

        Wave 1-D wires both paths. The native path is opt-in: the caller must
        install shared deps via `set_native_deps()` before `run()`. Without
        that, the native branch preserves the old NotImplementedError behavior
        for backwards compatibility.
        """
        async with self._sem:
            # v1.2: difficulty-aware compute gate (opt-in).
            # Consult the gate before dispatching; skip tasks that exceed
            # their difficulty-scaled per-task cap. Fail open on gate errors.
            if self._difficulty_gate is not None:
                try:
                    estimate = await self._difficulty_gate.estimate(task)
                    estimated_cost = task.budget_usd
                    if not self._budget.allows_task_with_difficulty(
                        estimated_cost, difficulty=estimate.difficulty
                    ):
                        return AgentResult(
                            task_id=task.id,
                            agent_id=f"gated-{task.id[:8]}",
                            state=AgentState.FAILED,
                            error=f"budget_gated_difficulty_{estimate.difficulty}",
                        )
                except Exception:
                    pass  # gate failure → fail open, dispatch normally
            choice = self._resolver_pick(task)
            if choice.kind is None:
                await self._emit_agent_spawn(
                    task=task,
                    agent_id=f"native-{task.id[:8]}",
                    kind=task.kind.value,
                )
                return await self._spawn_native_agent(task)
            if self._cli_runner_factory is None:
                raise RuntimeError(
                    "Orchestrator has a backend resolver but no CLI runner factory. "
                    "Call set_cli_runner_factory() before run()."
                )

            runner = self._cli_runner_factory(choice.kind)
            entity_type_dict = {"entity_type": "Entity", "fields": []}
            if self._spec.entities:
                primary = self._spec.entities[0]
                entity_type_dict = {
                    "entity_type": primary.name,
                    "fields": [
                        {"name": f.name, "type": f.type, "required": f.required, "enum": list(f.enum)}
                        for f in primary.fields
                    ],
                }

            await self._emit_agent_spawn(
                task=task,
                agent_id=f"sub-{task.id[:8]}",
                kind="subagent",
            )
            agent = SubagentResearcher(
                agent_id=f"sub-{task.id[:8]}",
                llm=self._llm,
                store=self._store,
                emit=self._bus.emit,
                run_id=self._run_id,
                runner=runner,
                cli_kind=choice.kind,
                entity_schema=entity_type_dict,
                budget=self._budget,
                goal=self._spec.goal,
                max_entities=self._spec.max_entities_per_subagent_call,
                timeout_s=float(self._spec.subagent_timeout_s),
                skill_registry=self._skill_registry,
            )
            return await agent.run(task)

    async def _emit_agent_spawn(self, task: Task, agent_id: str, kind: str) -> None:
        await self._bus.emit(
            AgentSpawn(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=AgentSpawnPayload(
                    agent_id=agent_id,
                    task_id=task.id,
                    kind=kind,
                ),
            )
        )

    async def _record_result_cost(self, result: AgentResult) -> None:
        if result.cost_usd <= 0:
            return
        self._budget.spend(float(result.cost_usd))
        await self._emit_budget_updates()

    async def _sync_budget_from_metrics(self, metrics: StoreMetrics) -> None:
        observed = float(metrics.cost_usd_total)
        delta = observed - self._budget.total_spent()
        if delta > 0:
            self._budget.spend(delta)
        await self._emit_budget_updates()

    async def _emit_budget_updates(self) -> None:
        tokens_in = 0
        tokens_out = 0
        tracker = getattr(self._llm, "cost_tracker", None)
        if tracker is not None:
            tokens_in = int(getattr(tracker, "total_tokens_in", 0) or 0)
            tokens_out = int(getattr(tracker, "total_tokens_out", 0) or 0)
        await self._bus.emit(
            CostUpdate(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=CostUpdatePayload(
                    cost_usd_total=self._budget.total_spent(),
                    tokens_in_total=tokens_in,
                    tokens_out_total=tokens_out,
                ),
            )
        )
        warning = self._budget.should_warn()
        if warning is None:
            return
        await self._bus.emit(
            BudgetWarning(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=BudgetWarningPayload(
                    cost_usd_total=self._budget.total_spent(),
                    budget_usd=self._budget.cap,
                    fraction=warning,
                ),
            )
        )

    async def _spawn_native_agent(self, task: Task) -> AgentResult:
        """Dispatch a task to the appropriate native Agent subclass by TaskKind."""
        if self._native_deps is None:
            raise NotImplementedError(
                "Native agent path requires native_deps; call "
                "Orchestrator.set_native_deps(NativeAgentDeps(...)) before run(). "
                "Alternatively, set `backend_policy: cli` in the spec and install "
                "a CLI subagent on PATH."
            )

        # Deferred imports to avoid pulling native deps until they're used.
        from researcher.agents.critic import CriticAgent
        from researcher.agents.discover import DiscoverAgent
        from researcher.agents.enrich import EnrichAgent
        from researcher.agents.expand import ExpandAgent
        from researcher.agents.verify import VerifyAgent

        entity_schema_dict = {"entity_type": "Entity", "fields": []}
        if self._spec.entities:
            primary = self._spec.entities[0]
            entity_schema_dict = {
                "entity_type": primary.name,
                "fields": [
                    {"name": f.name, "type": f.type, "required": f.required, "enum": list(f.enum)}
                    for f in primary.fields
                ],
            }

        common_kwargs = dict(
            agent_id=f"native-{task.id[:8]}",
            llm=self._llm,
            store=self._store,
            emit=self._bus.emit,
            run_id=self._run_id,
            deps=self._native_deps,
            entity_schema=entity_schema_dict,
        )

        if task.kind == TaskKind.DISCOVER:
            agent: SubagentResearcher | DiscoverAgent | ExpandAgent | VerifyAgent | EnrichAgent | CriticAgent = (
                DiscoverAgent(**common_kwargs, goal=self._spec.goal)
            )
        elif task.kind == TaskKind.EXPAND:
            agent = ExpandAgent(**common_kwargs, goal=self._spec.goal)
        elif task.kind == TaskKind.VERIFY:
            agent = VerifyAgent(**common_kwargs)
        elif task.kind == TaskKind.ENRICH:
            agent = EnrichAgent(**common_kwargs, goal=self._spec.goal)
        else:
            raise NotImplementedError(
                f"unsupported TaskKind for native path: {task.kind}"
            )
        return await agent.run(task)

    async def _emit_cycle_start(self, pending: int) -> None:
        await self._bus.emit(
            CycleStart(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=CycleStartPayload(cycle=self._cycle, pending_tasks=pending),
            )
        )

    async def _emit_cycle_end(self, metrics) -> None:
        await self._bus.emit(
            CycleEnd(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=CycleEndPayload(
                    cycle=self._cycle,
                    claims_written=self._writer.metrics.get("written", 0),
                    entities_total=metrics.entities_total,
                    cost_usd=metrics.cost_usd_total,
                ),
            )
        )

    async def _maybe_emit_coverage_report(self, metrics: StoreMetrics) -> None:
        """Emit a structured CoverageReport after cycle_end (v1.2 #8).

        Best-effort: stores that don't implement ``snapshot_coverage``
        (older test stubs, the bare base class) are silently skipped.
        Like ``_save_checkpoint``, this is purely a derived signal — a
        failure here must never crash the run loop.
        """
        snap_fn = getattr(self._store, "snapshot_coverage", None)
        if snap_fn is None or not callable(snap_fn):
            return
        try:
            snapshot = await snap_fn(
                confidence_threshold=self._coverage_confidence_threshold
            )
        except Exception:
            return
        try:
            await self._emit_coverage_report(metrics, snapshot)
        except Exception:
            pass

    async def _emit_coverage_report(
        self, metrics: StoreMetrics, snapshot: CoverageSnapshot
    ) -> None:
        payload = CoverageReportPayload(
            cycle=self._cycle,
            entities_by_type=dict(metrics.by_type),
            fields_below_confidence=dict(snapshot.fields_below_confidence),
            confidence_threshold=self._coverage_confidence_threshold,
            conflicts_open=int(metrics.conflicts_open),
            source_type_breakdown=dict(snapshot.source_type_breakdown),
            next_recommended_seeds=self._recommend_next_seeds(metrics, snapshot),
        )
        await self._bus.emit(
            CoverageReport(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=payload,
            )
        )

    @staticmethod
    def _recommend_next_seeds(
        metrics: StoreMetrics, snapshot: CoverageSnapshot
    ) -> list[str]:
        """Heuristic v1 recommender: return up to 3 short suggestion strings.

        Pure function (no I/O). The TUI / external tools render these as
        a bullet list. The list is intentionally simple — sparse types,
        open conflicts, low-confidence fields — and capped at three to
        keep the signal scannable.
        """
        seeds: list[str] = []

        # Sparse types: any entity type with fewer than 5 entities.
        sparse = sorted(
            (t for t, n in metrics.by_type.items() if int(n) < 5),
            key=lambda t: (metrics.by_type.get(t, 0), t),
        )
        for t in sparse:
            if len(seeds) >= 3:
                break
            seeds.append(f"expand coverage of {t}")

        # Open conflicts: a single line with the count.
        if int(metrics.conflicts_open) > 0 and len(seeds) < 3:
            seeds.append(f"resolve {int(metrics.conflicts_open)} open conflicts")

        # Low-confidence fields: pick the top 2 by count.
        if snapshot.fields_below_confidence and len(seeds) < 3:
            top = sorted(
                snapshot.fields_below_confidence.items(),
                key=lambda kv: (-int(kv[1]), kv[0]),
            )[:2]
            field_names = ", ".join(name for name, _ in top)
            seeds.append(f"verify low-confidence fields: {field_names}")

        return seeds[:3]

    async def _save_checkpoint(self) -> None:
        """Persist the orchestrator's control state for crash-safe resume.

        Best effort: a checkpoint failure must never crash an in-progress
        run, since the checkpoint is purely an optimization for restart.
        Stores that don't implement ``save_checkpoint`` (e.g. the in-memory
        stub used by some tests) are silently skipped.
        """
        if not hasattr(self._store, "save_checkpoint"):
            return
        try:
            data = {
                "scheduler": self._scheduler.serialize(),
                "budget": self._budget.serialize(),
                "cycle": self._cycle,
                "started_at": (
                    self._started_at.isoformat()
                    if self._started_at is not None
                    else None
                ),
            }
            await self._store.save_checkpoint(  # type: ignore[attr-defined]
                run_id=self._run_id,
                cycle=self._cycle,
                data=data,
            )
        except Exception:
            # Swallow: a corrupt store must never block the run.
            pass

    async def _graceful_stop(self, reason: StopReason) -> None:
        """Drain the writer, write run_summary, emit run_complete.

        Wave 1-E adds the per-task cancellation grace window and the
        provisional-claims discard policy. Wave 0 keeps it simple.
        """
        try:
            metrics = await self._store.snapshot_metrics()
        except Exception:
            metrics = None
        else:
            try:
                await self._sync_budget_from_metrics(metrics)
            except Exception:
                pass

        wall_s = 0.0
        if self._started_at is not None:
            wall_s = (datetime.now(UTC) - self._started_at).total_seconds()

        summary = {
            "run_id": self._run_id,
            "spec_id": self._spec.spec_id,
            "goal": self._spec.goal,
            "reason": reason.value,
            "entities": metrics.entities_total if metrics else 0,
            "entities_by_type": dict(metrics.by_type) if metrics else {},
            "fields_filled_pct": metrics.fields_filled_pct if metrics else 0.0,
            "conflicts_open": metrics.conflicts_open if metrics else 0,
            "cost_usd": (
                metrics.cost_usd_total if metrics else self._budget.total_spent()
            ),
            "budget_spent_usd": self._budget.total_spent(),
            "wall_s": wall_s,
            "subagent_calls_total": self._budget.subagent_calls_total,
            "revisions_total": self._budget.revisions_total,
            "mode": getattr(self._spec, "mode", "overwrite"),
            "db_path": self._db_path,
        }
        if self._obsidian_writer is not None:
            try:
                summary["obsidian"] = {
                    "root_dir": str(self._obsidian_writer.root_dir),
                    "stats": self._obsidian_writer.stats,
                }
            except Exception:
                summary["obsidian"] = {"stats": {"errors": 1}}
        try:
            council = await self._maybe_record_council_votes()
            if council:
                summary["verification_council_votes"] = council
        except Exception:
            pass
        try:
            await self._store.write_run_summary(self._run_id, summary)
        except Exception:
            pass

        await self._bus.emit(
            RunComplete(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=RunCompletePayload(
                    reason=reason.value,  # type: ignore[arg-type]
                    entities=(metrics.entities_total if metrics else 0),
                    cost_usd=(metrics.cost_usd_total if metrics else self._budget.total_spent()),
                    wall_s=wall_s,
                    db_path=self._db_path,
                ),
            )
        )

    async def _maybe_record_council_votes(self) -> int:
        verification = getattr(self._spec, "verification", None)
        if verification is None or getattr(verification, "mode", "standard") != "council":
            return 0
        models = list(getattr(verification, "models", []) or [])
        if not models:
            models = [
                self._spec.models.get("fast")
                or self._spec.models.get("smart")
                or "council"
            ]
        threshold = float(getattr(verification, "confidence_threshold", 0.65))
        required = {
            f.name
            for ent in self._spec.entities
            for f in ent.fields
            if getattr(f, "required", False)
        }
        rows = await self._store.query(  # type: ignore[attr-defined]
            """
            SELECT e.id AS entity_id, f.field_name AS field_name,
                   f.value_json AS value_json, f.confidence AS confidence
            FROM fields f
            JOIN entities e ON e.id = f.entity_id
            WHERE f.value_json != 'null'
              AND (f.confidence < ? OR f.field_name IN (
            """
            + ",".join(["?"] * max(len(required), 1))
            + "))",
            tuple([threshold, *(required or {"__none__"})]),
        )
        count = 0
        for row in rows:
            value = row.get("value_json")
            try:
                value = __import__("json").loads(value)
            except Exception:
                pass
            for model in models:
                confidence = float(row.get("confidence") or 0.0)
                disagreement = confidence < threshold
                vote = {
                    "vote_id": uuid4().hex,
                    "run_id": self._run_id,
                    "entity_id": row.get("entity_id"),
                    "field_name": row.get("field_name"),
                    "model": model,
                    "vote": value,
                    "confidence": confidence,
                    "rationale": (
                        "low-confidence field queued for review"
                        if disagreement
                        else "required field accepted"
                    ),
                }
                await self._store.record_verification_vote(vote)
                await self._bus.emit(
                    VerificationVote(
                        seq=0,
                        ts=datetime.now(UTC),
                        run_id=self._run_id,
                        payload=VerificationVotePayload(
                            entity_id=vote["entity_id"],
                            field=vote["field_name"],
                            model=model,
                            vote=value,
                            confidence=confidence,
                            disagreement=disagreement,
                        ),
                    )
                )
                count += 1
        return count
