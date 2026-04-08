"""EnrichAgent — follow relations from a known entity.

Wave 1-D native agent. Given a Task with target_entity_id, asks the LLM to
suggest related entities worth researching next and emits new DISCOVER tasks
for each via AgentResult.spawned_tasks.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

from researcher.agents.base import Agent, EventEmitter
from researcher.agents.native_deps import NativeAgentDeps
from researcher.llm.client import LLMClient
from researcher.models import (
    AgentResult,
    AgentState,
    LLMTier,
    Task,
    TaskKind,
)
from researcher.storage.store import KnowledgeStore


class _EnrichResponse(BaseModel):
    related: list[str] = []


class EnrichAgent(Agent):
    """LLM-driven relation expansion — spawns DISCOVER tasks for related entities."""

    kind = "enrich"

    def __init__(
        self,
        agent_id: str,
        llm: LLMClient,
        store: KnowledgeStore,
        emit: EventEmitter,
        run_id: str,
        deps: NativeAgentDeps,
        entity_schema: dict,
        goal: str,
    ) -> None:
        super().__init__(
            agent_id=agent_id, llm=llm, store=store, emit=emit, run_id=run_id
        )
        self._deps = deps
        self._entity_schema = entity_schema
        self._goal = goal

    async def run(self, task: Task) -> AgentResult:
        start = time.monotonic()
        await self.set_state(AgentState.PLANNING)

        if not task.target_entity_id:
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error="entity_not_found",
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        entity = await self.store.get_entity(task.target_entity_id)
        if entity is None:
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error="entity_not_found",
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        name_cell = entity.fields.get("name")
        entity_name = name_cell.value if name_cell is not None else task.target_entity_id
        entity_type = self._entity_schema.get("entity_type", "Entity")

        known_fields = {
            k: v.value
            for k, v in entity.fields.items()
            if v.value is not None
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a relation-following research agent. Given an "
                    "entity, list related entities worth researching next."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Goal: {self._goal}\n"
                    f"Entity: {entity_name} ({entity_type})\n"
                    f"Known fields: {known_fields}\n\n"
                    f"Return JSON with `related` as a list of related "
                    f"{entity_type} entity names worth researching next."
                ),
            },
        ]

        await self.set_state(AgentState.EXTRACTING)
        try:
            response = await self.llm.complete_structured(
                messages=messages,
                schema=_EnrichResponse,
                tier=LLMTier.SMART,
                task_id=task.id,
            )
        except Exception as e:
            await self.log("error", f"llm enrich failed: {e}")
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error=f"llm_failed: {e}",
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        deadline = task.deadline_ts or (
            datetime.now(timezone.utc) + timedelta(minutes=5)
        )
        spawned: list[Task] = []
        for name in response.related:
            name = (name or "").strip()
            if not name:
                continue
            spawned.append(
                Task(
                    kind=TaskKind.DISCOVER,
                    spec_ref=task.spec_ref,
                    seed_query=name,
                    parent_task_id=task.id,
                    depth=task.depth + 1,
                    budget_usd=task.budget_usd,
                    deadline_ts=deadline,
                )
            )

        await self.set_state(AgentState.DONE)
        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            state=AgentState.DONE,
            claims=[],
            spawned_tasks=spawned,
            wall_ms=int((time.monotonic() - start) * 1000),
        )
