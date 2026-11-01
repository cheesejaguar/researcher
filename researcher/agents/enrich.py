"""EnrichAgent — follow relations from a known entity.

Wave 1-D native agent. Given a Task with target_entity_id, asks the LLM to
suggest related entities worth researching next and emits new DISCOVER tasks
for each via AgentResult.spawned_tasks.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

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


class _RelationCandidate(BaseModel):
    relation: str
    target_name: str
    confidence: float = 0.7


class _EnrichResponse(BaseModel):
    related: list[str] = []
    relations: list[_RelationCandidate] = []


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
        super().__init__(agent_id=agent_id, llm=llm, store=store, emit=emit, run_id=run_id)
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

        known_fields = {k: v.value for k, v in entity.fields.items() if v.value is not None}
        declared = self._deps.relations
        relation_lines = "\n".join(
            f"- {r.name}: {r.source_type} -> {r.target_type}"
            + (f" ({r.description})" if r.description else "")
            for r in declared
        )
        relation_instruction = ""
        if relation_lines:
            relation_instruction = (
                "\nKnown relation types:\n"
                f"{relation_lines}\n"
                "For each relation you can support, add an item to `relations` "
                "with the relation label, the target entity name, and a "
                "confidence in [0, 1]. Only use the relation labels above."
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a relation-following research agent. Given an "
                    "entity, list related entities worth researching next and "
                    "extract typed relations when the schema defines them."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Goal: {self._goal}\n"
                    f"Entity: {entity_name} ({entity_type})\n"
                    f"Known fields: {known_fields}\n\n"
                    f"Return JSON with `related` as a list of related entity "
                    f"names worth researching next."
                    f"{relation_instruction}"
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

        deadline = task.deadline_ts or (datetime.now(UTC) + timedelta(minutes=5))
        spawned: list[Task] = []
        spawned_names: set[str] = set()

        def add_discover(name: str) -> None:
            normalized = (name or "").strip()
            if not normalized or normalized in spawned_names:
                return
            # Depth cap: relation-following must terminate even when the LLM
            # keeps proposing new neighbours.
            if task.depth >= self._deps.max_depth:
                return
            spawned_names.add(normalized)
            spawned.append(
                Task(
                    kind=TaskKind.DISCOVER,
                    spec_ref=task.spec_ref,
                    seed_query=normalized,
                    parent_task_id=task.id,
                    depth=task.depth + 1,
                    budget_usd=task.budget_usd,
                    deadline_ts=deadline,
                )
            )

        relation_by_label = {r.name: r for r in self._deps.relations}
        record_relation = getattr(self.store, "record_relation", None)
        for candidate in response.relations:
            relation_spec = relation_by_label.get(candidate.relation)
            target_name = (candidate.target_name or "").strip()
            if relation_spec is None or not target_name:
                continue
            target_id = await self._find_or_create_entity(relation_spec.target_type, target_name)
            if target_id is None:
                continue
            if callable(record_relation):
                try:
                    await record_relation(
                        source_id=task.target_entity_id,
                        target_id=target_id,
                        relation_label=relation_spec.name,
                        confidence=max(0.0, min(1.0, float(candidate.confidence))),
                        run_id=self._run_id,
                    )
                except Exception as exc:
                    await self.log("error", f"record_relation failed: {exc}")
            add_discover(target_name)

        for name in response.related:
            add_discover(name)

        await self.set_state(AgentState.DONE)
        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            state=AgentState.DONE,
            claims=[],
            spawned_tasks=spawned,
            wall_ms=int((time.monotonic() - start) * 1000),
        )

    async def _find_or_create_entity(self, entity_type: str, name: str) -> str | None:
        """Return the id for ``(entity_type, name)``, creating a stub row."""
        try:
            rows = await self.store.query(
                "SELECT id FROM entities WHERE entity_type = ? AND name = ?",
                (entity_type, name),
            )
            if rows:
                return str(rows[0]["id"])
        except Exception:
            pass
        try:
            return await self.store.upsert_entity(entity_type, name, {})
        except Exception as exc:
            await self.log("error", f"failed to create relation target {name!r}: {exc}")
            return None
