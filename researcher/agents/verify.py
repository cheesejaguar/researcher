"""VerifyAgent — resolve a conflict between candidate field values.

Wave 1-D native agent. Given a Task with target_entity_id and
field_hints=[field_name], looks up the matching open conflict, asks the LLM to
pick the winning value, emits a FactClaim with that value and confidence 0.9,
and marks the conflict resolved in the store.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from researcher.agents.base import Agent, EventEmitter
from researcher.agents.native_deps import NativeAgentDeps
from researcher.llm.client import LLMClient
from researcher.models import (
    AgentResult,
    AgentState,
    FactClaim,
    LLMTier,
    Provenance,
    Task,
)
from researcher.storage.store import Conflict, KnowledgeStore


class _VerifyResponse(BaseModel):
    winning_value: Any = None
    reason: str = ""


class VerifyAgent(Agent):
    """LLM-driven conflict resolver for ambiguous field values."""

    kind = "verify"

    def __init__(
        self,
        agent_id: str,
        llm: LLMClient,
        store: KnowledgeStore,
        emit: EventEmitter,
        run_id: str,
        deps: NativeAgentDeps,
        entity_schema: dict,
    ) -> None:
        super().__init__(
            agent_id=agent_id, llm=llm, store=store, emit=emit, run_id=run_id
        )
        self._deps = deps
        self._entity_schema = entity_schema

    async def run(self, task: Task) -> AgentResult:
        start = time.monotonic()
        await self.set_state(AgentState.PLANNING)

        field_name = task.field_hints[0] if task.field_hints else None
        if not task.target_entity_id or not field_name:
            await self.set_state(AgentState.DONE)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.DONE,
                claims=[],
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        open_conflicts = await self.store.get_conflicts(status="open")
        match: Conflict | None = None
        for c in open_conflicts:
            if c.entity_id == task.target_entity_id and c.field == field_name:
                match = c
                break

        if match is None:
            await self.log(
                "info",
                f"no open conflict for entity={task.target_entity_id} field={field_name}",
            )
            await self.set_state(AgentState.DONE)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.DONE,
                claims=[],
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        entity = await self.store.get_entity(task.target_entity_id)
        entity_type = self._entity_schema.get("entity_type", "Entity")
        if entity is not None:
            name_cell = entity.fields.get("name")
            entity_name = name_cell.value if name_cell is not None else task.target_entity_id
        else:
            entity_name = task.target_entity_id

        candidates_text = "\n".join(
            f"- value={cell.value!r} (confidence={cell.confidence:.2f})"
            for cell in match.candidates
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a conflict resolver. Given two or more candidate "
                    "values for the same field, pick the most likely correct "
                    "one and return it as JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Entity: {entity_name} ({entity_type})\n"
                    f"Field: {field_name}\n"
                    f"Candidates:\n{candidates_text}\n\n"
                    f"Return JSON with winning_value and a short reason."
                ),
            },
        ]

        await self.set_state(AgentState.EXTRACTING)
        try:
            response = await self.llm.complete_structured(
                messages=messages,
                schema=_VerifyResponse,
                tier=LLMTier.SMART,
                task_id=task.id,
            )
        except Exception as e:
            await self.log("error", f"llm verify failed: {e}")
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error=f"llm_failed: {e}",
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        # Update the conflict in the store to resolved.
        resolved = Conflict(
            conflict_id=match.conflict_id,
            entity_id=match.entity_id,
            field=match.field,
            candidates=list(match.candidates),
            status="resolved",
            winning_value=response.winning_value,
            reason=response.reason,
        )
        await self.store.record_conflict(resolved)

        # Build the claim — provenance points at the store itself since the
        # value was chosen, not freshly fetched.
        now = datetime.now(UTC)
        prov = Provenance(
            url=f"native://verify/{match.conflict_id}",
            fetched_at=now,
            snippet=response.reason or "",
            extractor_model="native/verify",
            agent_id=self.agent_id,
            task_id=task.id,
            span_id=f"verify_{uuid4().hex[:8]}",
        )
        claim = FactClaim(
            entity_type=entity_type,
            entity_name=str(entity_name),
            field=field_name,
            value=response.winning_value,
            confidence=0.9,
            provenance=prov,
            emitted_by=self.agent_id,
            task_id=task.id,
        )

        await self.set_state(AgentState.DONE)
        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            state=AgentState.DONE,
            claims=[claim],
            wall_ms=int((time.monotonic() - start) * 1000),
        )
