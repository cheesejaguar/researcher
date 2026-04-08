"""VerifyAgent — resolve a conflict between candidate field values.

v1.2 Tier-2 upgrade: BoN-MAV (Best-of-N Multi-Aspect Verifier) per arXiv
2502.20379. Instead of a single LLM call, VerifyAgent.run now fans out three
parallel aspect verifiers (factual_consistency, source_quality,
entity_resolution) and majority-votes the winning value. Weak verifiers in
consensus outperform a single strong verifier by up to 20% on small models.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from researcher.agents.base import Agent, EventEmitter
from researcher.agents.native_deps import NativeAgentDeps
from researcher.llm.client import LLMClient
from researcher.llm.prompts import PromptVariant
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
    confidence: float = 0.9


def _vote_key(value: Any) -> str:
    """Hashable key for majority voting; JSON-stringifies unhashable values."""
    return json.dumps(value, default=str, sort_keys=True)


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

        candidates_text = "; ".join(
            f"value={cell.value!r} (confidence={cell.confidence:.2f})"
            for cell in match.candidates
        )

        # Resolve the aspect variants from the prompt registry.
        try:
            verify_set = self._deps.prompts.get_set("verify")
        except KeyError:
            await self.log("error", "no verify prompt set registered")
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error="no_verify_prompt_set",
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        variants: list[PromptVariant] = list(verify_set.variants[:3])
        if not variants:
            await self.log("error", "verify prompt set has no variants")
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error="no_verify_variants",
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        await self.set_state(AgentState.EXTRACTING)

        async def _run_one_aspect(variant: PromptVariant) -> _VerifyResponse:
            messages = [
                {"role": "system", "content": variant.system},
                {
                    "role": "user",
                    "content": variant.user_template.format(
                        entity_name=entity_name,
                        field=field_name,
                        candidates=candidates_text,
                    ),
                },
            ]
            return await self.llm.complete_structured(
                messages=messages,
                schema=_VerifyResponse,
                tier=LLMTier.SMART,
                task_id=task.id,
            )

        aspect_results = await asyncio.gather(
            *(_run_one_aspect(v) for v in variants),
            return_exceptions=True,
        )

        valid_results: list[_VerifyResponse] = []
        for idx, r in enumerate(aspect_results):
            if isinstance(r, Exception):
                await self.log(
                    "warn",
                    f"aspect verifier {variants[idx].name} failed: {r}",
                )
                continue
            valid_results.append(r)  # type: ignore[arg-type]

        if not valid_results:
            await self.log("error", "all aspect verifiers failed")
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error="all_verifiers_failed",
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        # Majority vote — stringify values so lists/dicts are hashable.
        key_counter: Counter[str] = Counter(
            _vote_key(r.winning_value) for r in valid_results
        )
        winning_key, winning_votes = key_counter.most_common(1)[0]
        winning_value = next(
            r.winning_value
            for r in valid_results
            if _vote_key(r.winning_value) == winning_key
        )
        avg_confidence = sum(r.confidence for r in valid_results) / len(valid_results)
        reason = (
            f"BoN-MAV majority vote: {winning_votes}/{len(valid_results)} "
            f"aspects agree"
        )

        # Update the conflict in the store to resolved.
        resolved = Conflict(
            conflict_id=match.conflict_id,
            entity_id=match.entity_id,
            field=match.field,
            candidates=list(match.candidates),
            status="resolved",
            winning_value=winning_value,
            reason=reason,
        )
        await self.store.record_conflict(resolved)

        # Build the claim — provenance reflects the ensemble decision.
        now = datetime.now(UTC)
        prov = Provenance(
            url=f"native://verify/{match.conflict_id}",
            fetched_at=now,
            snippet=reason,
            extractor_model="native/verify-bon-mav",
            agent_id=self.agent_id,
            task_id=task.id,
            span_id=f"verify_{uuid4().hex[:8]}",
        )
        claim = FactClaim(
            entity_type=entity_type,
            entity_name=str(entity_name),
            field=field_name,
            value=winning_value,
            confidence=avg_confidence,
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
