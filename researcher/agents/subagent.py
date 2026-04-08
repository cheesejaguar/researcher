"""SubagentResearcher — Agent subclass that delegates research to a local CLI."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from uuid import uuid4

from researcher.agents.base import Agent, EventEmitter
from researcher.backends.cli_runner import CliRunner
from researcher.backends.models import CliKind, SubagentResponse
from researcher.backends.prompts import SYSTEM_PROMPT, build_user_prompt
from researcher.budget import Budget
from researcher.events import (
    AgentLog,
    AgentLogPayload,
    SubagentCall,
    SubagentCallPayload,
)
from researcher.llm.client import LLMClient
from researcher.models import (
    AgentResult,
    AgentState,
    FactClaim,
    Provenance,
    Task,
)
from researcher.storage.store import KnowledgeStore


class SubagentResearcher(Agent):
    """Delegates entire research tasks to a local CLI subagent (claude / codex)."""

    kind = "subagent"

    def __init__(
        self,
        agent_id: str,
        llm: LLMClient,
        store: KnowledgeStore,
        emit: EventEmitter,
        run_id: str,
        runner: CliRunner,
        cli_kind: CliKind,
        entity_schema: dict,
        budget: Budget,
        goal: str,
        max_entities: int = 20,
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__(agent_id=agent_id, llm=llm, store=store, emit=emit, run_id=run_id)
        self._runner = runner
        self._cli_kind = cli_kind
        self._entity_schema = entity_schema
        self._budget = budget
        self._goal = goal
        self._max_entities = max_entities
        self._timeout_s = timeout_s

    async def run(self, task: Task) -> AgentResult:
        await self.set_state(AgentState.PLANNING)

        # Budget gate: reject before dispatching the subprocess.
        if not self._budget.allows_subagent_call():
            await self._log_error("subagent cap reached before dispatch")
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error="subagent_cap",
            )

        prompt = self._build_prompt(task)
        schema = SubagentResponse.model_json_schema()

        await self.set_state(AgentState.FETCHING)
        start = time.monotonic()
        result = await self._runner.execute(
            prompt=prompt, schema=schema, timeout_s=self._timeout_s
        )
        wall_ms = int((time.monotonic() - start) * 1000)

        # Record the call in the budget regardless of success — the subprocess did run.
        self._budget.record_subagent_call()

        # Emit the subagent_call event before returning.
        claims_emitted = 0 if not result.ok or result.data is None else len(result.data.extractions)
        await self._emit(
            SubagentCall(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=SubagentCallPayload(
                    agent_id=self.agent_id,
                    task_id=task.id,
                    cli_kind=self._cli_kind.value,  # type: ignore[arg-type]
                    wall_ms=wall_ms,
                    exit_code=result.exit_code,
                    claims_emitted=claims_emitted,
                ),
            )
        )

        if not result.ok or result.data is None:
            await self._log_error(f"subagent failure: {result.error}")
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error=result.error,
                wall_ms=wall_ms,
            )

        # Happy path: map extractions -> FactClaims.
        response = result.data
        if not response.extractions:
            await self._log_info(
                f"subagent returned 0 extractions for entity={response.entity_name!r} (empty response)"
            )

        claims = self._build_claims(task, response)
        usage = result.raw_usage or {}
        tokens_in = int(usage.get("input_tokens", 0) or 0)
        tokens_out = int(usage.get("output_tokens", 0) or 0)

        await self.set_state(AgentState.DONE)
        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            state=AgentState.DONE,
            claims=claims,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,
            wall_ms=wall_ms,
        )

    def _build_prompt(self, task: Task) -> str:
        field_names = [f["name"] for f in self._entity_schema.get("fields", [])]
        if task.field_hints:
            field_names = task.field_hints
        task_description = task.seed_query or "Find all entities matching the goal."
        user_prompt = build_user_prompt(
            goal=self._goal,
            entity_type=self._entity_schema.get("entity_type", "Entity"),
            task_description=task_description,
            field_list=field_names,
            max_entities=self._max_entities,
            schema_json=json.dumps(
                SubagentResponse.model_json_schema(), separators=(",", ":")
            ),
        )
        return f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    def _build_claims(self, task: Task, response: SubagentResponse) -> list[FactClaim]:
        now = datetime.now(UTC)
        entity_type = self._entity_schema.get("entity_type", "Entity")
        extractor_model = f"{self._cli_kind.value}/unknown"
        claims: list[FactClaim] = []
        for i, ex in enumerate(response.extractions):
            prov = Provenance(
                url=ex.source_url,
                fetched_at=now,
                snippet=ex.snippet,
                extractor_model=extractor_model,
                agent_id=self.agent_id,
                task_id=task.id,
                span_id=f"cli_{i}",
            )
            claims.append(
                FactClaim(
                    claim_id=uuid4().hex,
                    entity_type=entity_type,
                    entity_name=response.entity_name,
                    field=ex.field,
                    value=ex.value,
                    confidence=ex.confidence,
                    provenance=prov,
                    emitted_by=self.agent_id,
                    task_id=task.id,
                )
            )
        return claims

    async def _log_info(self, msg: str) -> None:
        await self._emit(
            AgentLog(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=AgentLogPayload(agent_id=self.agent_id, level="info", msg=msg),
            )
        )

    async def _log_error(self, msg: str) -> None:
        await self._emit(
            AgentLog(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=AgentLogPayload(agent_id=self.agent_id, level="error", msg=msg),
            )
        )
