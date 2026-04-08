"""DiscoverAgent — seed query -> candidate entity names.

Wave 1-D native agent. Given a Task with a seed_query, runs a web search,
fetches the top results, extracts readable text, asks the LLM for candidate
entity names, then emits one FactClaim per candidate populating the primary
entity's `name` field.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel

from researcher.agents.base import Agent, EventEmitter
from researcher.agents.native_deps import NativeAgentDeps
from researcher.extract.chunker import chunk_text
from researcher.fetch.readability import extract_readable
from researcher.llm.client import LLMClient
from researcher.models import (
    AgentResult,
    AgentState,
    FactClaim,
    LLMTier,
    Provenance,
    Task,
)
from researcher.storage.store import KnowledgeStore


class _DiscoverResponse(BaseModel):
    entities: list[str] = []


class DiscoverAgent(Agent):
    """Seed query -> search -> fetch -> LLM extraction of candidate entity names."""

    kind = "discover"

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

        query = task.seed_query or self._goal
        try:
            search_results = await self._deps.search.search(query, max_results=10)
        except Exception as e:  # pragma: no cover — defensive
            await self.log("error", f"search failed: {e}")
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error=f"search_failed: {e}",
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        await self.set_state(AgentState.FETCHING)
        fetched: list[tuple[str, str]] = []  # (url, text)
        for result in search_results[: self._deps.max_fetch_per_task]:
            fetch = await self._deps.http.fetch(result.url)
            if not fetch.ok or not fetch.content:
                await self.log("info", f"skip {result.url}: {fetch.error}")
                continue
            text = extract_readable(fetch.content, url=result.url) or fetch.content
            chunks = chunk_text(text)
            if not chunks:
                continue
            top = chunks[0].text
            if len(chunks) > 1:
                top = f"{top}\n\n{chunks[1].text}"
            fetched.append((fetch.url, top))

        await self.set_state(AgentState.EXTRACTING)
        if not fetched:
            await self.set_state(AgentState.DONE)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.DONE,
                claims=[],
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        entity_type = self._entity_schema.get("entity_type", "Entity")
        combined_context = "\n\n---\n\n".join(
            f"SOURCE: {url}\n{body}" for url, body in fetched
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research agent that discovers new entities "
                    "matching a given topic. Return structured JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Goal: {self._goal}\n"
                    f"Seed query: {query}\n"
                    f"Entity type: {entity_type}\n\n"
                    f"From the sources below, list the distinct {entity_type} "
                    f"entity names you find.\n\n{combined_context}"
                ),
            },
        ]

        try:
            response = await self.llm.complete_structured(
                messages=messages,
                schema=_DiscoverResponse,
                tier=LLMTier.FAST,
                task_id=task.id,
            )
        except Exception as e:
            await self.log("error", f"llm extraction failed: {e}")
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error=f"llm_failed: {e}",
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        claims: list[FactClaim] = []
        now = datetime.now(timezone.utc)
        primary_url, primary_snippet = fetched[0]
        for name in response.entities:
            name = (name or "").strip()
            if not name:
                continue
            prov = Provenance(
                url=primary_url,
                fetched_at=now,
                snippet=primary_snippet[:500],
                extractor_model="native/discover",
                agent_id=self.agent_id,
                task_id=task.id,
                span_id=f"discover_{uuid4().hex[:8]}",
            )
            claims.append(
                FactClaim(
                    entity_type=entity_type,
                    entity_name=name,
                    field="name",
                    value=name,
                    confidence=0.7,
                    provenance=prov,
                    emitted_by=self.agent_id,
                    task_id=task.id,
                )
            )

        await self.set_state(AgentState.DONE)
        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            state=AgentState.DONE,
            claims=claims,
            wall_ms=int((time.monotonic() - start) * 1000),
        )
