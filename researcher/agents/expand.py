"""ExpandAgent — fill fields for a known entity.

Wave 1-D native agent. Given a Task with a target_entity_id and optional
field_hints, looks up the entity in the store, searches for sources, extracts
readable text, asks the LLM to fill in a dynamic schema, and emits one
FactClaim per populated field.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, create_model

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

_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


def _resolve_field_type(type_str: str) -> type:
    type_str = type_str.strip()
    if type_str.startswith("list[") and type_str.endswith("]"):
        inner = type_str[5:-1].strip()
        return list[_TYPE_MAP.get(inner, str)]  # type: ignore[valid-type]
    return _TYPE_MAP.get(type_str, str)


def _build_response_model(
    entity_type: str, field_specs: list[dict], allowed: list[str] | None
) -> type[BaseModel]:
    """Build a Pydantic model where every field is Optional[T] with default None.

    Honors the optional ``enum`` key on each field spec: when present, the
    field is narrowed to a ``Literal[*enum]`` (or a list of literals for
    ``list[str]`` fields) so the LLM is constrained to a closed vocabulary.
    """
    fields: dict[str, Any] = {}
    for f in field_specs:
        name = f.get("name")
        if name == "name":
            continue  # name is the identity; never re-extracted here
        if allowed is not None and name not in allowed:
            continue
        if not name:
            continue
        enum_values = f.get("enum") or []
        if enum_values:
            literal_type = Literal[tuple(enum_values)]  # type: ignore[valid-type]
            if f.get("type", "str").strip() == "list[str]":
                py_type = list[literal_type]  # type: ignore[valid-type]
            else:
                py_type = literal_type  # type: ignore[assignment]
        else:
            py_type = _resolve_field_type(f.get("type", "str"))
        fields[name] = (Optional[py_type], Field(default=None))
    if not fields:
        fields["_placeholder"] = (Optional[str], Field(default=None))
    return create_model(f"{entity_type}ExpandResponse", **fields)  # type: ignore[call-overload]


class ExpandAgent(Agent):
    """Fill fields on a known entity using web search + LLM extraction."""

    kind = "expand"

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
            await self.log("error", f"entity not found: {task.target_entity_id}")
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

        hint_text = " ".join(task.field_hints) if task.field_hints else ""
        query = f"{entity_name} {hint_text}".strip()

        search_results = await self._deps.search.search(query, max_results=10)

        await self.set_state(AgentState.FETCHING)
        fetched: list[tuple[str, str]] = []
        for result in search_results[: self._deps.max_fetch_per_task]:
            fetch = await self._deps.http.fetch(result.url)
            if not fetch.ok or (not fetch.content and fetch.raw_bytes is None):
                continue
            text = (
                extract_readable(
                    fetch.content,
                    url=result.url,
                    raw_bytes=fetch.raw_bytes,
                    content_type=fetch.content_type,
                )
                or fetch.content
            )
            chunks = chunk_text(text)
            if not chunks:
                continue
            top = chunks[0].text
            if len(chunks) > 1:
                top = f"{top}\n\n{chunks[1].text}"
            fetched.append((fetch.url, top))

        await self.set_state(AgentState.EXTRACTING)

        allowed_fields = list(task.field_hints) if task.field_hints else None
        response_model = _build_response_model(
            entity_type=entity_type,
            field_specs=self._entity_schema.get("fields", []),
            allowed=allowed_fields,
        )

        context = (
            "\n\n---\n\n".join(f"SOURCE: {url}\n{body}" for url, body in fetched) or "(no sources)"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a fact-extraction agent. Given an entity and source "
                    "text, return a JSON object populating as many of the schema "
                    "fields as the sources support. Leave unknown fields null."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Goal: {self._goal}\n"
                    f"Entity: {entity_name} ({entity_type})\n"
                    f"Fields to fill: {', '.join(allowed_fields) if allowed_fields else 'all'}\n\n"
                    f"Sources:\n{context}"
                ),
            },
        ]

        try:
            response = await self.llm.complete_structured(
                messages=messages,
                schema=response_model,
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
        now = datetime.now(UTC)
        primary_url = fetched[0][0] if fetched else "native://no-source"
        primary_snippet = fetched[0][1][:500] if fetched else ""

        payload = response.model_dump() if isinstance(response, BaseModel) else dict(response)

        # Optional skill-card-driven field validation. A field is rejected only
        # if at least one matching skill card declares it in `output_schema` and
        # the value fails validation. Fields not declared by any card always
        # pass — the schema is opt-in guardrail, not exhaustive coverage.
        skill_cards = []
        if self._deps.skill_registry is not None:
            skill_cards = self._deps.skill_registry.cards_for_entity_type(entity_type)

        for field_name, value in payload.items():
            if field_name == "_placeholder":
                continue
            if value is None:
                continue
            if isinstance(value, list) and len(value) == 0:
                continue
            if skill_cards:
                rejected_reason: str | None = None
                for card in skill_cards:
                    if not card.output_schema:
                        continue
                    if field_name not in card.output_schema:
                        continue
                    ok, reason = card.validate_output_field(field_name, value)
                    if not ok:
                        rejected_reason = reason
                        break
                if rejected_reason is not None:
                    await self.log(
                        "warn",
                        f"skill schema rejected field {field_name}={value!r}: {rejected_reason}",
                    )
                    continue
            prov = Provenance(
                url=primary_url,
                fetched_at=now,
                snippet=primary_snippet,
                extractor_model="native/expand",
                agent_id=self.agent_id,
                task_id=task.id,
                span_id=f"expand_{uuid4().hex[:8]}",
            )
            claims.append(
                FactClaim(
                    entity_type=entity_type,
                    entity_name=str(entity_name),
                    field=field_name,
                    value=value,
                    confidence=0.8,
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
