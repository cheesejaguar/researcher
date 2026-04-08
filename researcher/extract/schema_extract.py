"""Schema-guided extraction via LLMClient.complete_structured.

Given a text chunk, an entity name, and a target Pydantic schema,
asks the LLM to return a structured object matching the schema.
Used by the native ExpandAgent (Wave 1-D) to fill fields for a known entity.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from researcher.extract.chunker import Chunk
from researcher.llm.client import LLMClient
from researcher.models import LLMTier

T = TypeVar("T", bound=BaseModel)


def build_provenance_template(
    chunk: Chunk | None = None,
    *,
    url: str,
    agent_id: str,
    task_id: str,
    extractor_model: str,
    snippet: str = "",
    span_id: str = "",
) -> dict:
    """Construct a dict suitable for ``Provenance(**template)`` with chunk span metadata.

    The caller fills in ``fetched_at`` and any per-claim ``span_id``; this helper
    just propagates the chunk's byte offsets and id into the right fields so
    agents with a Chunk in scope (Discover/Expand) can attach field-level
    passage-span provenance to their FactClaims (PROV-AGENT, arXiv 2508.02866).
    """
    template: dict = {
        "url": url,
        "agent_id": agent_id,
        "task_id": task_id,
        "extractor_model": extractor_model,
        "snippet": snippet,
        "span_id": span_id,
    }
    if chunk is not None:
        template["passage_start"] = chunk.start_char
        template["passage_end"] = chunk.end_char
        template["chunk_id"] = f"chunk_{chunk.index}"
    return template


async def extract_structured(
    llm: LLMClient,
    tier: LLMTier,
    schema: type[T],
    text: str,
    entity_name: str,
    task_id: str,
    instructions: str = "",
) -> T:
    """Extract a structured object from `text` matching `schema`."""
    system = (
        "You are a fact-extraction agent. Given a text passage and a target "
        "entity, return a JSON object matching the schema. If a field is not "
        "present in the text, omit it or use null."
    )
    user = (
        f"Entity: {entity_name}\n\n"
        f"Text:\n{text}\n\n"
        f"{instructions}".strip()
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return await llm.complete_structured(
        messages=messages,
        schema=schema,
        tier=tier,
        task_id=task_id,
    )
