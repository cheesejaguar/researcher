"""Schema-guided extraction via LLMClient.complete_structured.

Given a text chunk, an entity name, and a target Pydantic schema,
asks the LLM to return a structured object matching the schema.
Used by the native ExpandAgent (Wave 1-D) to fill fields for a known entity.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from researcher.llm.client import LLMClient
from researcher.models import LLMTier

T = TypeVar("T", bound=BaseModel)


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
