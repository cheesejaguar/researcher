"""Tests for extract_structured — offline, with a mock LLMClient."""

from __future__ import annotations

from pydantic import BaseModel

from researcher.extract.schema_extract import extract_structured
from researcher.llm.client import LLMClient
from researcher.models import LLMTier


class Target(BaseModel):
    name: str
    year: int | None = None


class StubLLM(LLMClient):
    def __init__(self, payload: BaseModel) -> None:
        self._payload = payload
        self.last_messages: list[dict] | None = None
        self.last_schema: type[BaseModel] | None = None
        self.last_tier: LLMTier | None = None
        self.last_task_id: str | None = None

    async def complete(self, messages, tier, task_id, temperature=0.2, max_tokens=None):  # type: ignore[override]
        raise AssertionError("complete() must not be used by extract_structured")

    async def complete_structured(  # type: ignore[override]
        self,
        messages,
        schema,
        tier,
        task_id,
        temperature=0.0,
        schema_retry=True,
    ):
        self.last_messages = messages
        self.last_schema = schema
        self.last_tier = tier
        self.last_task_id = task_id
        return self._payload

    async def embed(self, texts):  # type: ignore[override]
        return [[0.0] for _ in texts]

    def content_hash(self, messages, tier):  # type: ignore[override]
        return "stub"


async def test_extract_structured_returns_payload_and_forwards_schema() -> None:
    llm = StubLLM(Target(name="Alice", year=1999))
    out = await extract_structured(
        llm=llm,
        tier=LLMTier.FAST,
        schema=Target,
        text="Alice was born in 1999. She is notable.",
        entity_name="Alice",
        task_id="task-1",
    )
    assert isinstance(out, Target)
    assert out.name == "Alice"
    assert out.year == 1999
    assert llm.last_schema is Target
    assert llm.last_tier == LLMTier.FAST
    assert llm.last_task_id == "task-1"


async def test_extract_structured_builds_system_and_user_messages() -> None:
    llm = StubLLM(Target(name="Bob"))
    await extract_structured(
        llm=llm,
        tier=LLMTier.SMART,
        schema=Target,
        text="Bob is a person.",
        entity_name="Bob",
        task_id="task-2",
        instructions="Focus on biographical facts only.",
    )
    assert llm.last_messages is not None
    assert len(llm.last_messages) == 2
    assert llm.last_messages[0]["role"] == "system"
    assert "fact-extraction" in llm.last_messages[0]["content"].lower()
    assert llm.last_messages[1]["role"] == "user"
    user = llm.last_messages[1]["content"]
    assert "Entity: Bob" in user
    assert "Bob is a person." in user
    assert "Focus on biographical facts only." in user
