"""Tests for OpenRouterClient — offline, with a mocked AsyncOpenAI class."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from researcher.llm.cache import LLMResponseCache
from researcher.llm.client import LLMResponse
from researcher.llm.cost_tracker import SimpleCostTracker
from researcher.llm.openrouter import OpenRouterClient
from researcher.models import LLMTier

# ---------- Fakes ----------

def _fake_openai_response(text: str, tokens_in: int = 11, tokens_out: int = 7) -> Any:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = tokens_in
    usage.completion_tokens = tokens_out
    obj = MagicMock()
    obj.choices = [choice]
    obj.usage = usage
    return obj


def _fake_openai_cls(responses: list[Any]) -> type:
    """Build a fake AsyncOpenAI class whose chat.completions.create returns the
    next item from `responses` on each call. Tracks call count on the class."""

    it = iter(responses)

    class FakeAsyncOpenAI:
        instances: list[FakeAsyncOpenAI] = []
        create_calls: list[dict] = []

        def __init__(self, api_key: str, base_url: str) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.chat = MagicMock()

            async def _create(**kwargs: Any) -> Any:
                type(self).create_calls.append(kwargs)
                return next(it)

            self.chat.completions = MagicMock()
            self.chat.completions.create = AsyncMock(side_effect=_create)
            type(self).instances.append(self)

    return FakeAsyncOpenAI


# ---------- content_hash ----------

def test_content_hash_deterministic():
    client_a = OpenRouterClient(api_key="sk-test")
    client_b = OpenRouterClient(api_key="sk-test")
    messages = [{"role": "user", "content": "hi"}]
    h1 = client_a.content_hash(messages, LLMTier.FAST)
    h2 = client_b.content_hash(messages, LLMTier.FAST)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_content_hash_differs_across_tiers():
    client = OpenRouterClient(api_key="sk-test")
    messages = [{"role": "user", "content": "hi"}]
    h_fast = client.content_hash(messages, LLMTier.FAST)
    h_smart = client.content_hash(messages, LLMTier.SMART)
    assert h_fast != h_smart


# ---------- complete ----------

@pytest.mark.asyncio
async def test_complete_returns_llm_response():
    FakeCls = _fake_openai_cls([_fake_openai_response("hello world", 12, 8)])
    client = OpenRouterClient(api_key="sk-test", async_openai_cls=FakeCls)
    resp = await client.complete(
        [{"role": "user", "content": "ping"}],
        LLMTier.FAST,
        task_id="t1",
    )
    assert isinstance(resp, LLMResponse)
    assert resp.text == "hello world"
    assert resp.usage.tokens_in == 12
    assert resp.usage.tokens_out == 8
    assert resp.usage.cache_hit is False
    assert resp.usage.cost_usd > 0
    assert FakeCls.create_calls and len(FakeCls.create_calls) == 1


@pytest.mark.asyncio
async def test_complete_tracks_usage_on_cost_tracker():
    FakeCls = _fake_openai_cls([_fake_openai_response("foo", 30, 10)])
    tracker = SimpleCostTracker()
    client = OpenRouterClient(
        api_key="sk-test",
        async_openai_cls=FakeCls,
        cost_tracker=tracker,
    )
    await client.complete(
        [{"role": "user", "content": "q"}],
        LLMTier.FAST,
        task_id="task-xyz",
    )
    by_task = tracker.by_task("task-xyz")
    assert by_task.tokens_in == 30
    assert by_task.tokens_out == 10
    assert tracker.total_usd() > 0


@pytest.mark.asyncio
async def test_complete_hits_cache_on_second_call(tmp_path: Path):
    FakeCls = _fake_openai_cls([_fake_openai_response("once", 5, 5)])
    cache = LLMResponseCache(tmp_path / "c.json")
    client = OpenRouterClient(
        api_key="sk-test",
        async_openai_cls=FakeCls,
        cache=cache,
    )
    messages = [{"role": "user", "content": "same"}]
    r1 = await client.complete(messages, LLMTier.FAST, task_id="t1")
    r2 = await client.complete(messages, LLMTier.FAST, task_id="t1")

    assert r1.text == r2.text == "once"
    assert r1.usage.cache_hit is False
    assert r2.usage.cache_hit is True
    # Only ONE real call to the underlying OpenAI client.
    assert len(FakeCls.create_calls) == 1


# ---------- complete_structured ----------

class _SamplePayload(BaseModel):
    name: str
    count: int


@pytest.mark.asyncio
async def test_complete_structured_roundtrips_pydantic():
    valid = json.dumps({"name": "widget", "count": 7})
    FakeCls = _fake_openai_cls([_fake_openai_response(valid)])
    client = OpenRouterClient(api_key="sk-test", async_openai_cls=FakeCls)
    out = await client.complete_structured(
        [{"role": "user", "content": "give me widget info"}],
        _SamplePayload,
        LLMTier.FAST,
        task_id="t1",
    )
    assert isinstance(out, _SamplePayload)
    assert out.name == "widget"
    assert out.count == 7


@pytest.mark.asyncio
async def test_complete_structured_retries_on_parse_error():
    # First response is garbage, second is valid JSON.
    garbage = _fake_openai_response("this is not json at all")
    valid = _fake_openai_response(json.dumps({"name": "x", "count": 1}))
    FakeCls = _fake_openai_cls([garbage, valid])
    client = OpenRouterClient(api_key="sk-test", async_openai_cls=FakeCls)
    out = await client.complete_structured(
        [{"role": "user", "content": "q"}],
        _SamplePayload,
        LLMTier.FAST,
        task_id="t1",
    )
    assert out.name == "x"
    assert out.count == 1
    assert len(FakeCls.create_calls) == 2


@pytest.mark.asyncio
async def test_complete_structured_raises_without_retry():
    garbage = _fake_openai_response("not json")
    FakeCls = _fake_openai_cls([garbage])
    client = OpenRouterClient(api_key="sk-test", async_openai_cls=FakeCls)
    with pytest.raises((json.JSONDecodeError, ValueError)):
        await client.complete_structured(
            [{"role": "user", "content": "q"}],
            _SamplePayload,
            LLMTier.FAST,
            task_id="t1",
            schema_retry=False,
        )
    assert len(FakeCls.create_calls) == 1


# ---------- prompt caching (cache_control) ----------


@pytest.mark.asyncio
async def test_complete_marks_first_system_message_as_cacheable():
    """The system message gets cache_control: {type: 'ephemeral'} so providers can cache it."""
    FakeCls = _fake_openai_cls([_fake_openai_response("ok", 100, 10)])
    client = OpenRouterClient(api_key="sk-test", async_openai_cls=FakeCls)
    messages = [
        {"role": "system", "content": "You are a research agent. Long shared context here."},
        {"role": "user", "content": "Find entities matching: wars"},
    ]
    await client.complete(messages, tier=LLMTier.FAST, task_id="t1")

    sent = FakeCls.create_calls[0]["messages"]
    assert len(sent) == 2
    # The system message should have cache_control attached.
    first = sent[0]
    assert "cache_control" in first or (
        isinstance(first.get("content"), list)
        and first["content"]
        and "cache_control" in first["content"][0]
    )


@pytest.mark.asyncio
async def test_prefix_caching_does_not_break_non_cacheable_messages():
    """User messages should NOT have cache_control attached."""
    FakeCls = _fake_openai_cls([_fake_openai_response("ok", 100, 10)])
    client = OpenRouterClient(api_key="sk-test", async_openai_cls=FakeCls)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user query"},
    ]
    await client.complete(messages, tier=LLMTier.FAST, task_id="t1")

    sent = FakeCls.create_calls[0]["messages"]
    user_msg = sent[1]
    # The user message's content shouldn't carry cache_control.
    if isinstance(user_msg.get("content"), list):
        for part in user_msg["content"]:
            assert "cache_control" not in part
    else:
        assert "cache_control" not in user_msg


@pytest.mark.asyncio
async def test_apply_prompt_caching_does_not_mutate_input():
    """The helper must not mutate the caller's messages list."""
    client = OpenRouterClient(api_key="sk-test")
    messages = [
        {"role": "system", "content": "shared"},
        {"role": "user", "content": "ask"},
    ]
    snapshot = [dict(m) for m in messages]
    out = client._apply_prompt_caching(messages)
    assert messages == snapshot
    assert out is not messages


@pytest.mark.asyncio
async def test_complete_records_cache_read_input_tokens_when_present():
    """If the response surfaces prompt_tokens_details.cached_tokens, record it."""
    resp = _fake_openai_response("ok", 100, 10)
    details = MagicMock()
    details.cached_tokens = 80
    resp.usage.prompt_tokens_details = details
    FakeCls = _fake_openai_cls([resp])
    client = OpenRouterClient(api_key="sk-test", async_openai_cls=FakeCls)
    out = await client.complete(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}],
        tier=LLMTier.FAST,
        task_id="t1",
    )
    assert out.usage.cache_read_input_tokens == 80


@pytest.mark.asyncio
async def test_complete_cache_read_input_tokens_defaults_to_zero():
    """If the response doesn't surface cached_tokens, default to 0."""
    resp = _fake_openai_response("ok", 100, 10)
    # Simulate an older OpenAI/OpenRouter response that omits prompt_tokens_details.
    resp.usage.prompt_tokens_details = None
    FakeCls = _fake_openai_cls([resp])
    client = OpenRouterClient(api_key="sk-test", async_openai_cls=FakeCls)
    out = await client.complete(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}],
        tier=LLMTier.FAST,
        task_id="t1",
    )
    assert out.usage.cache_read_input_tokens == 0
