"""Tests for LLMResponseCache — JSON-file content-hash cache."""

from __future__ import annotations

import time
from pathlib import Path

from researcher.llm.cache import LLMResponseCache
from researcher.llm.client import LLMResponse, LLMUsage


def _response(text: str = "hello") -> LLMResponse:
    return LLMResponse(
        text=text,
        usage=LLMUsage(
            tokens_in=10,
            tokens_out=20,
            cost_usd=0.001,
            model="openrouter/hermes-3-8b",
            cache_hit=False,
        ),
    )


def test_put_then_get_roundtrip(tmp_path: Path):
    cache = LLMResponseCache(tmp_path / "c.json")
    resp = _response("hi")
    cache.put("k1", resp)
    fetched = cache.get("k1")
    assert fetched is not None
    assert fetched.text == "hi"
    assert fetched.usage.tokens_in == 10
    assert fetched.usage.tokens_out == 20


def test_get_returns_none_on_miss(tmp_path: Path):
    cache = LLMResponseCache(tmp_path / "c.json")
    assert cache.get("missing-key") is None
    assert cache.stats["misses"] == 1
    assert cache.stats["hits"] == 0


def test_get_flips_cache_hit_flag_to_true(tmp_path: Path):
    cache = LLMResponseCache(tmp_path / "c.json")
    resp = _response()
    assert resp.usage.cache_hit is False
    cache.put("k", resp)
    fetched = cache.get("k")
    assert fetched is not None
    assert fetched.usage.cache_hit is True


def test_ttl_expiry_evicts_entry(tmp_path: Path):
    cache = LLMResponseCache(tmp_path / "c.json", default_ttl_s=0.01)
    cache.put("k", _response())
    time.sleep(0.02)
    assert cache.get("k") is None
    assert cache.stats["misses"] == 1


def test_persistence_across_instances(tmp_path: Path):
    path = tmp_path / "c.json"
    cache_a = LLMResponseCache(path)
    cache_a.put("k", _response("persist-me"))
    # New instance reads the same file.
    cache_b = LLMResponseCache(path)
    fetched = cache_b.get("k")
    assert fetched is not None
    assert fetched.text == "persist-me"
    assert fetched.usage.cache_hit is True


def test_corrupt_cache_file_recovers_to_empty(tmp_path: Path):
    path = tmp_path / "c.json"
    path.write_text("not valid json {{{")
    cache = LLMResponseCache(path)
    # No exception, behaves as empty.
    assert cache.get("anything") is None
    cache.put("k", _response("ok"))
    assert cache.get("k") is not None
