"""Tests for DiskCache — TTL, eviction, persistence, file locking."""

import json
import time
from pathlib import Path

import pytest

from researcher.backends.disk_cache import DiskCache
from researcher.backends.models import CliResult, Extraction, SubagentResponse


def _sample_result(entity_name: str = "WWII", tokens: int = 100) -> CliResult:
    return CliResult(
        ok=True,
        data=SubagentResponse(
            entity_name=entity_name,
            extractions=[
                Extraction(
                    field="start_year",
                    value=1939,
                    source_url="https://example.com",
                    snippet="began in 1939",
                    confidence=0.95,
                )
            ],
            diagnostics="",
        ),
        wall_ms=500,
        exit_code=0,
        raw_usage={"input_tokens": tokens, "output_tokens": tokens // 2},
    )


def test_roundtrip_put_get(tmp_path: Path):
    cache = DiskCache(path=tmp_path / "cache.json")
    result = _sample_result()
    cache.put("k1", result)
    got = cache.get("k1")
    assert got is not None
    assert got.ok
    assert got.data is not None
    assert got.data.entity_name == "WWII"
    assert got.data.extractions[0].value == 1939


def test_get_missing_returns_none(tmp_path: Path):
    cache = DiskCache(path=tmp_path / "cache.json")
    assert cache.get("nonexistent") is None


def test_persistence_across_instances(tmp_path: Path):
    path = tmp_path / "cache.json"
    cache1 = DiskCache(path=path)
    cache1.put("persistent", _sample_result("PersistedWar"))
    del cache1  # no explicit close needed

    cache2 = DiskCache(path=path)
    got = cache2.get("persistent")
    assert got is not None
    assert got.data is not None
    assert got.data.entity_name == "PersistedWar"


def test_ttl_expiry(tmp_path: Path):
    cache = DiskCache(path=tmp_path / "cache.json", default_ttl_s=0.05)
    cache.put("short", _sample_result())
    # Immediately available.
    assert cache.get("short") is not None
    time.sleep(0.1)
    # Expired.
    assert cache.get("short") is None


def test_ttl_override_per_put(tmp_path: Path):
    cache = DiskCache(path=tmp_path / "cache.json", default_ttl_s=3600)
    cache.put("short", _sample_result(), ttl_s=0.05)
    time.sleep(0.1)
    assert cache.get("short") is None


def test_max_entries_eviction(tmp_path: Path):
    cache = DiskCache(path=tmp_path / "cache.json", max_entries=3)
    cache.put("a", _sample_result("A"))
    time.sleep(0.01)  # ensure distinct cached_at timestamps
    cache.put("b", _sample_result("B"))
    time.sleep(0.01)
    cache.put("c", _sample_result("C"))
    assert cache.size() == 3
    time.sleep(0.01)
    cache.put("d", _sample_result("D"))
    assert cache.size() == 3
    # Oldest entry ("a") should have been evicted.
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None
    assert cache.get("d") is not None


def test_prune_removes_expired(tmp_path: Path):
    cache = DiskCache(path=tmp_path / "cache.json", default_ttl_s=0.05)
    cache.put("x", _sample_result())
    cache.put("y", _sample_result(), ttl_s=3600)
    time.sleep(0.1)
    pruned = cache.prune()
    assert pruned == 1
    assert cache.get("x") is None
    assert cache.get("y") is not None


def test_stats_track_hits_and_misses(tmp_path: Path):
    cache = DiskCache(path=tmp_path / "cache.json")
    cache.put("k", _sample_result())
    cache.get("k")  # hit
    cache.get("k")  # hit
    cache.get("missing")  # miss
    stats = cache.stats
    assert stats["hits"] == 2
    assert stats["misses"] == 1


def test_clear_empties_cache(tmp_path: Path):
    cache = DiskCache(path=tmp_path / "cache.json")
    cache.put("k", _sample_result())
    assert cache.size() == 1
    cache.clear()
    assert cache.size() == 0
    assert cache.get("k") is None


def test_corrupted_file_starts_fresh(tmp_path: Path):
    path = tmp_path / "cache.json"
    path.write_text("{not valid json at all")
    # Should not raise; should start with an empty cache.
    cache = DiskCache(path=path)
    assert cache.size() == 0
    # Should be writable afterward.
    cache.put("k", _sample_result())
    assert cache.get("k") is not None


def test_missing_parent_directory_is_created(tmp_path: Path):
    nested = tmp_path / "nested" / "deeper"
    path = nested / "cache.json"
    cache = DiskCache(path=path)
    cache.put("k", _sample_result())
    assert path.exists()
