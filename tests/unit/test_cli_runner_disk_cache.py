"""Tests that ClaudeCodeRunner and CodexRunner use the cross-run DiskCache."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from researcher.backends.cli_runner import ClaudeCodeRunner, CodexRunner
from researcher.backends.models import SubagentResponse


SCHEMA = SubagentResponse.model_json_schema()


def _fake_process(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.close = MagicMock()
    proc.stdin.drain = AsyncMock()
    return proc


def _claude_envelope(result_json: str, tokens: int = 100) -> bytes:
    return json.dumps(
        {
            "type": "result",
            "subtype": "final_result",
            "result": result_json,
            "usage": {"input_tokens": tokens, "output_tokens": tokens // 2},
            "total_cost_usd": 0.0,
            "is_error": False,
        }
    ).encode()


VALID_INNER = json.dumps({"entity_name": "WWII", "extractions": [], "diagnostics": ""})


@pytest.mark.asyncio
async def test_claude_runner_persists_successful_result_to_disk(tmp_path: Path):
    """First call executes the subprocess; a second ClaudeCodeRunner with the same
    cache_dir serves the same prompt from disk without spawning anything."""
    cache_dir = tmp_path / "cache"
    call_count = {"n": 0}

    async def fake_exec(*argv, **kwargs):
        call_count["n"] += 1
        return _fake_process(stdout=_claude_envelope(VALID_INNER))

    # First runner: call once, populate disk cache.
    runner1 = ClaudeCodeRunner(model="sonnet", cache_dir=cache_dir)
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        r1 = await runner1.execute(prompt="hello", schema=SCHEMA, timeout_s=30)
    assert r1.ok
    assert call_count["n"] == 1
    assert (cache_dir / "claude_cache.json").exists()

    # Second runner: same cache_dir, same prompt. Should NOT spawn.
    runner2 = ClaudeCodeRunner(model="sonnet", cache_dir=cache_dir)
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        r2 = await runner2.execute(prompt="hello", schema=SCHEMA, timeout_s=30)
    assert r2.ok
    assert call_count["n"] == 1, "second runner should have hit the disk cache, not spawned"


@pytest.mark.asyncio
async def test_claude_runner_without_cache_dir_does_not_touch_disk(tmp_path: Path):
    call_count = {"n": 0}

    async def fake_exec(*argv, **kwargs):
        call_count["n"] += 1
        return _fake_process(stdout=_claude_envelope(VALID_INNER))

    runner = ClaudeCodeRunner(model="sonnet")  # no cache_dir
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await runner.execute(prompt="hello", schema=SCHEMA, timeout_s=30)
    # No cache file anywhere near tmp_path.
    assert not any(tmp_path.rglob("*.json"))


@pytest.mark.asyncio
async def test_claude_runner_does_not_cache_failures_to_disk(tmp_path: Path):
    """Same discipline as the in-memory cache: failures are not persisted."""
    cache_dir = tmp_path / "cache"

    async def fake_exec(*argv, **kwargs):
        return _fake_process(stdout=b"not json")  # forces parse_failed after retry

    runner = ClaudeCodeRunner(model="sonnet", cache_dir=cache_dir)
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert not result.ok
    # Disk cache file may exist (load path creates parent), but it should contain no entries.
    cache_file = cache_dir / "claude_cache.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        assert data.get("entries", {}) == {}


@pytest.mark.asyncio
async def test_codex_runner_persists_successful_result_to_disk(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    last_message_file = tmp_path / "last.txt"
    last_message_file.write_text(VALID_INNER)

    call_count = {"n": 0}

    async def fake_exec(*argv, **kwargs):
        call_count["n"] += 1
        return _fake_process()

    runner1 = CodexRunner(
        model="o4-mini",
        last_message_file=last_message_file,
        schema_file_dir=tmp_path,
        cache_dir=cache_dir,
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        r1 = await runner1.execute(prompt="hi", schema=SCHEMA, timeout_s=30)
    assert r1.ok
    assert call_count["n"] == 1
    assert (cache_dir / "codex_cache.json").exists()

    runner2 = CodexRunner(
        model="o4-mini",
        last_message_file=last_message_file,
        schema_file_dir=tmp_path,
        cache_dir=cache_dir,
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        r2 = await runner2.execute(prompt="hi", schema=SCHEMA, timeout_s=30)
    assert r2.ok
    assert call_count["n"] == 1, "second runner should have hit the disk cache"
