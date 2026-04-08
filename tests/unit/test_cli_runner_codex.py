"""Tests for CodexRunner — argv construction, output parse, error matrix."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from researcher.backends.cli_runner import CodexRunner
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


@pytest.mark.asyncio
async def test_argv_construction_includes_required_flags(tmp_path: Path):
    last_message_file = tmp_path / "last.txt"
    last_message_file.write_text(
        json.dumps({"entity_name": "X", "extractions": [], "diagnostics": ""})
    )

    captured_argv: list = []

    async def fake_exec(*argv, **kwargs):
        captured_argv.extend(argv)
        return _fake_process()

    runner = CodexRunner(
        model="o4-mini",
        last_message_file=last_message_file,
        schema_file_dir=tmp_path,
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert result.ok, f"unexpected failure: {result.error}"
    assert captured_argv[0].endswith("codex")
    assert "exec" in captured_argv
    assert "--json" in captured_argv
    assert "--output-schema" in captured_argv
    assert "--output-last-message" in captured_argv
    assert "--sandbox" in captured_argv
    assert "read-only" in captured_argv
    assert "--skip-git-repo-check" in captured_argv
    assert "-m" in captured_argv
    assert "o4-mini" in captured_argv


@pytest.mark.asyncio
async def test_schema_file_is_written_and_cleaned_up(tmp_path: Path):
    last_message_file = tmp_path / "last.txt"
    last_message_file.write_text(
        json.dumps({"entity_name": "X", "extractions": [], "diagnostics": ""})
    )

    schema_path_captured: list[str] = []

    async def fake_exec(*argv, **kwargs):
        for i, a in enumerate(argv):
            if a == "--output-schema":
                schema_path_captured.append(argv[i + 1])
                break
        return _fake_process()

    runner = CodexRunner(
        model="o4-mini",
        last_message_file=last_message_file,
        schema_file_dir=tmp_path,
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert len(schema_path_captured) == 1
    # After the call, the temp schema file should be removed.
    assert not Path(schema_path_captured[0]).exists()


@pytest.mark.asyncio
async def test_parses_last_message_file_into_subagent_response(tmp_path: Path):
    last_message_file = tmp_path / "last.txt"
    last_message_file.write_text(
        json.dumps(
            {
                "entity_name": "WWII",
                "extractions": [
                    {
                        "field": "start_year",
                        "value": 1939,
                        "source_url": "https://example.com/wwii",
                        "snippet": "began in 1939",
                        "confidence": 0.9,
                    }
                ],
                "diagnostics": "ok",
            }
        )
    )

    async def fake_exec(*argv, **kwargs):
        return _fake_process()

    runner = CodexRunner(
        model="o4-mini",
        last_message_file=last_message_file,
        schema_file_dir=tmp_path,
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert result.ok
    assert result.data is not None
    assert result.data.entity_name == "WWII"
    assert result.data.extractions[0].value == 1939


@pytest.mark.asyncio
async def test_non_zero_exit_captures_stderr_tail(tmp_path: Path):
    last_message_file = tmp_path / "last.txt"

    async def fake_exec(*argv, **kwargs):
        return _fake_process(stderr=b"Error: config unreadable\n", returncode=3)

    runner = CodexRunner(
        model="o4-mini",
        last_message_file=last_message_file,
        schema_file_dir=tmp_path,
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert not result.ok
    assert "exit 3" in (result.error or "")


@pytest.mark.asyncio
async def test_usage_limit_pattern_match(tmp_path: Path):
    last_message_file = tmp_path / "last.txt"

    async def fake_exec(*argv, **kwargs):
        return _fake_process(stderr=b"quota exceeded for the month\n", returncode=1)

    runner = CodexRunner(
        model="o4-mini",
        last_message_file=last_message_file,
        schema_file_dir=tmp_path,
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert not result.ok
    assert result.error == "usage_limit_reached"
