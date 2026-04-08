"""Tests for ClaudeCodeRunner — subprocess mocking + error matrix."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from researcher.backends.cli_runner import ClaudeCodeRunner
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


def _claude_envelope(result_json: str, input_tokens: int = 100, output_tokens: int = 50) -> bytes:
    return json.dumps(
        {
            "type": "result",
            "subtype": "final_result",
            "result": result_json,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "total_cost_usd": 0.0,
            "is_error": False,
        }
    ).encode()


@pytest.mark.asyncio
async def test_argv_construction_includes_required_flags():
    valid = _claude_envelope(
        json.dumps({"entity_name": "X", "extractions": [], "diagnostics": ""})
    )
    captured_argv: list = []

    async def fake_exec(*argv, **kwargs):
        captured_argv.extend(argv)
        return _fake_process(stdout=valid)

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="hi", schema=SCHEMA, timeout_s=30)

    assert result.ok
    assert "claude" in captured_argv[0]
    assert "-p" in captured_argv
    assert "--print" in captured_argv
    assert "--output-format" in captured_argv
    assert "json" in captured_argv
    assert "--json-schema" in captured_argv
    assert "--bare" in captured_argv
    assert "--model" in captured_argv
    assert "sonnet" in captured_argv
    assert "--allowedTools" in captured_argv
    joined = " ".join(captured_argv)
    assert "WebSearch" in joined
    assert "WebFetch" in joined


@pytest.mark.asyncio
async def test_prompt_is_written_to_stdin_not_argv():
    valid = _claude_envelope(
        json.dumps({"entity_name": "X", "extractions": [], "diagnostics": ""})
    )
    stdin_writes: list[bytes] = []
    proc = _fake_process(stdout=valid)
    proc.stdin.write = lambda data: stdin_writes.append(data)

    async def fake_exec(*argv, **kwargs):
        assert "SECRET_PROMPT_BODY" not in " ".join(argv)
        return proc

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await runner.execute(prompt="SECRET_PROMPT_BODY", schema=SCHEMA, timeout_s=30)

    joined_stdin = b"".join(stdin_writes)
    assert b"SECRET_PROMPT_BODY" in joined_stdin


@pytest.mark.asyncio
async def test_parses_envelope_into_subagent_response():
    inner = {
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
    envelope = _claude_envelope(json.dumps(inner))

    async def fake_exec(*argv, **kwargs):
        return _fake_process(stdout=envelope)

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert result.ok
    assert result.data is not None
    assert result.data.entity_name == "WWII"
    assert len(result.data.extractions) == 1
    assert result.data.extractions[0].value == 1939
    assert result.raw_usage == {"input_tokens": 100, "output_tokens": 50}
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_timeout_kills_child_and_returns_failure():
    proc = _fake_process()

    async def fake_communicate():
        raise asyncio.TimeoutError()

    proc.communicate = fake_communicate

    async def fake_exec(*argv, **kwargs):
        return proc

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=0.1)

    assert not result.ok
    assert result.error == "timeout"
    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_non_zero_exit_captures_stderr_tail():
    async def fake_exec(*argv, **kwargs):
        return _fake_process(stdout=b"", stderr=b"bad flag: --foo\n", returncode=2)

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert not result.ok
    assert result.error is not None
    assert "exit 2" in result.error
    assert "bad flag" in result.error
    assert result.exit_code == 2


@pytest.mark.asyncio
async def test_auth_required_pattern_match():
    async def fake_exec(*argv, **kwargs):
        return _fake_process(
            stdout=b"",
            stderr=b"Please run: claude auth\n",
            returncode=1,
        )

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert not result.ok
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_usage_limit_pattern_match():
    async def fake_exec(*argv, **kwargs):
        return _fake_process(
            stdout=b"",
            stderr=b"Error: rate_limit exceeded on monthly quota\n",
            returncode=1,
        )

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert not result.ok
    assert result.error == "usage_limit_reached"


@pytest.mark.asyncio
async def test_malformed_envelope_retries_once_then_fails(tmp_path):
    call_count = {"n": 0}

    async def fake_exec(*argv, **kwargs):
        call_count["n"] += 1
        return _fake_process(stdout=b"not json at all")

    runner = ClaudeCodeRunner(model="sonnet", post_mortem_dir=tmp_path)
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert call_count["n"] == 2  # one original + one retry
    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("parse_failed")
    assert any(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_cancellation_kills_child_and_propagates():
    proc = _fake_process()

    async def slow_communicate():
        await asyncio.sleep(10.0)
        return (b"", b"")

    proc.communicate = slow_communicate

    async def fake_exec(*argv, **kwargs):
        return proc

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        task = asyncio.create_task(
            runner.execute(prompt="p", schema=SCHEMA, timeout_s=60)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    proc.kill.assert_called()


@pytest.mark.asyncio
async def test_spawn_failed_returns_ok_false():
    async def fake_exec(*argv, **kwargs):
        raise FileNotFoundError("[Errno 2] No such file or directory: 'claude'")

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("spawn_failed")
    assert result.exit_code is None


@pytest.mark.asyncio
async def test_failure_results_are_not_cached(tmp_path):
    """Cache only successful results — transient failures must not poison the cache."""
    call_count = {"n": 0}
    responses = [
        # First call: malformed → parse_failed (after retry)
        b"not json",
        b"still not json",  # the inner retry from execute() also fails
        # Second call (different execute() invocation): valid envelope
        json.dumps({
            "type": "result",
            "subtype": "final_result",
            "result": json.dumps({"entity_name": "OK", "extractions": [], "diagnostics": ""}),
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "total_cost_usd": 0.0,
            "is_error": False,
        }).encode(),
    ]

    async def fake_exec(*argv, **kwargs):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return _fake_process(stdout=responses[idx])

    runner = ClaudeCodeRunner(model="sonnet", post_mortem_dir=tmp_path)
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        # First execute() — should fail (parse_failed) and NOT cache.
        result1 = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)
        assert not result1.ok
        assert result1.error is not None and result1.error.startswith("parse_failed")
        # Second execute() with the same prompt — should NOT hit a cached failure;
        # it should call the subprocess again and succeed this time.
        result2 = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)
        assert result2.ok, f"second call should not be served from cache, got {result2.error}"
        assert result2.data is not None
        assert result2.data.entity_name == "OK"


@pytest.mark.asyncio
async def test_cancellation_during_stdin_write_kills_child():
    """If CancelledError fires before reaching wait_for, the proc must still be killed."""
    proc = _fake_process()
    cancellation_event = asyncio.Event()

    async def slow_drain():
        # Block long enough for the test to cancel us.
        cancellation_event.set()
        await asyncio.sleep(10.0)

    proc.stdin.drain = slow_drain

    async def fake_exec(*argv, **kwargs):
        return proc

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        task = asyncio.create_task(
            runner.execute(prompt="p", schema=SCHEMA, timeout_s=60)
        )
        # Wait until the runner is blocked inside drain().
        await cancellation_event.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    proc.kill.assert_called()


@pytest.mark.asyncio
async def test_double_kill_does_not_mask_cancellation():
    """Reproduce I-6: cancel during wait_for triggers BOTH inner and outer handlers.

    The inner handler kills+reaps the proc; the outer handler then tries to kill
    again. If proc.kill() raises ProcessLookupError on the second call, it must
    NOT replace the in-flight CancelledError.
    """
    proc = _fake_process()

    # Make wait_for cancellable: communicate sleeps so cancellation lands inside it.
    async def slow_communicate():
        await asyncio.sleep(10.0)
        return (b"", b"")

    proc.communicate = slow_communicate

    # Make the SECOND call to proc.kill() raise ProcessLookupError, like the real OS would.
    kill_call_count = {"n": 0}

    def kill_with_lookup_error():
        kill_call_count["n"] += 1
        if kill_call_count["n"] >= 2:
            raise ProcessLookupError("[Errno 3] No such process")

    proc.kill = kill_with_lookup_error

    async def fake_exec(*argv, **kwargs):
        return proc

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        task = asyncio.create_task(
            runner.execute(prompt="p", schema=SCHEMA, timeout_s=60)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        # The exception MUST be CancelledError, not ProcessLookupError.
        with pytest.raises(asyncio.CancelledError):
            await task
