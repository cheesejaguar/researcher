"""Manual live test for ClaudeCodeRunner.

This test actually spawns `claude -p` and verifies it returns a parseable
response. It's excluded from CI in two ways:

1. The `manual` pytest marker (registered in pyproject.toml).
2. A module-level skip unless the environment variable
   `RESEARCHER_RUN_MANUAL` is set to a non-empty value.

Run locally with:

    RESEARCHER_RUN_MANUAL=1 uv run pytest tests/manual/test_live_claude.py -v

Skips gracefully if `claude` is not on PATH or the user is not authenticated.
"""

from __future__ import annotations

import os
import shutil

import pytest

from researcher.backends.cli_runner import ClaudeCodeRunner
from researcher.backends.models import SubagentResponse


pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(
        not os.environ.get("RESEARCHER_RUN_MANUAL"),
        reason="Manual test: set RESEARCHER_RUN_MANUAL=1 to run",
    ),
    pytest.mark.skipif(
        shutil.which("claude") is None,
        reason="claude CLI not found on PATH",
    ),
]


@pytest.mark.asyncio
async def test_live_claude_returns_structured_response():
    """Spawn a real `claude -p` with WebSearch disabled and a minimal schema.

    Prompt is intentionally trivial and self-contained (no web tools needed)
    so the test is fast and deterministic-ish. The goal is to verify the
    CLI flags and envelope parsing still match what the CLI produces today.
    If this test fails, the CLI likely changed its flag surface or JSON
    envelope shape, and ClaudeCodeRunner needs to be updated.
    """
    runner = ClaudeCodeRunner(
        model="sonnet",
        # No disk cache — we want a fresh call every run.
    )
    schema = SubagentResponse.model_json_schema()
    prompt = (
        "Extract facts about 'Python programming language'. Return exactly one "
        "entity with entity_name='Python programming language' and two extractions: "
        "field='created_year' value=1991, and field='creator' value='Guido van Rossum'. "
        "Use source_url='https://en.wikipedia.org/wiki/Python_(programming_language)' "
        "and confidence=0.95 for both. snippet can be empty string. diagnostics='manual test'."
    )

    # Disable tools — the model has enough from the prompt itself.
    # 60s timeout: plenty for a tiny prompt without web calls.
    result = await runner.execute(
        prompt=prompt,
        schema=schema,
        timeout_s=60.0,
        tools=(),
    )

    if not result.ok:
        pytest.fail(
            f"Live claude call failed: error={result.error!r}, "
            f"exit_code={result.exit_code}, wall_ms={result.wall_ms}"
        )

    assert result.data is not None
    assert result.data.entity_name  # non-empty
    # We don't assert on exact extraction content — the model may paraphrase
    # or reorder. We just verify the envelope round-tripped.
    print(f"live claude response: {result.data.model_dump_json(indent=2)}")
    print(f"wall_ms={result.wall_ms}, raw_usage={result.raw_usage}")
