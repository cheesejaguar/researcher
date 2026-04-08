"""In-memory stub CliRunner — returns canned CliResult values by prompt hash.

Usage:
    stub = StubCliRunner()
    stub.add_response("hello world", CliResult(ok=True, data=resp, wall_ms=10, exit_code=0))
    result = await stub.execute(prompt="hello world", schema=SCHEMA, timeout_s=30)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from researcher.backends.models import CliResult, SubagentResponse


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def load_fixture_response(path: Path) -> SubagentResponse:
    """Load a canned SubagentResponse JSON from disk."""
    data = json.loads(path.read_text())
    return SubagentResponse.model_validate(data)


class StubCliRunner:
    def __init__(self) -> None:
        self._responses: dict[str, CliResult] = {}
        self._default: CliResult = CliResult(
            ok=False,
            error="stub: no fixture",
            wall_ms=0,
            exit_code=None,
        )
        self.calls: list[dict[str, Any]] = []

    def add_response(self, prompt: str, result: CliResult) -> None:
        self._responses[_hash_prompt(prompt)] = result

    def add_response_for_any(self, result: CliResult) -> None:
        """Return this result for any prompt not explicitly registered."""
        self._default = result

    async def execute(
        self,
        prompt: str,
        schema: dict,
        timeout_s: float,
        tools: tuple[str, ...] = ("WebSearch", "WebFetch"),
    ) -> CliResult:
        key = _hash_prompt(prompt)
        self.calls.append({"prompt_hash": key, "timeout_s": timeout_s, "tools": list(tools)})
        return self._responses.get(key, self._default)


def make_wars_discover_result() -> CliResult:
    """Convenience factory for the wars_discover.json fixture."""
    here = Path(__file__).parent.parent / "fixtures" / "subagent_responses" / "wars_discover.json"
    return CliResult(
        ok=True,
        data=load_fixture_response(here),
        wall_ms=1200,
        exit_code=0,
        raw_usage={"input_tokens": 500, "output_tokens": 300},
    )


def make_empty_result() -> CliResult:
    here = Path(__file__).parent.parent / "fixtures" / "subagent_responses" / "wars_empty.json"
    return CliResult(
        ok=True,
        data=load_fixture_response(here),
        wall_ms=800,
        exit_code=0,
        raw_usage={"input_tokens": 400, "output_tokens": 50},
    )


def make_timeout_result() -> CliResult:
    return CliResult(ok=False, error="timeout", wall_ms=120_000, exit_code=None)
