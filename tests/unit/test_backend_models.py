"""Tests for backend Pydantic models."""

import pytest
from pydantic import ValidationError

from researcher.backends.models import (
    BackendChoice,
    BackendUnavailableError,
    CliKind,
    CliResult,
    Extraction,
    SubagentResponse,
)


def test_clikind_values():
    assert [k.value for k in CliKind] == ["claude_code", "codex"]


def test_extraction_confidence_bounds():
    with pytest.raises(ValidationError):
        Extraction(
            field="start_year",
            value=1939,
            source_url="https://example.com",
            snippet="s",
            confidence=1.5,
        )


def test_subagent_response_happy_path():
    r = SubagentResponse(
        entity_name="WWII",
        extractions=[
            Extraction(
                field="start_year",
                value=1939,
                source_url="https://example.com/wwii",
                snippet="World War II began in 1939...",
                confidence=0.95,
            )
        ],
        diagnostics="found via wikipedia",
    )
    assert r.entity_name == "WWII"
    assert len(r.extractions) == 1


def test_cli_result_ok_shape():
    r = CliResult(
        ok=True,
        data=SubagentResponse(entity_name="X", extractions=[], diagnostics=""),
        error=None,
        wall_ms=1234,
        exit_code=0,
        raw_usage={"input_tokens": 100, "output_tokens": 50},
    )
    assert r.ok
    assert r.data is not None
    assert r.raw_usage["input_tokens"] == 100


def test_cli_result_failure_shape():
    r = CliResult(
        ok=False,
        data=None,
        error="timeout",
        wall_ms=120_000,
        exit_code=None,
        raw_usage=None,
    )
    assert not r.ok
    assert r.error == "timeout"


def test_backend_choice_defaults():
    c = BackendChoice(kind=None, reason="auto: no CLI detected")
    assert c.kind is None


def test_backend_choice_with_kind():
    c = BackendChoice(kind=CliKind.CLAUDE_CODE, reason="auto: claude detected")
    assert c.kind == CliKind.CLAUDE_CODE


def test_backend_unavailable_error_is_runtime_error():
    assert issubclass(BackendUnavailableError, RuntimeError)
