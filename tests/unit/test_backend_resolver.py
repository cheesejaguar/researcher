"""Tests for BackendResolver — autodetect + per-task pick + circuit break."""

import pytest

from researcher.backends.models import BackendUnavailableError, CliKind
from researcher.backends.resolver import BackendResolver


def _which_none(_: str) -> str | None:
    return None


def _which_claude_only(cmd: str) -> str | None:
    return "/usr/local/bin/claude" if cmd == "claude" else None


def _which_both(cmd: str) -> str | None:
    return {"claude": "/usr/local/bin/claude", "codex": "/usr/local/bin/codex"}.get(cmd)


def test_detects_nothing_when_which_returns_none():
    r = BackendResolver(which_fn=_which_none)
    assert r.detected == []


def test_detects_claude_only():
    r = BackendResolver(which_fn=_which_claude_only)
    assert r.detected == [CliKind.CLAUDE_CODE]


def test_detects_both_preferring_claude_first():
    r = BackendResolver(which_fn=_which_both)
    assert r.detected == [CliKind.CLAUDE_CODE, CliKind.CODEX]


def test_auto_policy_with_claude_picks_claude():
    r = BackendResolver(which_fn=_which_claude_only)
    choice = r.pick(policy="auto")
    assert choice.kind == CliKind.CLAUDE_CODE


def test_auto_policy_with_both_prefers_claude():
    r = BackendResolver(which_fn=_which_both)
    choice = r.pick(policy="auto")
    assert choice.kind == CliKind.CLAUDE_CODE


def test_auto_policy_with_none_falls_through():
    r = BackendResolver(which_fn=_which_none)
    choice = r.pick(policy="auto")
    assert choice.kind is None


def test_api_policy_always_returns_none():
    r = BackendResolver(which_fn=_which_both)
    choice = r.pick(policy="api")
    assert choice.kind is None


def test_cli_policy_with_none_raises():
    r = BackendResolver(which_fn=_which_none)
    with pytest.raises(BackendUnavailableError):
        r.pick(policy="cli")


def test_cli_policy_with_claude_returns_claude():
    r = BackendResolver(which_fn=_which_claude_only)
    choice = r.pick(policy="cli")
    assert choice.kind == CliKind.CLAUDE_CODE


def test_circuit_break_clears_detected():
    r = BackendResolver(which_fn=_which_both)
    assert r.detected == [CliKind.CLAUDE_CODE, CliKind.CODEX]
    r.clear_detected("usage_limit_reached")
    assert r.detected == []
    choice = r.pick(policy="auto")
    assert choice.kind is None
    assert "usage_limit" in choice.reason


def test_clear_detected_is_sticky_for_the_run():
    r = BackendResolver(which_fn=_which_both)
    r.clear_detected("auth_required")
    # Even after more calls, detected stays empty.
    r.pick(policy="auto")
    assert r.detected == []
