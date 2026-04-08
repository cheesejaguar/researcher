"""Tests for the `researcher run` CLI command.

The Wave 1-B wiring constructs a real Orchestrator + DuckDBKnowledgeStore +
EventBus from a YAML spec. These tests exercise the happy path of the wire-up.
`--backend api` now installs a full NativeAgentDeps bundle (SearchProvider +
HttpFetcher + PromptRegistry) and chooses between StubLLMClient and
OpenRouterClient based on whether OPENROUTER_API_KEY is set in the env, so
the native path runs end-to-end against the stub LLM when no API key is
configured.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def minimal_spec(tmp_path: Path) -> Path:
    spec_path = tmp_path / "minimal.yaml"
    spec_path.write_text(
        """
spec_id: minimal
goal: "Test"
entities:
  - name: Thing
    fields:
      - { name: name, type: str, required: true }
      - { name: count, type: int }
    search_templates: []
seeds:
  - "test seed"
models:
  fast: stub
  smart: stub
  heavy: stub
max_cycles: 1
budget_usd: 1.0
wall_limit_s: 60
"""
    )
    return spec_path


def test_run_help_shows_all_flags(cli_runner: CliRunner) -> None:
    from researcher.cli import app

    result = cli_runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0, result.output
    assert "--backend" in result.output
    assert "--obsidian-vault" in result.output
    assert "--runs-dir" in result.output


def test_run_creates_run_dir_with_store_and_events(
    cli_runner: CliRunner, minimal_spec: Path, tmp_path: Path, monkeypatch
) -> None:
    from researcher.cli import app

    # Ensure no OPENROUTER_API_KEY / TAVILY_API_KEY so the CLI falls back to
    # StubLLMClient + FileSeedsProvider on the native path.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    runs_dir = tmp_path / "runs"
    result = cli_runner.invoke(
        app,
        [
            "run",
            str(minimal_spec),
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            "test-run",
            "--backend",
            "api",
            "--fast-startup",
        ],
    )
    assert result.exit_code == 0, result.output

    run_dir = runs_dir / "test-run"
    assert run_dir.exists(), f"run directory was not created: {run_dir}"
    assert (run_dir / "store.duckdb").exists(), "DuckDB file missing"

    events_path = run_dir / "events.jsonl"
    assert events_path.exists(), "events.jsonl missing"
    content = events_path.read_text()
    assert '"type":"run_complete"' in content or '"type": "run_complete"' in content


def test_run_uses_openrouter_when_key_set(
    cli_runner: CliRunner, minimal_spec: Path, tmp_path: Path, monkeypatch
) -> None:
    """With OPENROUTER_API_KEY set, the CLI constructs an OpenRouterClient.

    Does NOT make a real API call — the spec has only one seed and the
    FileSeedsProvider (no TAVILY_API_KEY) returns no URLs, so the LLM is
    never actually invoked. This test is a wiring smoke check only.
    """
    from researcher.cli import app

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-fake-test-key")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    runs_dir = tmp_path / "runs"
    result = cli_runner.invoke(
        app,
        [
            "run",
            str(minimal_spec),
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            "or-smoke",
            "--backend",
            "api",
            "--fast-startup",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (runs_dir / "or-smoke" / "events.jsonl").exists()
