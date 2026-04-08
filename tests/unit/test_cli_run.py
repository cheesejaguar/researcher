"""Tests for the `researcher run` CLI command.

The Wave 1-B wiring constructs a real Orchestrator + DuckDBKnowledgeStore +
EventBus from a YAML spec. These tests exercise the happy path of the wire-up
(the graceful-error path, actually — the Wave 1-D native-agent path is not
yet implemented, so `--backend api` short-circuits into ``StopReason.ERROR``).
What we verify is that on the path to that graceful error the CLI creates
the expected artifacts: the run directory, the DuckDB file, and an
events.jsonl that ends with a `run_complete` envelope.
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
    cli_runner: CliRunner, minimal_spec: Path, tmp_path: Path
) -> None:
    from researcher.cli import app

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
            # Force the api path so we don't depend on CLI detection.
            # The orchestrator raises NotImplementedError inside _spawn_agent
            # (Wave 1-D), which the cycle loop catches and turns into
            # StopReason.ERROR — the artifacts must still exist.
            "--backend",
            "api",
        ],
    )
    # We don't assert exit_code == 0: the orchestrator ends in StopReason.ERROR
    # because the native agent path isn't built yet, but the CLI itself completes
    # cleanly (it prints "done reason=error" and returns 0). If anything at the
    # construction stage blew up, the assertions below would still give us a
    # useful signal.
    assert result.exit_code == 0, result.output

    run_dir = runs_dir / "test-run"
    assert run_dir.exists(), f"run directory was not created: {run_dir}"
    assert (run_dir / "store.duckdb").exists(), "DuckDB file missing"

    events_path = run_dir / "events.jsonl"
    assert events_path.exists(), "events.jsonl missing"
    content = events_path.read_text()
    # Every orchestrator run emits exactly one run_complete envelope from
    # _graceful_stop, regardless of StopReason.
    assert '"type":"run_complete"' in content or '"type": "run_complete"' in content
