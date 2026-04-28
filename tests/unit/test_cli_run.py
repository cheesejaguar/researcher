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

import json
import subprocess
import sys
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


@pytest.fixture
def source_spec(tmp_path: Path) -> Path:
    source = tmp_path / "source.md"
    source.write_text("Alpha source text for source pack search.")
    spec_path = tmp_path / "source_spec.yaml"
    spec_path.write_text(
        f"""
spec_id: sourced
goal: "Source test"
entities:
  - name: Thing
    fields:
      - {{ name: name, type: str, required: true }}
    search_templates: []
seeds:
  - "Alpha"
source_sets:
  - name: local
    paths:
      - "{source}"
source_policy:
  trusted_domains: []
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


def test_python_module_entrypoint_help_works() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "researcher", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Multi-agent deep-research orchestration" in result.stdout


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

    inspect_result = cli_runner.invoke(
        app,
        [
            "inspect",
            "test-run",
            "--runs-dir",
            str(runs_dir),
            "--json",
        ],
    )
    assert inspect_result.exit_code == 0, inspect_result.output
    summary = json.loads(inspect_result.output)
    assert summary["run_id"] == "test-run"
    assert summary["db_path"].endswith("store.duckdb")

    accept_result = cli_runner.invoke(
        app,
        [
            "accept",
            "test-run",
            "--runs-dir",
            str(runs_dir),
            "--min-entities",
            "0",
            "--min-field-fill",
            "0",
            "--max-cost-usd",
            "999",
            "--max-wall-s",
            "999",
            "--json",
        ],
    )
    assert accept_result.exit_code == 0, accept_result.output
    assert json.loads(accept_result.output)["passed"] is True


def test_inspect_and_accept_default_to_human_output(
    cli_runner: CliRunner, minimal_spec: Path, tmp_path: Path, monkeypatch
) -> None:
    from researcher.cli import app

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    runs_dir = tmp_path / "runs"
    run_result = cli_runner.invoke(
        app,
        [
            "run",
            str(minimal_spec),
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            "human-run",
            "--backend",
            "api",
            "--fast-startup",
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    inspect_result = cli_runner.invoke(
        app, ["inspect", "human-run", "--runs-dir", str(runs_dir)]
    )
    assert inspect_result.exit_code == 0, inspect_result.output
    assert "run_id" in inspect_result.output
    assert "strict_field_fill" in inspect_result.output

    accept_result = cli_runner.invoke(
        app,
        [
            "accept",
            "human-run",
            "--runs-dir",
            str(runs_dir),
            "--min-entities",
            "0",
            "--min-field-fill",
            "0",
            "--max-cost-usd",
            "999",
            "--max-wall-s",
            "999",
        ],
    )
    assert accept_result.exit_code == 0, accept_result.output
    assert "acceptance: PASS" in accept_result.output


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


def test_offline_auto_forces_api_backend(
    cli_runner: CliRunner, minimal_spec: Path, tmp_path: Path, monkeypatch
) -> None:
    from researcher.cli import app

    monkeypatch.setenv("OPENROUTER_API_KEY", "ignored-offline")
    runs_dir = tmp_path / "runs"
    result = cli_runner.invoke(
        app,
        [
            "run",
            str(minimal_spec),
            "--offline",
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            "offline-auto",
            "--fast-startup",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "--offline forces backend=api" in result.output
    meta = json.loads((runs_dir / "offline-auto" / "run.json").read_text())
    assert meta["backend"] == "api"


def test_doctor_json_reports_spec_status(
    cli_runner: CliRunner, minimal_spec: Path, tmp_path: Path
) -> None:
    from researcher.cli import app

    result = cli_runner.invoke(
        app,
        [
            "doctor",
            "--spec",
            str(minimal_spec),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["passed"] is True
    assert any(c["name"] == "spec" and c["status"] == "pass" for c in payload["checks"])


def test_init_spec_scaffolds_spec_and_skill(cli_runner: CliRunner, tmp_path: Path) -> None:
    from researcher.cli import app

    spec_path = tmp_path / "specs" / "demo.yaml"
    skill_path = tmp_path / "skills" / "demo.yaml"
    result = cli_runner.invoke(
        app,
        [
            "init",
            "spec",
            "--output",
            str(spec_path),
            "--skill-output",
            str(skill_path),
            "--spec-id",
            "demo",
            "--goal",
            "Demo research",
            "--entity",
            "DemoThing",
            "--seed",
            "demo seed",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "spec_id: demo" in spec_path.read_text()
    assert "entity_type: DemoThing" in skill_path.read_text()


def test_stop_writes_stop_request(cli_runner: CliRunner, tmp_path: Path) -> None:
    from researcher.cli import app

    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-stop"
    run_dir.mkdir(parents=True)
    result = cli_runner.invoke(
        app,
        ["stop", "run-stop", "--runs-dir", str(runs_dir)],
    )
    assert result.exit_code == 0, result.output
    assert (run_dir / "stop.requested").exists()


def test_export_json_and_csv(
    cli_runner: CliRunner, minimal_spec: Path, tmp_path: Path, monkeypatch
) -> None:
    from researcher.cli import app

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
            "export-run",
            "--backend",
            "api",
            "--fast-startup",
        ],
    )
    assert result.exit_code == 0, result.output

    json_result = cli_runner.invoke(
        app, ["export", "export-run", "--runs-dir", str(runs_dir), "--format", "json"]
    )
    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.output)["run_id"] == "export-run"

    csv_result = cli_runner.invoke(
        app, ["export", "export-run", "--runs-dir", str(runs_dir), "--format", "csv"]
    )
    assert csv_result.exit_code == 0, csv_result.output
    assert "id,type,name" in csv_result.output

    evidence_result = cli_runner.invoke(
        app, ["evidence", "export-run", "--runs-dir", str(runs_dir), "--format", "csv"]
    )
    assert evidence_result.exit_code == 0, evidence_result.output
    assert "entity_type,entity_name,field" in evidence_result.output

    report_result = cli_runner.invoke(
        app, ["report", "export-run", "--runs-dir", str(runs_dir), "--format", "md"]
    )
    assert report_result.exit_code == 0, report_result.output
    assert "# Analyst Research Report: export-run" in report_result.output

    runs_result = cli_runner.invoke(
        app, ["runs", "search", "export-run", "--runs-dir", str(runs_dir), "--json"]
    )
    assert runs_result.exit_code == 0, runs_result.output
    assert json.loads(runs_result.output)["runs"][0]["run_id"] == "export-run"


def test_sources_build_and_inspect(
    cli_runner: CliRunner, source_spec: Path, tmp_path: Path
) -> None:
    from researcher.cli import app

    runs_dir = tmp_path / "runs"
    build = cli_runner.invoke(
        app,
        [
            "sources",
            "build",
            str(source_spec),
            "--run-id",
            "source-run",
            "--runs-dir",
            str(runs_dir),
            "--json",
        ],
    )
    assert build.exit_code == 0, build.output
    payload = json.loads(build.output)
    assert payload["sources"] == 1
    assert payload["chunks"] >= 1

    inspect = cli_runner.invoke(
        app,
        [
            "sources",
            "inspect",
            "source-run",
            "--runs-dir",
            str(runs_dir),
            "--json",
        ],
    )
    assert inspect.exit_code == 0, inspect.output
    assert json.loads(inspect.output)["sources"][0]["source_type"] == "file"


def test_run_accepts_items_csv(
    cli_runner: CliRunner, minimal_spec: Path, tmp_path: Path, monkeypatch
) -> None:
    from researcher.cli import app

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    items_path = tmp_path / "items.csv"
    items_path.write_text("name,id\nAlpha,a\nBeta,b\n")
    runs_dir = tmp_path / "runs"
    result = cli_runner.invoke(
        app,
        [
            "run",
            str(minimal_spec),
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            "items-run",
            "--backend",
            "api",
            "--fast-startup",
            "--items",
            str(items_path),
            "--item-column",
            "name",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "loaded_items=2" in result.output
    meta = json.loads((runs_dir / "items-run" / "run.json").read_text())
    assert meta["items_path"] == str(items_path)
