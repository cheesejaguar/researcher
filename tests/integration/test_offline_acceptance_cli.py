"""Offline CLI acceptance harness smokes for the public example specs."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner


def _run_and_accept(spec_path: str, run_id: str, tmp_path: Path) -> dict:
    from researcher.cli import app

    runs_dir = tmp_path / "runs"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            spec_path,
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            run_id,
            "--backend",
            "api",
            "--offline",
            "--fast-startup",
            "--no-tui",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (runs_dir / run_id / "events.jsonl").exists()
    assert (runs_dir / run_id / "store.duckdb").exists()

    inspect_result = runner.invoke(
        app,
        ["inspect", run_id, "--runs-dir", str(runs_dir), "--json"],
    )
    assert inspect_result.exit_code == 0, inspect_result.output
    inspected = json.loads(inspect_result.output)
    assert inspected["entities_total"] >= 1
    assert inspected["reason"] in {"no_tasks", "plateau", "budget", "deadline"}

    accept_result = runner.invoke(
        app,
        [
            "accept",
            run_id,
            "--runs-dir",
            str(runs_dir),
            "--min-entities",
            "1",
            "--min-field-fill",
            "0.2",
            "--max-cost-usd",
            "3",
            "--max-wall-s",
            "600",
            "--json",
        ],
    )
    assert accept_result.exit_code == 0, accept_result.output
    accepted = json.loads(accept_result.output)
    assert accepted["passed"] is True

    evidence_result = runner.invoke(
        app,
        [
            "evidence",
            run_id,
            "--runs-dir",
            str(runs_dir),
            "--format",
            "csv",
        ],
    )
    assert evidence_result.exit_code == 0, evidence_result.output
    assert "entity_type,entity_name,field" in evidence_result.output

    report_result = runner.invoke(
        app,
        [
            "report",
            run_id,
            "--runs-dir",
            str(runs_dir),
            "--format",
            "md",
        ],
    )
    assert report_result.exit_code == 0, report_result.output
    assert f"# Analyst Research Report: {run_id}" in report_result.output
    return inspected


def test_wars_spec_offline_acceptance(tmp_path: Path) -> None:
    inspected = _run_and_accept("specs/wars.yaml", "wars-offline", tmp_path)
    assert "War" in inspected["entities_by_type"]


def test_glp1_spec_offline_acceptance(tmp_path: Path) -> None:
    inspected = _run_and_accept(
        "specs/glp1_trials.yaml", "glp1-offline", tmp_path
    )
    assert "ClinicalTrial" in inspected["entities_by_type"]
