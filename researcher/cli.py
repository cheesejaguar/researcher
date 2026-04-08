"""Typer CLI — `researcher run | watch | inspect | resume | stop`.

Wave 0 ships a skeleton whose subcommands print their intent and exit. Wave 1-E
wires them into the real Orchestrator + EventBus + TUI child process.
"""

from __future__ import annotations

from pathlib import Path

import typer

from researcher import __version__

app = typer.Typer(
    name="researcher",
    help="Multi-agent deep-research orchestration.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run(
    spec: Path = typer.Argument(..., exists=True, readable=True, help="Path to a RunSpec YAML file."),
    run_id: str = typer.Option("", "--run-id", help="Custom run id (default: autogen)."),
    no_tui: bool = typer.Option(False, "--no-tui", help="Do not auto-launch the Ink TUI child process."),
    offline: bool = typer.Option(False, "--offline", help="Use stub LLM client + fixture corpus."),
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="Backend policy: auto (use CLI if detected), cli (force CLI), api (force OpenRouter).",
    ),
    obsidian_vault: str = typer.Option(
        "",
        "--obsidian-vault",
        help="Path to an Obsidian vault root. Overrides the spec file. Empty string disables.",
    ),
) -> None:
    """Run a research job from a YAML spec. Auto-launches the TUI by default."""
    if backend not in ("auto", "cli", "api"):
        raise typer.BadParameter(f"--backend must be one of auto|cli|api, got {backend!r}")
    typer.echo(
        f"[researcher run] spec={spec} run_id={run_id or 'auto'} "
        f"tui={'off' if no_tui else 'on'} offline={offline} backend={backend} "
        f"obsidian_vault={obsidian_vault or 'none'}"
    )
    typer.echo("Wave 0 skeleton: orchestrator dispatch lands in Wave 1-E.")


@app.command()
def watch(
    run_id: str = typer.Argument(..., help="Run id to attach to."),
    replay: bool = typer.Option(False, "--replay", help="Replay JSONL from disk instead of attaching to the live socket."),
) -> None:
    """Attach the TUI to a running (or completed) run."""
    typer.echo(f"[researcher watch] run_id={run_id} replay={replay}")
    typer.echo("Wave 0 skeleton: TUI attach lands in Wave 1-F.")


@app.command()
def inspect(
    run_id: str = typer.Argument(..., help="Run id to inspect."),
) -> None:
    """Print a summary of a completed run (entity count, fill %, cost, etc.)."""
    typer.echo(f"[researcher inspect] run_id={run_id}")
    typer.echo("Wave 0 skeleton: read run_summary from runs/<run_id>/store.duckdb.")


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run id to resume."),
) -> None:
    """Resume a previously-paused run from its last checkpoint."""
    typer.echo(f"[researcher resume] run_id={run_id}")
    typer.echo("Wave 0 skeleton: resume protocol lands in Wave 1-E.")


@app.command()
def stop(
    run_id: str = typer.Argument(..., help="Run id to stop."),
) -> None:
    """Send a graceful stop signal to a running orchestrator."""
    typer.echo(f"[researcher stop] run_id={run_id}")
    typer.echo("Wave 0 skeleton: stop protocol lands in Wave 1-E.")


@app.command()
def version() -> None:
    """Print researcher version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
