"""Tests for the Orchestrator's Obsidian writer hook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from researcher.backends.resolver import BackendResolver
from researcher.budget import Budget
from researcher.integrations.obsidian import ObsidianWriter
from researcher.orchestrator import Orchestrator, StopReason
from researcher.scheduler import Scheduler
from researcher.spec import EntitySpec, FieldSpec, RunSpec
from researcher.storage.writer import FactWriter
from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
from tests.stubs.llm import StubLLMClient
from tests.support import make_resolver


class CountingObsidianWriter(ObsidianWriter):
    """The real ObsidianWriter (writing into a temp vault) with call counters
    on the lifecycle hooks the orchestrator is responsible for driving."""

    def __init__(self, vault_path: Path, raise_on_fact: bool = False) -> None:
        super().__init__(vault_path=vault_path)
        self.started = 0
        self.stopped = 0
        self.flushed = 0
        self.raise_on_fact = raise_on_fact

    async def start(self) -> None:
        self.started += 1
        await super().start()

    async def on_fact(self, claim, run_id: str) -> None:
        if self.raise_on_fact:
            raise RuntimeError("simulated obsidian failure")
        await super().on_fact(claim, run_id)

    async def flush(self) -> None:
        self.flushed += 1
        await super().flush()

    async def stop(self) -> None:
        self.stopped += 1
        await super().stop()


def _which_claude(cmd: str) -> str | None:
    return "/usr/local/bin/claude" if cmd == "claude" else None


async def _noop_fact(_entity_id, _claim):
    return None


async def _noop_conflict(_entity_id, _cells):
    return None


async def _make_orchestrator(
    store,
    bus,
    obsidian: Any,
    runner: StubCliRunner | None = None,
) -> Orchestrator:
    if runner is None:
        runner = StubCliRunner()
        runner.add_response_for_any(make_wars_discover_result())
    spec = RunSpec(
        spec_id="wars",
        goal="Wars",
        entities=[
            EntitySpec(
                name="War",
                fields=[FieldSpec(name="name", type="str", required=True)],
                search_templates=[],
            )
        ],
        seeds=["seed1"],
        models={"fast": "m", "smart": "m", "heavy": "m"},
        backend_policy="auto",
        max_cycles=1,
    )
    llm = StubLLMClient()
    resolver = make_resolver(store)
    entity_schema = {
        "entity_type": "War",
        "fields": [{"name": "name", "type": "str", "required": True}],
    }
    writer = FactWriter(
        store=store,
        resolver=resolver,
        entity_schema=entity_schema,
        emit_fact=_noop_fact,
        emit_conflict=_noop_conflict,
    )
    scheduler = Scheduler(spec=spec, store=store)
    budget = Budget(usd_cap=3.0, wall_cap_s=600)
    backend_resolver = BackendResolver(which_fn=_which_claude)
    orch = Orchestrator(
        spec=spec,
        store=store,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        bus=bus,  # type: ignore[arg-type]
        writer=writer,
        scheduler=scheduler,
        budget=budget,
        run_id="run-x",
        max_parallel_agents=2,
        obsidian_writer=obsidian,
    )
    orch.set_backend_resolver(backend_resolver)
    orch.set_cli_runner_factory(lambda kind: runner)
    return orch


@pytest.mark.asyncio
async def test_orchestrator_forwards_claims_to_obsidian_writer(tmp_path: Path, store, bus):
    ob = CountingObsidianWriter(tmp_path)
    orch = await _make_orchestrator(store, bus, ob)
    await orch.run()

    note = tmp_path / "researcher" / "War" / "World War II.md"
    assert note.exists()
    text = note.read_text()
    # All four extractions from the wars_discover fixture reach the note,
    # tagged with this run's id.
    for expected in ("1939", "1945", "Allies", "Axis", "run-x"):
        assert expected in text
    assert ob.stats["errors"] == 0
    # The post-run hook writes the per-type Dataview index.
    assert (tmp_path / "researcher" / "War" / "_index.md").exists()


@pytest.mark.asyncio
async def test_orchestrator_calls_start_and_stop_exactly_once(tmp_path: Path, store, bus):
    ob = CountingObsidianWriter(tmp_path)
    orch = await _make_orchestrator(store, bus, ob)
    await orch.run()
    assert ob.started == 1
    assert ob.stopped == 1


@pytest.mark.asyncio
async def test_orchestrator_flushes_at_cycle_end(tmp_path: Path, store, bus):
    ob = CountingObsidianWriter(tmp_path)
    orch = await _make_orchestrator(store, bus, ob)
    await orch.run()
    # One cycle -> one cycle-end flush, plus the final flush inside stop().
    assert ob.flushed == 2


@pytest.mark.asyncio
async def test_orchestrator_without_obsidian_writer_is_baseline(store, bus):
    orch = await _make_orchestrator(store, bus, None)
    reason = await orch.run()
    assert reason == StopReason.PLATEAU
    rows = await store.query("SELECT name FROM entities")
    assert [r["name"] for r in rows] == ["World War II"]


@pytest.mark.asyncio
async def test_obsidian_write_failure_does_not_crash_run(tmp_path: Path, store, bus):
    ob = CountingObsidianWriter(tmp_path, raise_on_fact=True)
    orch = await _make_orchestrator(store, bus, ob)
    reason = await orch.run()
    # The secondary sink failing must not change the run outcome or block
    # the primary store write.
    assert reason == StopReason.PLATEAU
    rows = await store.query("SELECT name FROM entities")
    assert [r["name"] for r in rows] == ["World War II"]
    assert ob.stopped == 1
