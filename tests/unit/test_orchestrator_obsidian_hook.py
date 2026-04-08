"""Tests for the Orchestrator's Obsidian writer hook."""

from __future__ import annotations

from typing import Any

import pytest

from researcher.backends.resolver import BackendResolver
from researcher.budget import Budget
from researcher.orchestrator import Orchestrator
from researcher.scheduler import Scheduler
from researcher.spec import EntitySpec, FieldSpec, RunSpec
from researcher.storage.writer import FactWriter
from tests.stubs.bus import StubEventBus
from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
from tests.stubs.llm import StubLLMClient
from tests.stubs.resolver import StubEntityResolver
from tests.stubs.store import StubKnowledgeStore


class MockObsidianWriter:
    """Spy that records all on_fact/start/flush/stop calls."""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.flushed = 0
        self.facts: list[tuple[str, str]] = []
        self.raise_on_fact: bool = False

    async def start(self) -> None:
        self.started += 1

    async def on_fact(self, claim, run_id: str) -> None:
        if self.raise_on_fact:
            raise RuntimeError("simulated obsidian failure")
        self.facts.append((claim.field, run_id))

    async def flush(self) -> None:
        self.flushed += 1

    async def stop(self) -> None:
        self.stopped += 1


def _which_claude(cmd: str) -> str | None:
    return "/usr/local/bin/claude" if cmd == "claude" else None


async def _noop_fact(_entity_id, _claim):
    return None


async def _noop_conflict(_entity_id, _cells):
    return None


async def _make_orchestrator(
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
    store = StubKnowledgeStore()
    await store.open()
    llm = StubLLMClient()
    bus = StubEventBus()
    resolver = StubEntityResolver()
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
async def test_orchestrator_forwards_claims_to_obsidian_writer():
    ob = MockObsidianWriter()
    orch = await _make_orchestrator(ob)
    await orch.run()
    # The wars_discover fixture has 4 extractions, so 4 on_fact calls.
    assert len(ob.facts) == 4
    # Every call passed the run_id.
    assert all(run_id == "run-x" for _, run_id in ob.facts)


@pytest.mark.asyncio
async def test_orchestrator_calls_start_and_stop_exactly_once():
    ob = MockObsidianWriter()
    orch = await _make_orchestrator(ob)
    await orch.run()
    assert ob.started == 1
    assert ob.stopped == 1


@pytest.mark.asyncio
async def test_orchestrator_flushes_at_cycle_end():
    ob = MockObsidianWriter()
    orch = await _make_orchestrator(ob)
    await orch.run()
    # One cycle → at least one flush call.
    assert ob.flushed >= 1


@pytest.mark.asyncio
async def test_orchestrator_without_obsidian_writer_is_baseline():
    orch = await _make_orchestrator(None)
    reason = await orch.run()
    assert reason is not None


@pytest.mark.asyncio
async def test_obsidian_write_failure_does_not_crash_run():
    ob = MockObsidianWriter()
    ob.raise_on_fact = True
    orch = await _make_orchestrator(ob)
    reason = await orch.run()
    assert reason is not None
    assert ob.stopped == 1
