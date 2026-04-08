"""Tests for the v1.2 #8 coverage gap structured output.

Covers:
- the new ``CoverageReport`` event + ``CoverageReportPayload`` schema,
- the ``KnowledgeStore.snapshot_coverage`` ABC method (stub + DuckDB),
- the orchestrator integration that emits a ``CoverageReport`` after each
  ``cycle_end``,
- the ``_recommend_next_seeds`` heuristic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from researcher.events import (
    CoverageReport,
    CoverageReportPayload,
    CycleEnd,
    parse_event,
)
from researcher.orchestrator import Orchestrator
from researcher.storage.duckdb_store import DuckDBKnowledgeStore
from researcher.storage.store import (
    CoverageSnapshot,
    FieldCell,
    StoreMetrics,
    _classify_source,
)
from tests.stubs.bus import StubEventBus
from tests.stubs.store import StubKnowledgeStore

RUN_ID = "run-coverage"


def _ts() -> datetime:
    return datetime(2026, 4, 8, 12, 0, tzinfo=UTC)


def _cell(value, conf: float = 0.9) -> FieldCell:
    return FieldCell(
        value=value,
        confidence=conf,
        provenance_ids=[],
        updated_at=datetime.now(UTC),
    )


# ---------- Payload shape ----------


def test_coverage_report_payload_accepts_all_fields():
    payload = CoverageReportPayload(
        cycle=2,
        entities_by_type={"War": 4, "Person": 11},
        fields_below_confidence={"start_year": 1, "end_year": 2},
        confidence_threshold=0.5,
        conflicts_open=3,
        source_type_breakdown={"http": 6, "file": 2, "stub": 1},
        next_recommended_seeds=[
            "expand coverage of War",
            "resolve 3 open conflicts",
        ],
    )
    dumped = payload.model_dump()
    assert dumped["cycle"] == 2
    assert dumped["entities_by_type"]["War"] == 4
    assert dumped["fields_below_confidence"]["start_year"] == 1
    assert dumped["confidence_threshold"] == 0.5
    assert dumped["conflicts_open"] == 3
    assert dumped["source_type_breakdown"]["http"] == 6
    assert "expand coverage of War" in dumped["next_recommended_seeds"]
    # Round trip.
    again = CoverageReportPayload.model_validate(dumped)
    assert again == payload


def test_coverage_report_is_parseable_by_parse_event():
    raw = {
        "type": "coverage_report",
        "seq": 7,
        "ts": _ts().isoformat(),
        "run_id": RUN_ID,
        "payload": {
            "cycle": 1,
            "entities_by_type": {"War": 2},
            "fields_below_confidence": {"start_year": 1},
            "confidence_threshold": 0.5,
            "conflicts_open": 0,
            "source_type_breakdown": {"http": 3},
            "next_recommended_seeds": ["expand coverage of War"],
        },
    }
    evt = parse_event(raw)
    assert isinstance(evt, CoverageReport)
    assert evt.payload.cycle == 1
    assert evt.payload.entities_by_type == {"War": 2}
    assert evt.payload.next_recommended_seeds == ["expand coverage of War"]


# ---------- _classify_source helper ----------


def test_classify_source_categories():
    assert _classify_source("http://example.com") == "http"
    assert _classify_source("https://example.com") == "http"
    assert _classify_source("file:///tmp/foo.txt") == "file"
    assert _classify_source("native://discover") == "native"
    assert _classify_source("stub://x") == "stub"
    assert _classify_source("ftp://example.com") == "other"
    assert _classify_source("") == "other"


# ---------- StubKnowledgeStore.snapshot_coverage ----------


@pytest.mark.asyncio
async def test_stub_snapshot_coverage_empty():
    store = StubKnowledgeStore()
    await store.open()
    snap = await store.snapshot_coverage()
    assert isinstance(snap, CoverageSnapshot)
    assert snap.fields_below_confidence == {}
    assert snap.source_type_breakdown == {}


@pytest.mark.asyncio
async def test_stub_snapshot_coverage_counts_low_confidence_fields():
    store = StubKnowledgeStore()
    await store.open()
    # Three cells on the same field name; only the last two are < 0.5.
    await store.upsert_entity(
        "War",
        "WWI",
        {"start_year": _cell(1914, conf=0.9)},
    )
    await store.upsert_entity(
        "War",
        "WWII",
        {"start_year": _cell(1939, conf=0.4)},
    )
    await store.upsert_entity(
        "War",
        "Korea",
        {"start_year": _cell(1950, conf=0.3)},
    )
    snap = await store.snapshot_coverage(confidence_threshold=0.5)
    assert snap.fields_below_confidence == {"start_year": 2}


@pytest.mark.asyncio
async def test_stub_snapshot_coverage_classifies_sources():
    store = StubKnowledgeStore()
    await store.open()
    urls = [
        "http://a",
        "https://b",
        "file:///c",
        "native://d",
        "stub://e",
        "ftp://f",
    ]
    for i, url in enumerate(urls):
        await store.record_provenance(
            f"prov-{i}",
            {"entity_id": f"e-{i}", "field": "x", "url": url},
        )
    snap = await store.snapshot_coverage()
    assert snap.source_type_breakdown == {
        "http": 2,
        "file": 1,
        "native": 1,
        "stub": 1,
        "other": 1,
    }


# ---------- DuckDBKnowledgeStore.snapshot_coverage ----------


class _WarEntity(BaseModel):
    name: str
    start_year: int
    end_year: int | None = None


@pytest.mark.asyncio
async def test_duckdb_snapshot_coverage_counts_low_confidence(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "cov.duckdb")
    await store.open()
    try:
        await store.init_schema(_WarEntity)
        await store.upsert_entity(
            "War",
            "WWI",
            {"start_year": _cell(1914, conf=0.9)},
        )
        await store.upsert_entity(
            "War",
            "WWII",
            {"start_year": _cell(1939, conf=0.4)},
        )
        await store.upsert_entity(
            "War",
            "Korea",
            {"start_year": _cell(1950, conf=0.3)},
        )
        snap = await store.snapshot_coverage(confidence_threshold=0.5)
        assert snap.fields_below_confidence == {"start_year": 2}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_duckdb_snapshot_coverage_classifies_sources(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "cov2.duckdb")
    await store.open()
    try:
        await store.init_schema(_WarEntity)
        urls = [
            "http://a",
            "https://b",
            "file:///c",
            "native://d",
            "stub://e",
            "ftp://f",
        ]
        for i, url in enumerate(urls):
            await store.record_provenance(
                f"prov-{i}",
                {"entity_id": f"e-{i}", "field": "x", "url": url},
            )
        snap = await store.snapshot_coverage()
        assert snap.source_type_breakdown == {
            "http": 2,
            "file": 1,
            "native": 1,
            "stub": 1,
            "other": 1,
        }
    finally:
        await store.close()


# ---------- _recommend_next_seeds heuristic ----------


def _metrics_for(by_type: dict[str, int], conflicts_open: int = 0) -> StoreMetrics:
    return StoreMetrics(
        entities_total=sum(by_type.values()),
        by_type=by_type,
        fields_filled_pct=0.0,
        conflicts_open=conflicts_open,
        cost_usd_total=0.0,
    )


def test_coverage_report_recommends_expand_for_sparse_types():
    metrics = _metrics_for({"War": 2})
    snap = CoverageSnapshot(
        fields_below_confidence={},
        source_type_breakdown={},
    )
    seeds = Orchestrator._recommend_next_seeds(metrics, snap)
    assert any("expand" in s and "War" in s for s in seeds)


def test_coverage_report_recommends_resolve_for_open_conflicts():
    metrics = _metrics_for({"War": 50}, conflicts_open=3)
    snap = CoverageSnapshot(
        fields_below_confidence={},
        source_type_breakdown={},
    )
    seeds = Orchestrator._recommend_next_seeds(metrics, snap)
    assert any("resolve" in s and "3" in s for s in seeds)


def test_coverage_report_recommends_verify_for_low_confidence():
    metrics = _metrics_for({"War": 50})
    snap = CoverageSnapshot(
        fields_below_confidence={"start_year": 4, "end_year": 2},
        source_type_breakdown={},
    )
    seeds = Orchestrator._recommend_next_seeds(metrics, snap)
    assert any("verify" in s for s in seeds)
    joined = " ".join(seeds)
    assert "start_year" in joined or "end_year" in joined


def test_coverage_report_recommendations_capped_at_three():
    # All three categories fire, with several sparse types and several
    # low-confidence fields. We should still get at most 3 strings.
    metrics = _metrics_for({"War": 2, "Person": 1, "Place": 1}, conflicts_open=5)
    snap = CoverageSnapshot(
        fields_below_confidence={"a": 3, "b": 2, "c": 1},
        source_type_breakdown={},
    )
    seeds = Orchestrator._recommend_next_seeds(metrics, snap)
    assert len(seeds) <= 3


# ---------- Orchestrator integration ----------


def _sample_spec_for_orch(max_cycles: int = 1):
    from researcher.spec import EntitySpec, FieldSpec, RunSpec

    return RunSpec(
        spec_id="wars",
        goal="Wars",
        entities=[
            EntitySpec(
                name="War",
                fields=[FieldSpec(name="name", type="str", required=True)],
                search_templates=[],
            )
        ],
        seeds=["major wars"],
        models={"fast": "m", "smart": "m", "heavy": "m"},
        backend_policy="auto",
        max_cycles=max_cycles,
    )


async def _noop_fact(_entity_id, _claim):
    return None


async def _noop_conflict(_entity_id, _cells):
    return None


async def _make_orch(store, *, lacks_snapshot: bool = False):
    from researcher.backends.resolver import BackendResolver
    from researcher.budget import Budget
    from researcher.scheduler import Scheduler
    from researcher.storage.writer import FactWriter
    from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
    from tests.stubs.llm import StubLLMClient
    from tests.stubs.resolver import StubEntityResolver

    def _which_claude(cmd: str) -> str | None:
        return "/usr/local/bin/claude" if cmd == "claude" else None

    spec = _sample_spec_for_orch()
    bus = StubEventBus()
    llm = StubLLMClient()
    resolver = StubEntityResolver()
    writer = FactWriter(
        store=store,
        resolver=resolver,
        entity_schema={
            "entity_type": "War",
            "fields": [{"name": "name", "type": "str", "required": True}],
        },
        emit_fact=_noop_fact,
        emit_conflict=_noop_conflict,
    )
    scheduler = Scheduler(spec=spec, store=store)
    budget = Budget(usd_cap=3.0, wall_cap_s=600)
    backend_resolver = BackendResolver(which_fn=_which_claude)
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())

    orch = Orchestrator(
        spec=spec,
        store=store,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        bus=bus,  # type: ignore[arg-type]
        writer=writer,
        scheduler=scheduler,
        budget=budget,
        run_id=RUN_ID,
        max_parallel_agents=2,
    )
    orch.set_backend_resolver(backend_resolver)
    orch.set_cli_runner_factory(lambda kind: runner)
    return orch, bus


@pytest.mark.asyncio
async def test_orchestrator_emits_coverage_report_after_cycle_end():
    store = StubKnowledgeStore()
    await store.open()
    orch, bus = await _make_orch(store)

    await orch.run()

    cycle_ends = [
        i for i, e in enumerate(bus.events) if isinstance(e, CycleEnd)
    ]
    coverage = [
        i for i, e in enumerate(bus.events) if isinstance(e, CoverageReport)
    ]
    assert len(cycle_ends) >= 1
    assert len(coverage) >= 1
    # CoverageReport must come AFTER its corresponding CycleEnd.
    assert coverage[0] > cycle_ends[0]
    cov_evt = bus.events[coverage[0]]
    assert isinstance(cov_evt, CoverageReport)
    assert cov_evt.payload.cycle == 1
    assert cov_evt.payload.confidence_threshold == 0.5


class _StoreWithoutSnapshotCoverage(StubKnowledgeStore):
    """Test store that intentionally lacks snapshot_coverage."""

    # Override at the class level so hasattr() returns False.
    snapshot_coverage = None  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_orchestrator_skips_coverage_report_when_store_lacks_method():
    # Build a tiny store that does NOT define snapshot_coverage at all.
    class _BareStore(StubKnowledgeStore):
        pass

    # Surgically remove snapshot_coverage from this instance.
    store = _BareStore()
    await store.open()
    # Make hasattr return False for the instance.
    object.__setattr__(store, "snapshot_coverage", None)

    orch, bus = await _make_orch(store)

    # Patch hasattr indirectly: orchestrator gates on
    # ``hasattr(self._store, "snapshot_coverage")`` AND callable. We
    # additionally guard inside the helper. The simplest way to make this
    # test deterministic is to monkey-patch the helper's gate by removing
    # the attribute from the instance dict and shadowing the bound method.
    # In practice, the orchestrator's hasattr-on-callable check will treat
    # ``None`` as a missing method.
    await orch.run()

    coverage = [e for e in bus.events if isinstance(e, CoverageReport)]
    assert coverage == []


@pytest.mark.asyncio
async def test_orchestrator_uses_custom_confidence_threshold():
    store = StubKnowledgeStore()
    await store.open()

    from researcher.backends.resolver import BackendResolver
    from researcher.budget import Budget
    from researcher.scheduler import Scheduler
    from researcher.storage.writer import FactWriter
    from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
    from tests.stubs.llm import StubLLMClient
    from tests.stubs.resolver import StubEntityResolver

    def _which_claude(cmd: str) -> str | None:
        return "/usr/local/bin/claude" if cmd == "claude" else None

    spec = _sample_spec_for_orch()
    bus = StubEventBus()
    writer = FactWriter(
        store=store,
        resolver=StubEntityResolver(),
        entity_schema={
            "entity_type": "War",
            "fields": [{"name": "name", "type": "str", "required": True}],
        },
        emit_fact=_noop_fact,
        emit_conflict=_noop_conflict,
    )
    scheduler = Scheduler(spec=spec, store=store)
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    orch = Orchestrator(
        spec=spec,
        store=store,  # type: ignore[arg-type]
        llm=StubLLMClient(),  # type: ignore[arg-type]
        bus=bus,  # type: ignore[arg-type]
        writer=writer,
        scheduler=scheduler,
        budget=Budget(usd_cap=3.0, wall_cap_s=600),
        run_id=RUN_ID,
        max_parallel_agents=2,
        coverage_confidence_threshold=0.7,
    )
    orch.set_backend_resolver(BackendResolver(which_fn=_which_claude))
    orch.set_cli_runner_factory(lambda kind: runner)

    await orch.run()
    coverage = [e for e in bus.events if isinstance(e, CoverageReport)]
    assert coverage
    assert coverage[0].payload.confidence_threshold == 0.7
