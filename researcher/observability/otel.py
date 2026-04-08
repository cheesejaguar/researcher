"""OTLP trace export adapter for the researcher EventBus (v1.3 #4).

The JSONL-backed :class:`researcher.events.EventBus` is authoritative for
replay but not interoperable with standard observability tooling. This module
ships an :class:`OtelEventAdapter` that subscribes to the event stream and
translates each event into a structured span in a hierarchical trace:

    run                       (opened via :meth:`OtelEventAdapter.start_run`)
    +--- cycle                (CycleStart -> CycleEnd)
          +--- agent.<kind>   (AgentSpawn -> AgentStateChange done/failed)
                +--- fact     (FactWritten, only when include_facts=True)

The module deliberately does NOT import the opentelemetry SDK at the top
level. A thin :class:`SpanExporter` Protocol is the only contract the
adapter relies on; tests inject an in-memory stub and production wiring
uses :func:`build_otlp_exporter` — which soft-imports the SDK and raises
a clear :class:`RuntimeError` if the optional dependency is missing. That
keeps ``uv sync`` free of OTEL deps by default and lets CLI wiring
fail-open when the extras are not installed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from researcher.events import (
    AgentLog,
    AgentSpawn,
    AgentStateChange,
    CycleEnd,
    CycleStart,
    FactWritten,
    RunComplete,
)

# ---------- SpanExporter Protocol ----------


@runtime_checkable
class SpanExporter(Protocol):
    """Thin Protocol so the OTEL SDK stays an optional dep.

    Tests inject an in-memory stub implementing this shape; production
    wiring passes in an OTLP gRPC/HTTP exporter via
    :func:`build_otlp_exporter`.
    """

    def start_span(
        self,
        name: str,
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        start_time_unix_nano: int,
        attributes: dict[str, Any],
    ) -> None: ...

    def end_span(
        self,
        *,
        span_id: str,
        end_time_unix_nano: int,
        status: str = "ok",  # "ok" or "error"
        attributes: dict[str, Any] | None = None,
    ) -> None: ...


# ---------- Internal open-span bookkeeping ----------


@dataclass
class _OpenSpan:
    span_id: str
    parent_span_id: str | None
    name: str
    start_time_unix_nano: int
    attributes: dict[str, Any] = field(default_factory=dict)


# ---------- Helpers ----------


def _ns_from_dt(dt: datetime | None) -> int:
    if dt is None:
        return int(datetime.now().timestamp() * 1_000_000_000)
    return int(dt.timestamp() * 1_000_000_000)


def _trace_id_from_run_id(run_id: str) -> str:
    """Derive a stable 32-hex-char trace_id from a run_id.

    Uses a SHA-256 prefix so the same run_id always yields the same
    trace_id across adapters, replays, and live tails.
    """
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return digest[:32]


# ---------- Adapter ----------


class OtelEventAdapter:
    """Subscribes to researcher Events and emits spans on a SpanExporter.

    Mapping:
      CycleStart     -> start 'cycle' span (parent: run span)
      CycleEnd       -> end   'cycle' span
      AgentSpawn     -> start 'agent.<kind>' span (parent: cycle span)
      AgentStateChange (-> 'done' or 'failed') -> end the agent span
      AgentLog       -> attribute append on the agent span
      FactWritten    -> child 'fact' span if include_facts=True (start==end)
      RunComplete    -> end the 'run' span (final)

    The run span itself is opened via :meth:`start_run` so the caller
    controls the wall-clock boundary; unknown event types are ignored.
    """

    def __init__(
        self,
        exporter: SpanExporter,
        *,
        run_id: str,
        service_name: str = "researcher",
        include_facts: bool = False,
    ) -> None:
        self._exporter = exporter
        self._run_id = run_id
        self._service_name = service_name
        self._include_facts = include_facts

        self._trace_id = _trace_id_from_run_id(run_id)
        self._next_id = 1  # monotonic span-id counter

        self._run_span: _OpenSpan | None = None
        self._cycle_spans: dict[int, _OpenSpan] = {}
        self._agent_spans: dict[str, _OpenSpan] = {}

    # ---- id helpers ----

    def _next_span_id(self) -> str:
        """Return a fresh 16-hex-char span id (monotonic counter)."""
        value = self._next_id
        self._next_id += 1
        return f"{value:016x}"

    # ---- lifecycle ----

    def start_run(self, started_at_ns: int | None = None) -> None:
        """Open the root 'run' span.

        Safe to call once per adapter; subsequent calls are no-ops so
        observers that wire ``start_run`` defensively won't double-open.
        """
        if self._run_span is not None:
            return
        start_ns = started_at_ns if started_at_ns is not None else _ns_from_dt(None)
        span_id = self._next_span_id()
        attrs: dict[str, Any] = {
            "service.name": self._service_name,
            "researcher.run_id": self._run_id,
        }
        self._run_span = _OpenSpan(
            span_id=span_id,
            parent_span_id=None,
            name="run",
            start_time_unix_nano=start_ns,
            attributes=attrs,
        )
        self._exporter.start_span(
            "run",
            trace_id=self._trace_id,
            span_id=span_id,
            parent_span_id=None,
            start_time_unix_nano=start_ns,
            attributes=attrs,
        )

    def flush(self) -> None:
        """End any still-open spans with status='error' + reason='flush'.

        Ordering is inside-out so children close before parents: fact
        spans are always point spans (start==end), agents close, then
        cycles, then the run span itself.
        """
        flush_ns = _ns_from_dt(None)
        # Agents first.
        for agent_id, span in list(self._agent_spans.items()):
            self._exporter.end_span(
                span_id=span.span_id,
                end_time_unix_nano=flush_ns,
                status="error",
                attributes={**span.attributes, "reason": "flush"},
            )
            del self._agent_spans[agent_id]
        # Cycles.
        for cycle, span in list(self._cycle_spans.items()):
            self._exporter.end_span(
                span_id=span.span_id,
                end_time_unix_nano=flush_ns,
                status="error",
                attributes={**span.attributes, "reason": "flush"},
            )
            del self._cycle_spans[cycle]
        # Run span.
        if self._run_span is not None:
            self._exporter.end_span(
                span_id=self._run_span.span_id,
                end_time_unix_nano=flush_ns,
                status="error",
                attributes={**self._run_span.attributes, "reason": "flush"},
            )
            self._run_span = None

    # ---- dispatch ----

    def handle(self, event: Any) -> None:
        """Dispatch a researcher event into span lifecycle operations.

        Unknown / non-Event objects are silently ignored so an over-eager
        subscriber never crashes the core emit path.
        """
        if isinstance(event, CycleStart):
            self._handle_cycle_start(event)
        elif isinstance(event, CycleEnd):
            self._handle_cycle_end(event)
        elif isinstance(event, AgentSpawn):
            self._handle_agent_spawn(event)
        elif isinstance(event, AgentStateChange):
            self._handle_agent_state_change(event)
        elif isinstance(event, AgentLog):
            self._handle_agent_log(event)
        elif isinstance(event, FactWritten):
            self._handle_fact_written(event)
        elif isinstance(event, RunComplete):
            self._handle_run_complete(event)
        # Unknown types are intentionally ignored.

    # ---- handlers ----

    def _handle_cycle_start(self, event: CycleStart) -> None:
        if self._run_span is None:
            # Defensive: start the run span lazily if the caller forgot.
            self.start_run(_ns_from_dt(event.ts))
        assert self._run_span is not None
        cycle = event.payload.cycle
        span_id = self._next_span_id()
        start_ns = _ns_from_dt(event.ts)
        attrs: dict[str, Any] = {
            "researcher.cycle": cycle,
            "researcher.pending_tasks": event.payload.pending_tasks,
        }
        span = _OpenSpan(
            span_id=span_id,
            parent_span_id=self._run_span.span_id,
            name="cycle",
            start_time_unix_nano=start_ns,
            attributes=attrs,
        )
        self._cycle_spans[cycle] = span
        self._exporter.start_span(
            "cycle",
            trace_id=self._trace_id,
            span_id=span_id,
            parent_span_id=self._run_span.span_id,
            start_time_unix_nano=start_ns,
            attributes=attrs,
        )

    def _handle_cycle_end(self, event: CycleEnd) -> None:
        cycle = event.payload.cycle
        span = self._cycle_spans.pop(cycle, None)
        if span is None:
            return
        end_ns = _ns_from_dt(event.ts)
        end_attrs = {
            **span.attributes,
            "researcher.claims_written": event.payload.claims_written,
            "researcher.entities_total": event.payload.entities_total,
            "researcher.cost_usd": event.payload.cost_usd,
        }
        self._exporter.end_span(
            span_id=span.span_id,
            end_time_unix_nano=end_ns,
            status="ok",
            attributes=end_attrs,
        )

    def _handle_agent_spawn(self, event: AgentSpawn) -> None:
        # Parent to the most recent open cycle span if any; otherwise the
        # run span; otherwise silently skip (no trace context).
        parent_id: str | None
        if self._cycle_spans:
            # Highest cycle number wins — most recent cycle_start.
            last_cycle = max(self._cycle_spans.keys())
            parent_id = self._cycle_spans[last_cycle].span_id
        elif self._run_span is not None:
            parent_id = self._run_span.span_id
        else:
            return

        kind = event.payload.kind
        agent_id = event.payload.agent_id
        name = f"agent.{kind}"
        span_id = self._next_span_id()
        start_ns = _ns_from_dt(event.ts)
        attrs: dict[str, Any] = {
            "researcher.agent_id": agent_id,
            "researcher.task_id": event.payload.task_id,
            "researcher.agent_kind": kind,
        }
        span = _OpenSpan(
            span_id=span_id,
            parent_span_id=parent_id,
            name=name,
            start_time_unix_nano=start_ns,
            attributes=attrs,
        )
        self._agent_spans[agent_id] = span
        self._exporter.start_span(
            name,
            trace_id=self._trace_id,
            span_id=span_id,
            parent_span_id=parent_id,
            start_time_unix_nano=start_ns,
            attributes=attrs,
        )

    def _handle_agent_state_change(self, event: AgentStateChange) -> None:
        new_state = event.payload.new
        if new_state not in ("done", "failed"):
            return
        agent_id = event.payload.agent_id
        span = self._agent_spans.pop(agent_id, None)
        if span is None:
            return
        end_ns = _ns_from_dt(event.ts)
        status = "ok" if new_state == "done" else "error"
        end_attrs = {
            **span.attributes,
            "researcher.final_state": new_state,
        }
        self._exporter.end_span(
            span_id=span.span_id,
            end_time_unix_nano=end_ns,
            status=status,
            attributes=end_attrs,
        )

    def _handle_agent_log(self, event: AgentLog) -> None:
        agent_id = event.payload.agent_id
        span = self._agent_spans.get(agent_id)
        if span is None:
            return
        logs = span.attributes.setdefault("logs", [])
        logs.append(
            {
                "level": event.payload.level,
                "msg": event.payload.msg,
            }
        )

    def _handle_fact_written(self, event: FactWritten) -> None:
        if not self._include_facts:
            return
        # Parent to the deepest open span: agent > cycle > run.
        parent_id: str | None
        if self._agent_spans:
            # Pick any open agent; there's typically one at a time per cycle.
            parent_id = next(iter(self._agent_spans.values())).span_id
        elif self._cycle_spans:
            last_cycle = max(self._cycle_spans.keys())
            parent_id = self._cycle_spans[last_cycle].span_id
        elif self._run_span is not None:
            parent_id = self._run_span.span_id
        else:
            return

        ts_ns = _ns_from_dt(event.ts)
        span_id = self._next_span_id()
        attrs: dict[str, Any] = {
            "researcher.entity_id": event.payload.entity_id,
            "researcher.entity_type": event.payload.entity_type,
            "researcher.field": event.payload.field,
            "researcher.confidence": event.payload.confidence,
            "researcher.source_url": event.payload.source_url,
        }
        self._exporter.start_span(
            "fact",
            trace_id=self._trace_id,
            span_id=span_id,
            parent_span_id=parent_id,
            start_time_unix_nano=ts_ns,
            attributes=attrs,
        )
        self._exporter.end_span(
            span_id=span_id,
            end_time_unix_nano=ts_ns,
            status="ok",
            attributes=attrs,
        )

    def _handle_run_complete(self, event: RunComplete) -> None:
        if self._run_span is None:
            return
        end_ns = _ns_from_dt(event.ts)
        end_attrs = {
            **self._run_span.attributes,
            "researcher.reason": event.payload.reason,
            "researcher.entities": event.payload.entities,
            "researcher.cost_usd": event.payload.cost_usd,
            "researcher.wall_s": event.payload.wall_s,
        }
        self._exporter.end_span(
            span_id=self._run_span.span_id,
            end_time_unix_nano=end_ns,
            status="ok",
            attributes=end_attrs,
        )
        self._run_span = None


# ---------- Real OTLP exporter (optional) ----------


def build_otlp_exporter(endpoint: str) -> SpanExporter:
    """Build a real OTLP exporter. Requires opentelemetry-sdk to be installed.

    Raises :class:`RuntimeError` with a clear install hint if the optional
    ``otel`` extra has not been installed. The returned object satisfies
    the :class:`SpanExporter` Protocol via a thin wrapper that drives the
    SDK's ``Tracer.start_span`` + span ``end`` primitives.

    Parameters
    ----------
    endpoint:
        OTLP collector endpoint, e.g. ``"localhost:4317"``.
    """
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
            BatchSpanProcessor,
        )
    except ImportError as exc:  # pragma: no cover — exercised via monkeypatch test
        raise RuntimeError(
            "opentelemetry-sdk not installed — install with 'uv sync --extra otel'"
        ) from exc

    resource = Resource.create({"service.name": "researcher"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )
    tracer = provider.get_tracer("researcher.observability")

    class _SdkExporter:
        """Thin wrapper mapping SpanExporter Protocol to the OTEL SDK Tracer."""

        def __init__(self) -> None:
            self._open: dict[str, Any] = {}

        def start_span(
            self,
            name: str,
            *,
            trace_id: str,
            span_id: str,
            parent_span_id: str | None,
            start_time_unix_nano: int,
            attributes: dict[str, Any],
        ) -> None:
            # The real SDK manages trace/span IDs itself; we key our own
            # dict by the adapter's logical span_id and let the SDK attach
            # its own IDs. Parent linking via the adapter's ID scheme is
            # best-effort — use the SDK's current span context where we can.
            sdk_span = tracer.start_span(name, attributes=attributes)
            sdk_span._adapter_start_ns = start_time_unix_nano  # type: ignore[attr-defined]
            self._open[span_id] = sdk_span

        def end_span(
            self,
            *,
            span_id: str,
            end_time_unix_nano: int,
            status: str = "ok",
            attributes: dict[str, Any] | None = None,
        ) -> None:
            sdk_span = self._open.pop(span_id, None)
            if sdk_span is None:
                return
            if attributes:
                for k, v in attributes.items():
                    try:
                        sdk_span.set_attribute(k, v)
                    except Exception:
                        pass
            try:
                from opentelemetry.trace import (  # type: ignore[import-not-found]
                    Status,
                    StatusCode,
                )

                sdk_span.set_status(
                    Status(StatusCode.ERROR if status == "error" else StatusCode.OK)
                )
            except Exception:
                pass
            sdk_span.end(end_time=end_time_unix_nano)

    return _SdkExporter()
