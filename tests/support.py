"""Shared test wiring built from the project's real components.

Only true external boundaries (LLM providers, paid search APIs, the
network) are faked in the test suite. Everything else — the DuckDB
knowledge store, the event bus, the entity resolver, the HTTP fetcher —
is the production implementation, pointed at temp dirs and in-process
transports.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import httpx

from researcher.cli import _default_embed_fn
from researcher.events import Event, EventBus
from researcher.fetch.http import HttpFetcher
from researcher.fetch.politeness import PolitenessLimiter
from researcher.storage.resolver import DefaultEntityResolver
from researcher.storage.store import KnowledgeStore


class RecordingEventBus(EventBus):
    """The real :class:`EventBus` (JSONL + socket queue) plus an in-memory
    list of every emitted event, captured through the public subscriber hook.
    """

    def __init__(self, jsonl_path: Path | str) -> None:
        super().__init__(jsonl_path=jsonl_path)
        self.events: list[Event] = []
        self.add_subscriber(self.events.append)


def make_resolver(store: KnowledgeStore) -> DefaultEntityResolver:
    """Real embedding resolver backed by ``store``, using the CLI's offline
    deterministic hash embedder (no model download)."""
    return DefaultEntityResolver(store=store, embed_fn=_default_embed_fn)


def mock_http(
    pages: Mapping[str, tuple[int, str]] | Callable[[httpx.Request], httpx.Response],
) -> HttpFetcher:
    """Real :class:`HttpFetcher` whose network transport is an in-process
    ``httpx.MockTransport``.

    ``pages`` maps URL -> (status, html body); unknown URLs return 404.
    A callable handler can be passed for full control.
    """
    if callable(pages):
        handler = pages
    else:
        table = dict(pages)

        def handler(request: httpx.Request) -> httpx.Response:
            status, body = table.get(str(request.url), (404, ""))
            return httpx.Response(status, text=body, headers={"content-type": "text/html"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpFetcher(
        # Real per-host limiter, but fast enough not to slow the suite down.
        politeness=PolitenessLimiter(default_rps=1000.0),
        client=client,
        max_retries=1,
        backoff_initial_s=0.0,
    )
