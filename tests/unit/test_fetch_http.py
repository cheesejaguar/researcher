"""Tests for HttpFetcher — offline, using httpx.MockTransport."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from researcher.fetch.http import FetchResult, HttpFetcher
from researcher.fetch.politeness import PolitenessLimiter
from researcher.fetch.robots import RobotsCache


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, follow_redirects=True)


async def test_fetch_success_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="hello", headers={"content-type": "text/plain"})

    client = _mock_client(handler)
    fetcher = HttpFetcher(
        politeness=PolitenessLimiter(default_rps=100.0),
        robots=None,
        client=client,
    )
    result = await fetcher.fetch("https://ok.example/page")
    assert isinstance(result, FetchResult)
    assert result.ok is True
    assert result.status == 200
    assert result.content == "hello"
    assert result.headers["content-type"] == "text/plain"
    assert result.error is None
    await client.aclose()


async def test_fetch_retries_on_5xx_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text="ok-now")

    client = _mock_client(handler)
    fetcher = HttpFetcher(
        politeness=PolitenessLimiter(default_rps=100.0),
        client=client,
        max_retries=3,
    )
    result = await fetcher.fetch("https://flaky.example/resource")
    assert calls["n"] == 3
    assert result.ok is True
    assert result.status == 200
    assert result.content == "ok-now"
    await client.aclose()


async def test_fetch_permanent_5xx_returns_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _mock_client(handler)
    fetcher = HttpFetcher(
        politeness=PolitenessLimiter(default_rps=100.0),
        client=client,
        max_retries=2,
    )
    result = await fetcher.fetch("https://broken.example/x")
    assert result.ok is False
    # The last attempt should report status 500 back from the response.
    assert result.status == 500
    assert result.error is not None
    await client.aclose()


async def test_fetch_4xx_does_not_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="not found")

    client = _mock_client(handler)
    fetcher = HttpFetcher(
        politeness=PolitenessLimiter(default_rps=100.0),
        client=client,
        max_retries=3,
    )
    result = await fetcher.fetch("https://missing.example/gone")
    assert calls["n"] == 1, "4xx must not retry"
    assert result.ok is False
    assert result.status == 404
    await client.aclose()


async def test_fetch_robots_disallowed_short_circuits() -> None:
    hit = {"http": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hit["http"] += 1
        return httpx.Response(200, text="should not be fetched")

    async def robots_fetcher(url: str) -> str:
        return "User-agent: *\nDisallow: /\n"

    client = _mock_client(handler)
    fetcher = HttpFetcher(
        politeness=PolitenessLimiter(default_rps=100.0),
        robots=RobotsCache(robots_fetcher),
        client=client,
    )
    result = await fetcher.fetch("https://blocked.example/secret")
    assert result.ok is False
    assert result.error == "robots_disallowed"
    assert hit["http"] == 0, "must not make HTTP call when robots disallows"
    await client.aclose()


async def test_fetch_acquires_politeness_before_http() -> None:
    """Ordering proof: the politeness limiter is entered before the request fires."""
    order: list[str] = []

    class SpyLimiter(PolitenessLimiter):
        async def acquire(self, url: str):  # type: ignore[override]
            order.append("politeness")
            async with super().acquire(url):
                yield

    # SpyLimiter.acquire is an async generator — wrap in asynccontextmanager.
    from contextlib import asynccontextmanager

    class SpyLimiter2(PolitenessLimiter):
        @asynccontextmanager
        async def acquire(self, url: str):  # type: ignore[override]
            order.append("politeness")
            async with super().acquire(url):
                yield

    def handler(request: httpx.Request) -> httpx.Response:
        order.append("http")
        return httpx.Response(200, text="ok")

    client = _mock_client(handler)
    fetcher = HttpFetcher(
        politeness=SpyLimiter2(default_rps=100.0),
        client=client,
    )
    result = await fetcher.fetch("https://order.example/p")
    assert result.ok is True
    assert order == ["politeness", "http"]
    await client.aclose()
