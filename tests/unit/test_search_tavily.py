"""Tests for TavilyProvider — offline, using httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from researcher.search.tavily import TavilyProvider


def _mock_client(handler) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


async def test_no_api_key_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure no env var bleeds in from the host.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = TavilyProvider(api_key="")
    results = await provider.search("anything")
    assert results == []
    assert provider.name == "tavily"


async def test_successful_response_parses() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        payload = {
            "results": [
                {"title": "T1", "url": "https://one/", "content": "c1"},
                {"title": "T2", "url": "https://two/", "content": "c2"},
            ]
        }
        return httpx.Response(200, json=payload)

    client = _mock_client(handler)
    provider = TavilyProvider(api_key="tv-key-123", client=client)
    results = await provider.search("deep research", max_results=5)
    assert len(results) == 2
    assert results[0].title == "T1"
    assert results[0].url == "https://one/"
    assert results[0].snippet == "c1"
    assert results[0].rank == 0
    assert results[1].rank == 1
    assert captured["body"]["api_key"] == "tv-key-123"
    assert captured["body"]["query"] == "deep research"
    assert captured["body"]["max_results"] == 5
    assert "api.tavily.com" in captured["url"]
    await client.aclose()


async def test_5xx_returns_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    client = _mock_client(handler)
    provider = TavilyProvider(api_key="tv-key", client=client)
    results = await provider.search("q")
    assert results == []
    await client.aclose()
