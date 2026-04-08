"""Tavily search provider (https://tavily.com/).

API key from TAVILY_API_KEY env var. Uses httpx for the POST request.
"""

from __future__ import annotations

import os

import httpx

from researcher.search.base import SearchResult


class TavilyProvider:
    name = "tavily"
    _endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 15.0,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("TAVILY_API_KEY", "")
        self._client = client
        self._timeout_s = timeout_s

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        if not self._api_key:
            return []
        client = self._client or httpx.AsyncClient(timeout=self._timeout_s)
        owns = self._client is None
        try:
            try:
                resp = await client.post(
                    self._endpoint,
                    json={
                        "api_key": self._api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                    },
                )
            except httpx.HTTPError:
                return []
            if resp.status_code != 200:
                return []
            try:
                data = resp.json()
            except Exception:
                return []
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    rank=i,
                )
                for i, r in enumerate(data.get("results", []))
            ]
        finally:
            if owns:
                await client.aclose()
