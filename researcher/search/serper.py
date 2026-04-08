"""Serper.dev search provider (https://serper.dev/).

API key from SERPER_API_KEY env var, sent as X-API-KEY header.
POSTs JSON {"q": query, "num": max_results} to /search.
"""

from __future__ import annotations

import os

import httpx

from researcher.search.base import SearchResult


class SerperProvider:
    name = "serper"
    _endpoint = "https://google.serper.dev/search"

    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 15.0,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("SERPER_API_KEY", "")
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
                    headers={
                        "X-API-KEY": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": max_results},
                )
            except httpx.HTTPError:
                return []
            if resp.status_code != 200:
                return []
            try:
                data = resp.json()
            except Exception:
                return []
            organic = data.get("organic") or []
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                    rank=i,
                )
                for i, r in enumerate(organic)
            ]
        finally:
            if owns:
                await client.aclose()
