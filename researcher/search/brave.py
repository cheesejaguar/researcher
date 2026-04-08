"""Brave search provider (https://api.search.brave.com/).

API key from BRAVE_API_KEY env var, sent as X-Subscription-Token header.
Endpoint is GET with ?q=...&count=... query params.
"""

from __future__ import annotations

import os

import httpx

from researcher.search.base import SearchResult


class BraveProvider:
    name = "brave"
    _endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 15.0,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("BRAVE_API_KEY", "")
        self._client = client
        self._timeout_s = timeout_s

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        if not self._api_key:
            return []
        client = self._client or httpx.AsyncClient(timeout=self._timeout_s)
        owns = self._client is None
        try:
            try:
                resp = await client.get(
                    self._endpoint,
                    params={"q": query, "count": max_results},
                    headers={
                        "X-Subscription-Token": self._api_key,
                        "Accept": "application/json",
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
            web_results = (data.get("web") or {}).get("results") or []
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("description", ""),
                    rank=i,
                )
                for i, r in enumerate(web_results)
            ]
        finally:
            if owns:
                await client.aclose()
