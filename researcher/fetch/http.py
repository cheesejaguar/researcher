"""Async HTTP fetch with retries, politeness, and robots respect.

Uses httpx.AsyncClient under the hood. The client is injected so tests
can pass a mock. Fetch order:
  1. Check robots.txt (skip if forbidden)
  2. Acquire politeness limiter for the domain
  3. Make the HTTP request
  4. Retry on 5xx with exponential backoff (up to max_retries attempts)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from researcher.fetch.politeness import PolitenessLimiter
from researcher.fetch.robots import RobotsCache


@dataclass
class FetchResult:
    url: str
    status: int
    content: str
    headers: dict[str, str]
    ok: bool
    error: str | None = None


class HttpFetcher:
    def __init__(
        self,
        politeness: PolitenessLimiter,
        robots: RobotsCache | None = None,
        client: httpx.AsyncClient | None = None,
        user_agent: str = "researcher/0.1 (+https://github.com/cheesejaguar/researcher)",
        max_retries: int = 3,
        timeout_s: float = 20.0,
        backoff_initial_s: float = 0.5,
    ) -> None:
        self._politeness = politeness
        self._robots = robots
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            timeout=timeout_s,
            follow_redirects=True,
        )
        self._max_retries = max_retries
        self._owns_client = client is None
        self._backoff_initial_s = backoff_initial_s

    async def fetch(self, url: str) -> FetchResult:
        # Robots check first — short-circuit before touching the limiter.
        if self._robots is not None:
            allowed = await self._robots.allowed(url)
            if not allowed:
                return FetchResult(
                    url=url,
                    status=0,
                    content="",
                    headers={},
                    ok=False,
                    error="robots_disallowed",
                )

        async with self._politeness.acquire(url):
            backoff = self._backoff_initial_s
            last_error: str | None = None
            last_status = 0
            for attempt in range(self._max_retries):
                try:
                    resp = await self._client.get(url)
                    last_status = resp.status_code
                    if 500 <= resp.status_code < 600:
                        last_error = f"http {resp.status_code}"
                        if attempt + 1 < self._max_retries:
                            await asyncio.sleep(backoff)
                            backoff *= 2
                            continue
                        # Exhausted retries on 5xx — return the last response.
                        return FetchResult(
                            url=str(resp.url),
                            status=resp.status_code,
                            content=resp.text,
                            headers=dict(resp.headers),
                            ok=False,
                            error=last_error,
                        )
                    # Any non-5xx is terminal: return whatever we got.
                    ok = 200 <= resp.status_code < 400
                    return FetchResult(
                        url=str(resp.url),
                        status=resp.status_code,
                        content=resp.text,
                        headers=dict(resp.headers),
                        ok=ok,
                        error=None if ok else f"http {resp.status_code}",
                    )
                except httpx.HTTPError as e:
                    last_error = f"httpx: {e}"
                    if attempt + 1 < self._max_retries:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    break
            return FetchResult(
                url=url,
                status=last_status,
                content="",
                headers={},
                ok=False,
                error=last_error or "unknown",
            )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
