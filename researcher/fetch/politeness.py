"""Per-domain politeness limiter built on aiolimiter.

Enforces a max request rate per domain to avoid hammering origins.
Used by fetch.http.fetch() as the outer-most gate BEFORE the LLM
semaphore to avoid deadlock (see main plan architectural decision 3).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from aiolimiter import AsyncLimiter


class PolitenessLimiter:
    """Per-domain rate limiter. Default 0.5 req/sec = 1 req every 2 seconds."""

    def __init__(self, default_rps: float = 0.5, burst: int = 1) -> None:
        self._default_rps = default_rps
        self._burst = burst
        self._per_domain: dict[str, AsyncLimiter] = {}

    def _domain_for(self, url: str) -> str:
        return urlparse(url).netloc.lower()

    def _get(self, domain: str) -> AsyncLimiter:
        limiter = self._per_domain.get(domain)
        if limiter is None:
            # AsyncLimiter(max_rate=burst, time_period=burst/rps)
            limiter = AsyncLimiter(self._burst, self._burst / self._default_rps)
            self._per_domain[domain] = limiter
        return limiter

    @asynccontextmanager
    async def acquire(self, url: str) -> AsyncIterator[None]:
        domain = self._domain_for(url)
        limiter = self._get(domain)
        async with limiter:
            yield
