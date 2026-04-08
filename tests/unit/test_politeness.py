"""Tests for PolitenessLimiter — per-domain rate limiting."""

from __future__ import annotations

import asyncio
import time

from researcher.fetch.politeness import PolitenessLimiter


async def test_acquire_yields_without_error() -> None:
    limiter = PolitenessLimiter(default_rps=100.0)
    async with limiter.acquire("https://example.com/page"):
        pass  # just prove the context manager yields


async def test_per_domain_isolation_runs_in_parallel() -> None:
    # Two different domains should NOT serialize each other even at slow rps.
    limiter = PolitenessLimiter(default_rps=1.0)  # 1 req / sec

    async def hit(url: str) -> float:
        async with limiter.acquire(url):
            return time.monotonic()

    start = time.monotonic()
    t1, t2 = await asyncio.gather(
        hit("https://foo.example/"),
        hit("https://bar.example/"),
    )
    elapsed = max(t1, t2) - start
    # Two different domains => both should acquire near-instantly.
    assert elapsed < 0.5


async def test_same_domain_serializes_burst() -> None:
    # Two requests to the same domain at 5 rps should take at least ~0.2s apart.
    limiter = PolitenessLimiter(default_rps=5.0)

    async def hit() -> float:
        async with limiter.acquire("https://same.example/page"):
            return time.monotonic()

    t0 = await hit()
    t1 = await hit()
    delta = t1 - t0
    # The second hit must be rate-limited by the same-domain limiter.
    assert delta >= 0.15, f"expected >=0.15s gap, got {delta:.3f}s"
