"""Tests for RobotsCache — offline, no real network."""

from __future__ import annotations

from researcher.fetch.robots import RobotsCache


async def test_allow_all_when_robots_is_unreachable() -> None:
    async def fetcher(url: str) -> str:
        raise RuntimeError("network down")

    cache = RobotsCache(fetcher)
    assert await cache.allowed("https://unreachable.example/path") is True


async def test_respects_disallow_rule() -> None:
    robots_txt = "User-agent: *\nDisallow: /private/\n"

    async def fetcher(url: str) -> str:
        assert url == "https://foo.example/robots.txt"
        return robots_txt

    cache = RobotsCache(fetcher)
    assert await cache.allowed("https://foo.example/public/page") is True
    assert await cache.allowed("https://foo.example/private/secret") is False


async def test_caches_per_domain_only_one_fetch() -> None:
    calls: list[str] = []

    async def fetcher(url: str) -> str:
        calls.append(url)
        return "User-agent: *\nAllow: /\n"

    cache = RobotsCache(fetcher)
    await cache.allowed("https://cached.example/a")
    await cache.allowed("https://cached.example/b")
    await cache.allowed("https://cached.example/c")
    # Only one robots.txt fetch for the same domain.
    assert calls == ["https://cached.example/robots.txt"]


async def test_malformed_robots_treated_as_allow_all() -> None:
    async def fetcher(url: str) -> str:
        return "this is not valid robots syntax \x00\x01"

    cache = RobotsCache(fetcher)
    # Malformed content should not blow up; allow-all is the safe default.
    assert await cache.allowed("https://malformed.example/page") is True


async def test_rejects_url_without_scheme() -> None:
    async def fetcher(url: str) -> str:
        raise AssertionError("fetcher should not be called")

    cache = RobotsCache(fetcher)
    assert await cache.allowed("not-a-url") is False
