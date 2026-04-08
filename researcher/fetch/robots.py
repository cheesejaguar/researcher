"""robots.txt cache that respects disallow rules per domain.

In-memory cache keyed on (scheme, netloc). Fetches robots.txt once
per domain using an injected fetcher (so tests don't hit the network).
Returns True iff the user-agent is allowed to fetch the given URL.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

Fetcher = Callable[[str], Awaitable[str]]


class RobotsCache:
    def __init__(self, fetcher: Fetcher, user_agent: str = "researcher/0.1") -> None:
        self._fetcher = fetcher
        self._user_agent = user_agent
        self._cache: dict[str, RobotFileParser | None] = {}

    async def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        key = f"{parsed.scheme}://{parsed.netloc}"
        if key not in self._cache:
            robots_url = f"{key}/robots.txt"
            try:
                content = await self._fetcher(robots_url)
                rp: RobotFileParser | None = RobotFileParser()
                assert rp is not None  # for type checker
                rp.parse(content.splitlines())
            except Exception:
                rp = None  # treat as allow-all if robots.txt unreachable
            self._cache[key] = rp
        rp = self._cache[key]
        if rp is None:
            return True
        try:
            return rp.can_fetch(self._user_agent, url)
        except Exception:
            return True
