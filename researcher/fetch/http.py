"""Async HTTP fetch with retries, politeness, robots, and SSRF policy.

Uses httpx.AsyncClient under the hood. The client is injected so tests
can pass a mock. Fetch order:
  1. Validate scheme/host/source policy (and DNS unless a test client is used)
  2. Check robots.txt (skip if forbidden)
  3. Acquire politeness limiter for the domain
  4. Make the HTTP request with manual redirects, revalidating every hop
  5. Retry on 5xx with exponential backoff (up to max_retries attempts)
  6. Cap the response body size
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import socket
from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urljoin, urlparse

import httpx

from researcher.fetch.politeness import PolitenessLimiter
from researcher.fetch.robots import RobotsCache
from researcher.sources import domain_allowed
from researcher.spec import SourcePolicy

ResolveHost = Callable[[str], Awaitable[list[str]] | list[str]]


@dataclass
class FetchResult:
    url: str
    status: int
    content: str
    headers: dict[str, str]
    ok: bool
    error: str | None = None
    content_type: str = ""  # raw content-type header, lowercased, no params
    raw_bytes: bytes | None = None  # populated only for binary content (e.g. PDF)


def _bare_content_type(resp: httpx.Response) -> str:
    return resp.headers.get("content-type", "").split(";")[0].strip().lower()


def _is_pdf(url: str, content_type: str) -> bool:
    if content_type == "application/pdf":
        return True
    return url.lower().split("?", 1)[0].rstrip("/").endswith(".pdf")


def _is_public_ip(raw: str) -> bool:
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


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
        max_bytes: int = 10_000_000,
        max_redirects: int = 5,
        source_policy: SourcePolicy | None = None,
        legacy_allowlist: list[str] | None = None,
        resolve_host: ResolveHost | None = None,
    ) -> None:
        self._politeness = politeness
        self._robots = robots
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            timeout=timeout_s,
            follow_redirects=False,
        )
        self._max_retries = max_retries
        self._owns_client = client is None
        self._backoff_initial_s = backoff_initial_s
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects
        self._source_policy = source_policy
        self._legacy_allowlist = list(legacy_allowlist or [])
        self._resolve_host = resolve_host

    async def fetch(self, url: str) -> FetchResult:
        validation_error = self._validate_url_shape(url)
        if validation_error is not None:
            return self._error(url, validation_error)
        # DNS validation is only meaningful when this fetcher owns the real
        # transport (or a resolver is explicitly injected). Test transports
        # and offline fixtures use synthetic hostnames by design.
        if self._owns_client or self._resolve_host is not None:
            dns_error = await self._validate_host(url)
            if dns_error is not None:
                return self._error(url, dns_error)

        # Robots check first — short-circuit before touching the limiter.
        if self._robots is not None:
            allowed = await self._robots.allowed(url)
            if not allowed:
                return self._error(url, "robots_disallowed")

        async with self._politeness.acquire(url):
            backoff = self._backoff_initial_s
            last_error: str | None = None
            last_status = 0
            for attempt in range(self._max_retries):
                current_url = url
                try:
                    for _ in range(self._max_redirects + 1):
                        redirect_target: str | None = None
                        async with self._client.stream(
                            "GET", current_url, follow_redirects=False
                        ) as resp:
                            last_status = resp.status_code
                            content_type = _bare_content_type(resp)
                            resp_url = str(resp.url)
                            if 300 <= resp.status_code < 400:
                                location = resp.headers.get("location", "")
                                if not location:
                                    return self._response_result(
                                        resp_url,
                                        resp.status_code,
                                        b"",
                                        dict(resp.headers),
                                        content_type,
                                        error="redirect_without_location",
                                    )
                                redirect_target = urljoin(resp_url, location)
                                policy_error = self._validate_url_shape(redirect_target)
                                if policy_error is None and (
                                    self._owns_client or self._resolve_host is not None
                                ):
                                    policy_error = await self._validate_host(redirect_target)
                                if policy_error is not None:
                                    return self._error(redirect_target, policy_error)
                            else:
                                body = await self._read_limited(resp)
                                if body is None:
                                    return self._error(
                                        resp_url,
                                        "response_too_large",
                                        status=resp.status_code,
                                    )
                                is_retryable = 500 <= resp.status_code < 600
                                if is_retryable:
                                    last_error = f"http {resp.status_code}"
                                else:
                                    return self._response_result(
                                        resp_url,
                                        resp.status_code,
                                        body,
                                        dict(resp.headers),
                                        content_type,
                                    )
                        if redirect_target is not None:
                            current_url = redirect_target
                            continue
                        break
                    if last_error is not None and attempt + 1 < self._max_retries:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    if last_error is not None:
                        return self._error(current_url, last_error, status=last_status)
                except httpx.HTTPError as e:
                    last_error = f"httpx: {e}"
                    if attempt + 1 < self._max_retries:
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    break
            return self._error(url, last_error or "unknown", status=last_status)

    def _validate_url_shape(self, url: str) -> str | None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return "blocked_scheme"
        if not parsed.hostname:
            return "blocked_host"
        if parsed.username or parsed.password:
            return "blocked_credentials"
        if self._source_policy is not None and not domain_allowed(
            url, self._source_policy, self._legacy_allowlist
        ):
            return "source_policy_blocked"
        return None

    async def _validate_host(self, url: str) -> str | None:
        host = urlparse(url).hostname or ""
        try:
            addresses = await self._resolve_addresses(host)
        except OSError:
            return "dns_resolution_failed"
        if not addresses:
            return "dns_resolution_failed"
        if any(not _is_public_ip(addr) for addr in addresses):
            return "blocked_private_address"
        return None

    async def _resolve_addresses(self, host: str) -> list[str]:
        if self._resolve_host is not None:
            result = self._resolve_host(host)
            if inspect.isawaitable(result):
                result = await result
            return [str(item) for item in result]
        infos = await asyncio.to_thread(socket.getaddrinfo, host, 443, type=socket.SOCK_STREAM)
        return sorted({str(info[4][0]) for info in infos})

    async def _read_limited(self, resp: httpx.Response) -> bytes | None:
        declared = resp.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self._max_bytes:
                    return None
            except ValueError:
                pass
        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > self._max_bytes:
                return None
            chunks.append(chunk)
        return b"".join(chunks)

    def _response_result(
        self,
        url: str,
        status: int,
        body: bytes,
        headers: dict[str, str],
        content_type: str,
        *,
        error: str | None = None,
    ) -> FetchResult:
        is_pdf = _is_pdf(url, content_type)
        ok = 200 <= status < 400 and error is None
        encoding = "utf-8"
        if not is_pdf:
            try:
                encoding = httpx.Response(status_code=status, headers=headers).encoding or "utf-8"
            except Exception:
                pass
        return FetchResult(
            url=url,
            status=status,
            content="" if is_pdf else body.decode(encoding, errors="replace"),
            headers=headers,
            ok=ok,
            error=error if error is not None else (None if ok else f"http {status}"),
            content_type=content_type,
            raw_bytes=body if is_pdf else None,
        )

    @staticmethod
    def _error(url: str, error: str, status: int = 0) -> FetchResult:
        return FetchResult(
            url=url,
            status=status,
            content="",
            headers={},
            ok=False,
            error=error,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
