"""Source-pack ingestion, search, and source-policy helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from researcher.extract.chunker import chunk_text
from researcher.search.base import SearchProvider, SearchResult
from researcher.spec import SourcePolicy, SourceSet


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    title: str
    url: str
    source_type: str
    path: str
    metadata: dict


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    source_id: str
    ordinal: int
    text: str


def domain_for_url(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.netloc or "").lower()


def domain_allowed(url: str, policy: SourcePolicy, legacy_allowlist: list[str] | None = None) -> bool:
    domain = domain_for_url(url)
    if not domain:
        return True
    deny = {d.lower() for d in policy.deny_domains}
    allow = {d.lower() for d in policy.allow_domains}
    allow.update(d.lower() for d in (legacy_allowlist or []))
    trusted = {d.lower() for d in policy.trusted_domains}
    if any(domain == d or domain.endswith(f".{d}") for d in deny):
        return False
    if policy.require_trusted and trusted:
        return any(domain == d or domain.endswith(f".{d}") for d in trusted)
    if allow:
        return any(domain == d or domain.endswith(f".{d}") for d in allow)
    return True


def source_authority(url: str, policy: SourcePolicy) -> float:
    domain = domain_for_url(url)
    trusted = {d.lower() for d in policy.trusted_domains}
    denied = {d.lower() for d in policy.deny_domains}
    if any(domain == d or domain.endswith(f".{d}") for d in trusted):
        return 1.0
    if any(domain == d or domain.endswith(f".{d}") for d in denied):
        return 0.0
    return 0.5


def load_items(path: Path, item_column: str) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        items: list[dict[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if isinstance(raw, dict):
                items.append({str(k): str(v) for k, v in raw.items()})
        return items

    import csv

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [{str(k): str(v) for k, v in row.items() if k is not None} for row in reader]
    for idx, row in enumerate(rows):
        if item_column not in row:
            raise ValueError(f"row {idx} missing item column {item_column!r}")
    return rows


def ingest_source_sets(source_sets: list[SourceSet]) -> tuple[list[SourceRecord], list[SourceChunk]]:
    records: list[SourceRecord] = []
    chunks: list[SourceChunk] = []
    for source_set in source_sets:
        for raw in source_set.paths:
            path = Path(raw).expanduser()
            paths = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
            for p in paths:
                if not p.exists() or p.suffix.lower() in {".duckdb", ".sqlite", ".db"}:
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                url = p.resolve().as_uri()
                source_id = _stable_id(source_set.name, url)
                records.append(
                    SourceRecord(
                        source_id=source_id,
                        title=p.stem,
                        url=url,
                        source_type="file",
                        path=str(p),
                        metadata={"source_set": source_set.name},
                    )
                )
                chunks.extend(_chunks_for(source_id, text))
        for idx, url in enumerate(source_set.urls):
            source_id = _stable_id(source_set.name, url)
            title = domain_for_url(url) or f"url-{idx + 1}"
            records.append(
                SourceRecord(
                    source_id=source_id,
                    title=title,
                    url=url,
                    source_type="url",
                    path="",
                    metadata={"source_set": source_set.name},
                )
            )
            chunks.append(
                SourceChunk(
                    chunk_id=f"{source_id}:0",
                    source_id=source_id,
                    ordinal=0,
                    text=url,
                )
            )
        for domain in source_set.crawl_domains:
            url = f"https://{domain.strip('/')}"
            source_id = _stable_id(source_set.name, url)
            records.append(
                SourceRecord(
                    source_id=source_id,
                    title=domain,
                    url=url,
                    source_type="domain",
                    path="",
                    metadata={"source_set": source_set.name, "crawl": True},
                )
            )
            chunks.append(
                SourceChunk(
                    chunk_id=f"{source_id}:0",
                    source_id=source_id,
                    ordinal=0,
                    text=f"Domain crawl seed: {url}",
                )
            )
    return records, chunks


class SourcePackSearchProvider:
    """Search local source chunks first, then fall back to another provider."""

    name = "source_pack"

    def __init__(
        self,
        chunks: list[dict],
        fallback: SearchProvider | None = None,
        policy: SourcePolicy | None = None,
        legacy_allowlist: list[str] | None = None,
    ) -> None:
        self._chunks = chunks
        self._fallback = fallback
        self._policy = policy or SourcePolicy()
        self._legacy_allowlist = legacy_allowlist or []

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        terms = [t.lower() for t in query.split() if len(t) > 2]
        scored: list[tuple[int, dict]] = []
        for chunk in self._chunks:
            url = str(chunk.get("url", ""))
            if not domain_allowed(url, self._policy, self._legacy_allowlist):
                continue
            text = str(chunk.get("text", ""))
            lowered = text.lower()
            score = sum(lowered.count(t) for t in terms)
            if score > 0 or not terms:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("url", ""))))
        results = [
            SearchResult(
                title=str(chunk.get("title") or chunk.get("source_id") or "source"),
                url=str(chunk.get("url", "")),
                snippet=str(chunk.get("text", ""))[:500],
                rank=i,
            )
            for i, (_score, chunk) in enumerate(scored[:max_results])
        ]
        if len(results) >= max_results or self._fallback is None:
            return results[:max_results]
        fallback = await self._fallback.search(query, max_results=max_results - len(results))
        filtered = [
            r
            for r in fallback
            if domain_allowed(r.url, self._policy, self._legacy_allowlist)
        ]
        return (results + filtered)[:max_results]


class PolicySearchProvider:
    """Filter another SearchProvider through SourcePolicy/domain_allowlist."""

    def __init__(
        self,
        fallback: SearchProvider,
        policy: SourcePolicy,
        legacy_allowlist: list[str] | None = None,
    ) -> None:
        self._fallback = fallback
        self._policy = policy
        self._legacy_allowlist = legacy_allowlist or []
        self.name = f"policy_{getattr(fallback, 'name', 'search')}"

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        results = await self._fallback.search(query, max_results=max_results)
        return [
            r
            for r in results
            if domain_allowed(r.url, self._policy, self._legacy_allowlist)
        ][:max_results]


def _chunks_for(source_id: str, text: str) -> list[SourceChunk]:
    chunks = chunk_text(text, max_chars=2500)
    if not chunks and text.strip():
        return [SourceChunk(chunk_id=f"{source_id}:0", source_id=source_id, ordinal=0, text=text[:2500])]
    return [
        SourceChunk(
            chunk_id=f"{source_id}:{idx}",
            source_id=source_id,
            ordinal=idx,
            text=chunk.text,
        )
        for idx, chunk in enumerate(chunks)
    ]


def _stable_id(namespace: str, value: str) -> str:
    digest = hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()
    return digest[:24]
