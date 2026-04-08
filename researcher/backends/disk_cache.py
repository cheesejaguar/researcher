"""Cross-run disk cache for CLI runner responses.

Stores a dict[str, CliResult] as JSON with per-entry TTL, max-entries
eviction (oldest-first), and fcntl file-locking for multi-process safety.

Keys are opaque sha256 digests the runners compute from (argv, prompt, schema).
Values are CliResult instances that the Pydantic serializer can round-trip.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from researcher.backends.models import CliResult


_VERSION = 1


class DiskCache:
    """JSON-backed cache with TTL, size cap, and file-locked writes.

    Not process-safe across machines (NFS quirks), but safe on a single
    host against concurrent Python processes via fcntl.flock.
    """

    def __init__(
        self,
        path: Path | str,
        max_entries: int = 1000,
        default_ttl_s: float = 86400.0,  # 24 hours
    ) -> None:
        self._path = Path(path)
        self._max_entries = max_entries
        self._default_ttl_s = default_ttl_s
        self._entries: dict[str, dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ---------- Public API ----------

    def get(self, key: str) -> Optional[CliResult]:
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        expires_at = entry.get("expires_at", 0.0)
        if expires_at > 0 and time.time() >= expires_at:
            # Expired — drop and miss.
            del self._entries[key]
            self._save()
            self._misses += 1
            return None
        self._hits += 1
        try:
            return CliResult.model_validate(entry["cli_result"])
        except Exception:
            # Corrupt entry; drop it.
            del self._entries[key]
            self._save()
            self._misses += 1
            return None

    def put(
        self,
        key: str,
        value: CliResult,
        ttl_s: Optional[float] = None,
    ) -> None:
        ttl = ttl_s if ttl_s is not None else self._default_ttl_s
        now = time.time()
        expires_at = now + ttl if ttl > 0 else 0.0
        self._entries[key] = {
            "cached_at": now,
            "expires_at": expires_at,
            "cli_result": value.model_dump(mode="json"),
        }
        # Evict oldest if over cap.
        while len(self._entries) > self._max_entries:
            oldest_key = min(
                self._entries, key=lambda k: self._entries[k].get("cached_at", 0.0)
            )
            del self._entries[oldest_key]
            self._evictions += 1
        self._save()

    def prune(self) -> int:
        """Remove expired entries; return count pruned."""
        now = time.time()
        to_drop = [
            k for k, e in self._entries.items()
            if e.get("expires_at", 0.0) > 0 and now >= e.get("expires_at", 0.0)
        ]
        for k in to_drop:
            del self._entries[k]
        if to_drop:
            self._save()
        return len(to_drop)

    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._save()

    @property
    def stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "size": len(self._entries),
        }

    # ---------- Persistence with file lock ----------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
                except OSError:
                    pass
                try:
                    data = json.load(fh)
                finally:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
            if not isinstance(data, dict) or data.get("version") != _VERSION:
                return
            entries = data.get("entries", {})
            if isinstance(entries, dict):
                self._entries = entries
        except (json.JSONDecodeError, OSError, ValueError):
            # Corrupt or unreadable file — start fresh.
            self._entries = {}

    def _save(self) -> None:
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        data = {"version": _VERSION, "entries": self._entries}
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                except OSError:
                    pass
                try:
                    json.dump(data, fh)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
                finally:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
            os.replace(tmp_path, self._path)
        except OSError as e:
            # Best-effort: don't crash the runner if we can't write the cache.
            if e.errno != errno.ENOSPC:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
