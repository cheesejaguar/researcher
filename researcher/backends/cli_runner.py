"""Subprocess runners for the CLI subagent path.

Both runners share a common Protocol. Each `execute()` call:
  1. Builds argv (never shell=True, never interpolates prompt into argv).
  2. Spawns the child via `asyncio.create_subprocess_exec`.
  3. Writes the prompt body to stdin; closes stdin.
  4. Awaits `communicate()` with a timeout.
  5. Parses output into a `CliResult`.

Errors become typed CliResult(ok=False); `CancelledError` propagates and
the child is killed first.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Protocol

from pydantic import ValidationError

from researcher.backends.disk_cache import DiskCache
from researcher.backends.models import CliResult, SubagentResponse


# Patterns that indicate auth or usage-limit errors in CLI stderr.
_AUTH_PATTERNS = ("please run: claude auth", "not signed in", "please log in", "codex login")
_USAGE_LIMIT_PATTERNS = ("rate_limit", "usage_limit", "quota", "monthly limit")


def _match_any(haystack: str, needles: tuple[str, ...]) -> bool:
    h = haystack.lower()
    return any(n in h for n in needles)


def _cache_key(argv: list[str], prompt: str, schema: dict) -> str:
    blob = json.dumps(
        {"argv": argv, "prompt": prompt, "schema": schema},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()


async def _kill_and_reap(proc: Any) -> None:
    """Best-effort kill + reap. Swallows ProcessLookupError if the proc is already gone.

    Used by every cleanup site so a double-kill (e.g. inner cancel handler reaps,
    then outer handler tries again) doesn't mask the in-flight exception with
    ProcessLookupError.
    """
    if proc is None:
        return
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    try:
        await proc.wait()
    except Exception:
        pass


class CliRunner(Protocol):
    """Protocol all CLI runners implement."""

    async def execute(
        self,
        prompt: str,
        schema: dict,
        timeout_s: float,
        tools: tuple[str, ...] = ("WebSearch", "WebFetch"),
    ) -> CliResult: ...


class ClaudeCodeRunner:
    """Runs `claude -p --print --output-format json --json-schema ...`."""

    def __init__(
        self,
        model: str = "sonnet",
        append_system_prompt: str = "",
        post_mortem_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._model = model
        self._append_system_prompt = append_system_prompt
        self._post_mortem_dir = post_mortem_dir
        self._cache: dict[str, CliResult] = {}
        self._disk_cache: Optional[DiskCache] = None
        if cache_dir is not None:
            self._disk_cache = DiskCache(path=Path(cache_dir) / "claude_cache.json")

    async def execute(
        self,
        prompt: str,
        schema: dict,
        timeout_s: float,
        tools: tuple[str, ...] = ("WebSearch", "WebFetch"),
    ) -> CliResult:
        argv = self._build_argv(schema=schema, tools=tools)
        key = _cache_key(argv, prompt, schema)

        # L1: in-memory cache.
        if key in self._cache:
            return self._cache[key]

        # L2: disk cache.
        if self._disk_cache is not None:
            cached = self._disk_cache.get(key)
            if cached is not None:
                self._cache[key] = cached
                return cached

        result = await self._run_once(argv, prompt, timeout_s, stricter=False)

        # Malformed JSON → retry once with a stricter nudge in the prompt.
        if (not result.ok) and result.error and result.error.startswith("parse_failed"):
            stricter_prompt = (
                prompt
                + "\n\nIMPORTANT: Return ONLY a single JSON object matching the schema. "
                + "No prose, no markdown, no commentary. Your previous response could not be parsed."
            )
            result = await self._run_once(argv, stricter_prompt, timeout_s, stricter=True)

        # Only cache successful results — failures are transient and should be retried.
        if result.ok:
            self._cache[key] = result
            if self._disk_cache is not None:
                self._disk_cache.put(key, result)
        return result

    def _build_argv(self, *, schema: dict, tools: tuple[str, ...]) -> list[str]:
        argv: list[str] = [
            "claude",
            "-p",
            "--print",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema),
            "--bare",
            "--model",
            self._model,
            "--allowedTools",
            *tools,
            "--dangerously-skip-permissions",
        ]
        if self._append_system_prompt:
            argv.extend(["--append-system-prompt", self._append_system_prompt])
        return argv

    async def _run_once(
        self,
        argv: list[str],
        prompt: str,
        timeout_s: float,
        stricter: bool,
    ) -> CliResult:
        start = time.monotonic()
        proc: Any = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            return CliResult(
                ok=False,
                error=f"spawn_failed: {e}",
                wall_ms=int((time.monotonic() - start) * 1000),
                exit_code=None,
            )
        except PermissionError as e:
            return CliResult(
                ok=False,
                error=f"spawn_failed: {e}",
                wall_ms=int((time.monotonic() - start) * 1000),
                exit_code=None,
            )

        try:
            if proc.stdin is not None:
                proc.stdin.write(prompt.encode("utf-8"))
                if hasattr(proc.stdin, "drain"):
                    maybe_coro = proc.stdin.drain()
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro
                proc.stdin.close()

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                await _kill_and_reap(proc)
                return CliResult(
                    ok=False,
                    error="timeout",
                    wall_ms=int((time.monotonic() - start) * 1000),
                    exit_code=None,
                )
            except asyncio.CancelledError:
                await _kill_and_reap(proc)
                raise
        except asyncio.CancelledError:
            await _kill_and_reap(proc)
            raise

        wall_ms = int((time.monotonic() - start) * 1000)
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

        # Auth + usage patterns take priority over exit code.
        if _match_any(stderr_str, _AUTH_PATTERNS):
            return CliResult(
                ok=False, error="auth_required", wall_ms=wall_ms, exit_code=proc.returncode
            )
        if _match_any(stderr_str, _USAGE_LIMIT_PATTERNS):
            return CliResult(
                ok=False,
                error="usage_limit_reached",
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )

        if proc.returncode != 0:
            tail = stderr_str.strip().splitlines()[-5:] if stderr_str.strip() else [""]
            tail_str = " | ".join(tail)[:2048]
            return CliResult(
                ok=False,
                error=f"exit {proc.returncode}: {tail_str}",
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )

        # Parse the Claude Code JSON envelope.
        try:
            envelope = json.loads(stdout.decode("utf-8", errors="replace"))
            inner_str = envelope.get("result", "")
            inner = json.loads(inner_str) if isinstance(inner_str, str) else inner_str
            data = SubagentResponse.model_validate(inner)
            return CliResult(
                ok=True,
                data=data,
                wall_ms=wall_ms,
                exit_code=proc.returncode,
                raw_usage=envelope.get("usage"),
            )
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            self._dump_post_mortem(stdout, stricter=stricter)
            return CliResult(
                ok=False,
                error=f"parse_failed: {e}",
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )

    def _dump_post_mortem(self, stdout: bytes, stricter: bool) -> None:
        if self._post_mortem_dir is None:
            return
        self._post_mortem_dir.mkdir(parents=True, exist_ok=True)
        suffix = "retry" if stricter else "first"
        ts = int(time.time() * 1000)
        path = self._post_mortem_dir / f"claude_{ts}_{suffix}.txt"
        try:
            path.write_bytes(stdout)
        except OSError:
            pass


class CodexRunner:
    """Runs `codex exec --json --output-schema <file> --output-last-message <file>`.

    Unlike Claude Code, Codex streams JSONL events on stdout and writes the
    final assistant message to `--output-last-message <file>`. We point that
    at a file we control, then read it back after the process exits.
    """

    def __init__(
        self,
        model: str = "o4-mini",
        last_message_file: Optional[Path] = None,
        schema_file_dir: Optional[Path] = None,
        post_mortem_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._model = model
        self._last_message_file = last_message_file
        self._schema_file_dir = schema_file_dir or Path(tempfile.gettempdir())
        self._post_mortem_dir = post_mortem_dir
        self._cache: dict[str, CliResult] = {}
        self._disk_cache: Optional[DiskCache] = None
        if cache_dir is not None:
            self._disk_cache = DiskCache(path=Path(cache_dir) / "codex_cache.json")

    async def execute(
        self,
        prompt: str,
        schema: dict,
        timeout_s: float,
        tools: tuple[str, ...] = ("WebSearch", "WebFetch"),  # Codex has browsing by default
    ) -> CliResult:
        # Write schema to a temp file.
        fd, schema_path = tempfile.mkstemp(
            prefix="researcher_schema_",
            suffix=".json",
            dir=str(self._schema_file_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(schema, fh)

            # Last-message file: use the injected one for tests, or a fresh temp file.
            if self._last_message_file is not None:
                last_message_path = self._last_message_file
            else:
                lm_fd, lm_path = tempfile.mkstemp(
                    prefix="researcher_lastmsg_", suffix=".txt"
                )
                os.close(lm_fd)
                last_message_path = Path(lm_path)

            argv = [
                "codex",
                "exec",
                "--json",
                "--output-schema",
                schema_path,
                "--output-last-message",
                str(last_message_path),
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "-m",
                self._model,
                "--dangerously-bypass-approvals-and-sandbox",
            ]

            # Cache key must be stable across invocations: the tempfile paths
            # (schema_path, last_message_path) are nondeterministic per-call,
            # so we strip them and only hash the stable portions.
            stable_argv = [
                "codex",
                "exec",
                "--json",
                "--output-schema",
                "<schema>",
                "--output-last-message",
                "<last_message>",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "-m",
                self._model,
                "--dangerously-bypass-approvals-and-sandbox",
            ]
            key = _cache_key(stable_argv, prompt, schema)

            # L1: in-memory.
            if key in self._cache:
                return self._cache[key]

            # L2: disk.
            if self._disk_cache is not None:
                cached = self._disk_cache.get(key)
                if cached is not None:
                    self._cache[key] = cached
                    return cached

            result = await self._run_once(argv, prompt, last_message_path, timeout_s)
            if result.ok:
                self._cache[key] = result
                if self._disk_cache is not None:
                    self._disk_cache.put(key, result)
            return result
        finally:
            try:
                os.unlink(schema_path)
            except OSError:
                pass

    async def _run_once(
        self, argv: list[str], prompt: str, last_message_path: Path, timeout_s: float
    ) -> CliResult:
        start = time.monotonic()
        proc: Any = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, PermissionError) as e:
            return CliResult(
                ok=False,
                error=f"spawn_failed: {e}",
                wall_ms=int((time.monotonic() - start) * 1000),
                exit_code=None,
            )

        try:
            if proc.stdin is not None:
                proc.stdin.write(prompt.encode("utf-8"))
                if hasattr(proc.stdin, "drain"):
                    drain_result = proc.stdin.drain()
                    if asyncio.iscoroutine(drain_result):
                        await drain_result
                proc.stdin.close()

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                await _kill_and_reap(proc)
                return CliResult(
                    ok=False,
                    error="timeout",
                    wall_ms=int((time.monotonic() - start) * 1000),
                    exit_code=None,
                )
            except asyncio.CancelledError:
                await _kill_and_reap(proc)
                raise
        except asyncio.CancelledError:
            await _kill_and_reap(proc)
            raise

        wall_ms = int((time.monotonic() - start) * 1000)
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

        if _match_any(stderr_str, _AUTH_PATTERNS):
            return CliResult(
                ok=False, error="auth_required", wall_ms=wall_ms, exit_code=proc.returncode
            )
        if _match_any(stderr_str, _USAGE_LIMIT_PATTERNS):
            return CliResult(
                ok=False,
                error="usage_limit_reached",
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )

        if proc.returncode != 0:
            tail = stderr_str.strip().splitlines()[-5:] if stderr_str.strip() else [""]
            tail_str = " | ".join(tail)[:2048]
            return CliResult(
                ok=False,
                error=f"exit {proc.returncode}: {tail_str}",
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )

        # Read the last-message file rather than stdout (which is JSONL events).
        try:
            raw = last_message_path.read_text(encoding="utf-8")
            inner = json.loads(raw)
            data = SubagentResponse.model_validate(inner)
            return CliResult(
                ok=True,
                data=data,
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            return CliResult(
                ok=False,
                error=f"parse_failed: {e}",
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )
