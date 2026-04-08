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
import time
from pathlib import Path
from typing import Any, Optional, Protocol

from pydantic import ValidationError

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
    ) -> None:
        self._model = model
        self._append_system_prompt = append_system_prompt
        self._post_mortem_dir = post_mortem_dir
        self._cache: dict[str, CliResult] = {}

    async def execute(
        self,
        prompt: str,
        schema: dict,
        timeout_s: float,
        tools: tuple[str, ...] = ("WebSearch", "WebFetch"),
    ) -> CliResult:
        argv = self._build_argv(schema=schema, tools=tools)
        key = _cache_key(argv, prompt, schema)
        if key in self._cache:
            return self._cache[key]

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
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                return CliResult(
                    ok=False,
                    error="timeout",
                    wall_ms=int((time.monotonic() - start) * 1000),
                    exit_code=None,
                )
            except asyncio.CancelledError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                raise
        except asyncio.CancelledError:
            if proc is not None:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
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
