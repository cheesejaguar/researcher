"""BackendResolver — probes $PATH and picks a backend per task."""

from __future__ import annotations

import shutil
from typing import Callable, Literal, Optional

from researcher.backends.models import (
    BackendChoice,
    BackendUnavailableError,
    CliKind,
)


Policy = Literal["auto", "cli", "api"]

# Map CliKind -> executable name to probe.
_PROBES: dict[CliKind, str] = {
    CliKind.CLAUDE_CODE: "claude",
    CliKind.CODEX: "codex",
}


class BackendResolver:
    """Picks a backend for each task.

    At construction, probes $PATH via an injectable `which_fn`. Production
    code passes `shutil.which`; tests pass a stub.

    `detected` is populated in a fixed priority order: Claude Code first,
    then Codex. `pick(policy)` honors the policy and returns the first
    detected CLI (or None for `api` / no-detections / `auto` fall-through).

    `clear_detected(reason)` is the circuit-break hook: once called, all
    subsequent `pick()` calls return `kind=None` for the remainder of the
    resolver's lifetime (which equals the run lifetime).
    """

    def __init__(self, which_fn: Callable[[str], Optional[str]] = shutil.which) -> None:
        self._which_fn = which_fn
        self._detected: list[CliKind] = []
        self._cleared_reason: Optional[str] = None
        self._probe()

    def _probe(self) -> None:
        for kind in (CliKind.CLAUDE_CODE, CliKind.CODEX):
            if self._which_fn(_PROBES[kind]) is not None:
                self._detected.append(kind)

    @property
    def detected(self) -> list[CliKind]:
        if self._cleared_reason is not None:
            return []
        return list(self._detected)

    def clear_detected(self, reason: str) -> None:
        """Circuit break — subsequent picks return kind=None for the rest of the run."""
        self._cleared_reason = reason

    def pick(self, policy: Policy = "auto") -> BackendChoice:
        if policy == "api":
            return BackendChoice(kind=None, reason="policy=api")

        if policy == "cli":
            if not self.detected:
                raise BackendUnavailableError(
                    "policy=cli but no CLI subagent detected on PATH (looked for: claude, codex)"
                )
            kind = self.detected[0]
            return BackendChoice(kind=kind, reason=f"policy=cli, picked {kind.value}")

        # policy == "auto"
        if self._cleared_reason is not None:
            return BackendChoice(kind=None, reason=f"circuit_break: {self._cleared_reason}")
        if not self._detected:
            return BackendChoice(kind=None, reason="auto: no CLI detected")
        kind = self._detected[0]
        return BackendChoice(kind=kind, reason=f"auto: {kind.value} detected")
