"""Budget discipline — USD spend tracking + wall clock + per-task/entity caps + warnings."""

from __future__ import annotations

import time
from typing import Optional


class BudgetExceededError(RuntimeError):
    """Raised when the orchestrator tries to proceed past a hard budget cap."""


class Budget:
    """Tracks USD spend against a hard cap with warning fractions and per-task/entity sub-caps.

    Warnings fire exactly once per threshold, in order. `should_warn()` returns the
    next threshold that has been crossed since the last call, or None.

    Per-task cap (default 5%) is computed against `remaining()` — runaway retry loops
    hit this first. Per-entity cap (default 3%) is against the total cap — no single
    entity can dominate the run.
    """

    def __init__(
        self,
        usd_cap: float,
        wall_cap_s: float,
        warn_fractions: tuple[float, ...] = (0.5, 0.8, 0.95),
        per_task_fraction: float = 0.05,
        per_entity_fraction: float = 0.03,
        max_subagent_calls: int = 500,
    ) -> None:
        if usd_cap <= 0:
            raise ValueError("usd_cap must be > 0")
        self._cap = float(usd_cap)
        self._wall_cap_s = float(wall_cap_s)
        self._spent = 0.0
        self._warn_fractions = tuple(sorted(warn_fractions))
        self._warned: set[float] = set()
        self._per_task_fraction = per_task_fraction
        self._per_entity_fraction = per_entity_fraction
        self._wall_started_at: Optional[float] = None
        self._max_subagent_calls = max_subagent_calls
        self._subagent_calls_total = 0

    # ---------- Spend tracking ----------

    def spend(self, usd: float) -> None:
        if usd < 0:
            raise ValueError("spend must be non-negative")
        self._spent += usd

    def total_spent(self) -> float:
        return self._spent

    def remaining(self) -> float:
        return max(0.0, self._cap - self._spent)

    def exceeded(self) -> bool:
        return self._spent >= self._cap

    def raise_if_exceeded(self) -> None:
        if self.exceeded():
            raise BudgetExceededError(
                f"budget exceeded: spent ${self._spent:.4f} of ${self._cap:.4f} cap"
            )

    # ---------- Warnings ----------

    def should_warn(self) -> Optional[float]:
        """Return the next warn-fraction that has been newly crossed, or None.

        Each fraction is returned at most once across the lifetime of this Budget.
        """
        frac = self._spent / self._cap if self._cap > 0 else 0.0
        for f in self._warn_fractions:
            if f in self._warned:
                continue
            if frac >= f:
                self._warned.add(f)
                return f
        return None

    # ---------- Sub-caps ----------

    def allows_task(self, estimated_usd: float) -> bool:
        """True iff a proposed task's estimated cost is within the per-task fraction of remaining."""
        if estimated_usd < 0:
            return False
        ceiling = self.remaining() * self._per_task_fraction
        return estimated_usd <= ceiling

    def allows_entity(self, estimated_usd: float) -> bool:
        """True iff a proposed entity's estimated cost is within the per-entity fraction of total."""
        if estimated_usd < 0:
            return False
        ceiling = self._cap * self._per_entity_fraction
        return estimated_usd <= ceiling

    # ---------- Wall clock ----------

    def start_wall_clock(self) -> None:
        self._wall_started_at = time.monotonic()

    def wall_exceeded(self) -> bool:
        if self._wall_started_at is None:
            return False
        return (time.monotonic() - self._wall_started_at) >= self._wall_cap_s

    # ---------- Subagent calls ----------

    @property
    def subagent_calls_total(self) -> int:
        return self._subagent_calls_total

    def record_subagent_call(self) -> None:
        self._subagent_calls_total += 1

    def allows_subagent_call(self) -> bool:
        return self._subagent_calls_total < self._max_subagent_calls
