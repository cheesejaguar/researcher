"""Tests for researcher.budget — cost discipline."""

import pytest

from researcher.budget import Budget, BudgetExceededError


def test_initial_state():
    b = Budget(usd_cap=3.0, wall_cap_s=600)
    assert b.remaining() == 3.0
    assert not b.exceeded()
    assert b.total_spent() == 0.0


def test_spend_reduces_remaining():
    b = Budget(usd_cap=3.0, wall_cap_s=600)
    b.spend(0.75)
    assert b.remaining() == pytest.approx(2.25)
    assert b.total_spent() == pytest.approx(0.75)


def test_exceeded_when_over_cap():
    b = Budget(usd_cap=1.0, wall_cap_s=600)
    b.spend(0.5)
    assert not b.exceeded()
    b.spend(0.5)
    assert b.exceeded()
    b.spend(0.01)  # past the cap is still "exceeded", not an error
    assert b.exceeded()


def test_warn_fires_once_per_threshold():
    b = Budget(usd_cap=1.0, wall_cap_s=600, warn_fractions=(0.5, 0.8, 0.95))
    assert b.should_warn() is None
    b.spend(0.4)
    assert b.should_warn() is None  # below 50%
    b.spend(0.15)  # now at 55% -> crosses 0.5
    assert b.should_warn() == 0.5
    assert b.should_warn() is None  # consumed
    b.spend(0.30)  # 85% -> crosses 0.8
    assert b.should_warn() == 0.8
    b.spend(0.15)  # 100% -> crosses 0.95 AND is exceeded
    assert b.should_warn() == 0.95


def test_allows_task_rejects_oversized():
    # Per-task cap defaults to 5% of remaining.
    b = Budget(usd_cap=1.0, wall_cap_s=600)
    assert b.allows_task(0.04)  # under 5%
    assert not b.allows_task(0.06)  # over 5%


def test_allows_task_shrinks_with_remaining():
    b = Budget(usd_cap=1.0, wall_cap_s=600)
    b.spend(0.8)  # remaining=0.2, 5% = 0.01
    assert b.allows_task(0.009)
    assert not b.allows_task(0.05)


def test_allows_entity_rejects_oversized():
    # Per-entity cap defaults to 3% of total cap (NOT remaining).
    b = Budget(usd_cap=1.0, wall_cap_s=600)
    assert b.allows_entity(0.029)
    assert not b.allows_entity(0.031)


def test_wall_exceeded_tracks_start_time():
    import time

    b = Budget(usd_cap=1.0, wall_cap_s=0.1)  # 100ms
    b.start_wall_clock()
    assert not b.wall_exceeded()
    time.sleep(0.15)
    assert b.wall_exceeded()


def test_wall_exceeded_before_start_is_false():
    b = Budget(usd_cap=1.0, wall_cap_s=0.1)
    # Never called start_wall_clock: treat as not started, not exceeded.
    assert not b.wall_exceeded()


def test_raise_if_exceeded_helper():
    b = Budget(usd_cap=1.0, wall_cap_s=600)
    b.spend(0.5)
    b.raise_if_exceeded()  # should not raise
    b.spend(0.6)
    with pytest.raises(BudgetExceededError):
        b.raise_if_exceeded()


# ---------- per-task / per-entity cumulative sub-caps ----------


def test_cumulative_task_spend_blocks_repeated_retries():
    b = Budget(usd_cap=1.0, wall_cap_s=600, per_task_fraction=0.05)
    assert b.allows_task_cumulative("t1", 0.04)
    b.spend_for_task("t1", 0.04)
    # A second 0.04 estimate would total 0.08 > 5% of 1.0.
    assert not b.allows_task_cumulative("t1", 0.04)
    # Other tasks still have their own slice.
    assert b.allows_task_cumulative("t2", 0.04)


def test_cumulative_entity_spend_blocks_one_runaway_entity():
    b = Budget(usd_cap=1.0, wall_cap_s=600, per_entity_fraction=0.03)
    b.spend_for_entity("e1", 0.02)
    assert b.allows_entity_cumulative("e1", 0.01)
    assert not b.allows_entity_cumulative("e1", 0.011)
    assert b.allows_entity_cumulative("e2", 0.03)


def test_subcap_spend_rejects_negative_values():
    b = Budget(usd_cap=1.0, wall_cap_s=600)
    with pytest.raises(ValueError):
        b.spend_for_task("t1", -0.01)
    with pytest.raises(ValueError):
        b.spend_for_entity("e1", -0.01)


def test_subcap_serialization_roundtrip():
    b = Budget(usd_cap=1.0, wall_cap_s=600)
    b.spend_for_task("t1", 0.02)
    b.spend_for_entity("e1", 0.01)
    blob = b.serialize()

    restored = Budget(usd_cap=1.0, wall_cap_s=600)
    restored.restore(blob)
    assert restored.task_spent("t1") == pytest.approx(0.02)
    assert restored.entity_spent("e1") == pytest.approx(0.01)
    # Restored counters keep enforcing the same ceilings after a resume.
    assert not restored.allows_task_cumulative("t1", 0.04)
