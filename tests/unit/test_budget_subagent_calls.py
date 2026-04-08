"""Tests for the subagent-call counter + cap on Budget."""

from researcher.budget import Budget


def test_initial_subagent_counter_is_zero():
    b = Budget(usd_cap=3.0, wall_cap_s=600)
    assert b.subagent_calls_total == 0
    assert b.allows_subagent_call()


def test_record_subagent_call_increments_counter():
    b = Budget(usd_cap=3.0, wall_cap_s=600)
    b.record_subagent_call()
    b.record_subagent_call()
    assert b.subagent_calls_total == 2


def test_allows_subagent_call_enforces_cap():
    b = Budget(usd_cap=3.0, wall_cap_s=600, max_subagent_calls=2)
    assert b.allows_subagent_call()
    b.record_subagent_call()
    assert b.allows_subagent_call()
    b.record_subagent_call()
    assert not b.allows_subagent_call()


def test_subagent_counter_does_not_affect_usd_budget():
    b = Budget(usd_cap=1.0, wall_cap_s=600, max_subagent_calls=1000)
    for _ in range(100):
        b.record_subagent_call()
    assert b.total_spent() == 0.0
    assert b.remaining() == 1.0
    assert not b.exceeded()
