"""Tests for SimpleCostTracker — in-memory aggregation of LLMUsage per task."""

from researcher.llm.client import LLMUsage
from researcher.llm.cost_tracker import SimpleCostTracker


def _usage(tokens_in: int = 10, tokens_out: int = 5, cost: float = 0.001,
           model: str = "openrouter/hermes-3-8b", cache_hit: bool = False) -> LLMUsage:
    return LLMUsage(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        model=model,
        cache_hit=cache_hit,
    )


def test_initial_state_is_zero():
    tracker = SimpleCostTracker()
    assert tracker.total_usd() == 0.0
    assert tracker.total_tokens_in == 0
    assert tracker.total_tokens_out == 0
    empty = tracker.by_task("never-seen")
    assert empty.tokens_in == 0
    assert empty.tokens_out == 0
    assert empty.cost_usd == 0.0


def test_add_records_usage_for_a_task():
    tracker = SimpleCostTracker()
    tracker.add("task-1", _usage(tokens_in=100, tokens_out=50, cost=0.5))
    assert tracker.total_usd() == 0.5
    assert tracker.total_tokens_in == 100
    assert tracker.total_tokens_out == 50
    by_task = tracker.by_task("task-1")
    assert by_task.tokens_in == 100
    assert by_task.tokens_out == 50
    assert by_task.cost_usd == 0.5


def test_add_aggregates_across_calls_for_same_task():
    tracker = SimpleCostTracker()
    tracker.add("task-1", _usage(tokens_in=10, tokens_out=5, cost=0.1))
    tracker.add("task-1", _usage(tokens_in=20, tokens_out=15, cost=0.4))
    by_task = tracker.by_task("task-1")
    assert by_task.tokens_in == 30
    assert by_task.tokens_out == 20
    assert abs(by_task.cost_usd - 0.5) < 1e-9
    assert abs(tracker.total_usd() - 0.5) < 1e-9


def test_add_keeps_separate_buckets_per_task():
    tracker = SimpleCostTracker()
    tracker.add("task-a", _usage(tokens_in=10, tokens_out=5, cost=0.1))
    tracker.add("task-b", _usage(tokens_in=30, tokens_out=20, cost=0.4))
    a = tracker.by_task("task-a")
    b = tracker.by_task("task-b")
    assert a.tokens_in == 10
    assert b.tokens_in == 30
    assert abs(tracker.total_usd() - 0.5) < 1e-9
    assert tracker.total_tokens_in == 40
    assert tracker.total_tokens_out == 25


def test_add_propagates_cache_hit_flag():
    tracker = SimpleCostTracker()
    tracker.add("task-1", _usage(cache_hit=False))
    assert tracker.by_task("task-1").cache_hit is False
    tracker.add("task-1", _usage(cache_hit=True))
    # Once any call hits the cache, the rolled-up usage is marked cache_hit.
    assert tracker.by_task("task-1").cache_hit is True
