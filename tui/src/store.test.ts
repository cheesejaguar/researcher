/**
 * Reducer tests for the Researcher TUI store.
 *
 * These tests exercise the pure `reduce` function and the whole-fixture
 * `reduceAll` helper against hand-crafted events and the on-disk
 * `sample_run.jsonl` fixture.
 */

import { describe, expect, it } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import * as url from "node:url";

import {
  initialState,
  reduce,
  reduceAll,
  MAX_RECENT_FACTS,
  MAX_RECENT_LOGS,
  createStore,
  type AppState,
} from "./store.js";
import { parseEvent, type Event } from "./transport/types.js";

// Resolve the fixture relative to this test file so we don't depend on cwd.
const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
const FIXTURE_PATH = path.join(__dirname, "fixtures", "sample_run.jsonl");

function loadFixture(): Event[] {
  const raw = fs.readFileSync(FIXTURE_PATH, "utf-8");
  return raw
    .split("\n")
    .filter((l) => l.trim().length > 0)
    .map((l) => parseEvent(JSON.parse(l)));
}

function mkEvent<T extends Event["type"]>(
  type: T,
  payload: unknown,
  seq = 0,
  runId = "run-test",
  ts = "2026-04-08T12:00:00Z",
): Event {
  return parseEvent({ type, seq, ts, run_id: runId, payload });
}

describe("initialState", () => {
  it("is empty", () => {
    const s = initialState();
    expect(s.runId).toBe("");
    expect(s.cycles).toBe(0);
    expect(s.currentCycle).toBeNull();
    expect(s.entitiesTotal).toBe(0);
    expect(s.costUsd).toBe(0);
    expect(s.agents).toEqual({});
    expect(s.recentFacts).toEqual([]);
    expect(s.recentLogs).toEqual([]);
    expect(s.runComplete).toBe(false);
    expect(s.stopReason).toBeNull();
    expect(s.subagentCalls).toBe(0);
    expect(s.eventsSeen).toBe(0);
  });
});

describe("reduce", () => {
  it("cycle_start bumps currentCycle and captures run_id", () => {
    const e = mkEvent("cycle_start", { cycle: 3, pending_tasks: 5 });
    const s = reduce(initialState(), e);
    expect(s.currentCycle).toBe(3);
    expect(s.lastCycleStart).toBe(5);
    expect(s.runId).toBe("run-test");
    expect(s.eventsSeen).toBe(1);
  });

  it("agent_spawn registers a new agent entry", () => {
    const e = mkEvent("agent_spawn", {
      agent_id: "sub-42",
      task_id: "t1",
      kind: "discover",
    });
    const s = reduce(initialState(), e);
    expect(s.agents["sub-42"]).toBeDefined();
    expect(s.agents["sub-42"].state).toBe("spawned");
  });

  it("agent_state_change updates the agent's state", () => {
    let s: AppState = initialState();
    s = reduce(
      s,
      mkEvent("agent_spawn", {
        agent_id: "sub-1",
        task_id: "t1",
        kind: "discover",
      }),
    );
    s = reduce(
      s,
      mkEvent("agent_state_change", {
        agent_id: "sub-1",
        old: "spawned",
        new: "fetching",
      }),
    );
    expect(s.agents["sub-1"].state).toBe("fetching");
  });

  it("fact_written appends to recentFacts and is bounded to MAX_RECENT_FACTS", () => {
    let s: AppState = initialState();
    // Emit 25 facts; only the last 10 should be retained.
    for (let i = 0; i < 25; i++) {
      s = reduce(
        s,
        mkEvent("fact_written", {
          entity_id: `ent-${i}`,
          entity_type: "War",
          field: "name",
          value: `Name ${i}`,
          confidence: 0.9,
          source_url: "https://example.com",
        }),
      );
    }
    expect(s.recentFacts.length).toBe(MAX_RECENT_FACTS);
    // The oldest retained fact should be ent-15 (25 - 10 = 15).
    expect(s.recentFacts[0].entityId).toBe("ent-15");
    expect(s.recentFacts[s.recentFacts.length - 1].entityId).toBe("ent-24");
  });

  it("agent_log appends to recentLogs bounded to MAX_RECENT_LOGS", () => {
    let s: AppState = initialState();
    for (let i = 0; i < MAX_RECENT_LOGS + 5; i++) {
      s = reduce(
        s,
        mkEvent("agent_log", {
          agent_id: "sub-1",
          level: "info",
          msg: `msg-${i}`,
        }),
      );
    }
    expect(s.recentLogs.length).toBe(MAX_RECENT_LOGS);
    expect(s.recentLogs[s.recentLogs.length - 1].msg).toBe(
      `msg-${MAX_RECENT_LOGS + 4}`,
    );
  });

  it("cost_update updates cost and tokens", () => {
    const s = reduce(
      initialState(),
      mkEvent("cost_update", {
        cost_usd_total: 1.25,
        tokens_in_total: 4000,
        tokens_out_total: 2500,
      }),
    );
    expect(s.costUsd).toBe(1.25);
    expect(s.tokensIn).toBe(4000);
    expect(s.tokensOut).toBe(2500);
  });

  it("run_complete sets runComplete and stopReason", () => {
    const s = reduce(
      initialState(),
      mkEvent("run_complete", {
        reason: "plateau",
        entities: 7,
        cost_usd: 0.42,
        wall_s: 123.4,
        db_path: "runs/run-test/store.duckdb",
      }),
    );
    expect(s.runComplete).toBe(true);
    expect(s.stopReason).toBe("plateau");
    expect(s.entitiesTotal).toBe(7);
    expect(s.wallS).toBe(123.4);
  });

  it("subagent_call increments subagentCalls", () => {
    const s = reduce(
      initialState(),
      mkEvent("subagent_call", {
        agent_id: "sub-1",
        task_id: "t1",
        cli_kind: "claude_code",
        wall_ms: 6000,
        exit_code: 0,
        claims_emitted: 4,
      }),
    );
    expect(s.subagentCalls).toBe(1);
  });
});

describe("reduceAll over fixture", () => {
  it("produces a sensible final state for sample_run.jsonl", () => {
    const events = loadFixture();
    expect(events.length).toBeGreaterThan(0);

    const s = reduceAll(events);

    // Run metadata
    expect(s.runId).toBe("run-demo");
    expect(s.runComplete).toBe(true);
    expect(s.stopReason).toBe("plateau");

    // Cycles
    expect(s.currentCycle).toBe(1);
    expect(s.cycles).toBeGreaterThanOrEqual(1);

    // Agent dag
    expect(Object.keys(s.agents)).toContain("sub-1");
    expect(s.agents["sub-1"].state).toBe("done");

    // Facts
    expect(s.recentFacts.length).toBeGreaterThanOrEqual(4);
    expect(s.recentFacts.map((f) => f.field)).toEqual(
      expect.arrayContaining(["name", "start_year", "end_year", "belligerents"]),
    );

    // Subagent call counted
    expect(s.subagentCalls).toBe(1);

    // Final entity count comes from run_complete
    expect(s.entitiesTotal).toBe(1);
    expect(s.wallS).toBe(16.0);

    // All events folded
    expect(s.eventsSeen).toBe(events.length);
  });
});

describe("createStore", () => {
  it("dispatches events and notifies subscribers", () => {
    const store = createStore();
    const seen: AppState[] = [];
    const unsub = store.subscribe((s) => seen.push(s));

    store.dispatch(mkEvent("cycle_start", { cycle: 1, pending_tasks: 2 }));
    store.dispatch(
      mkEvent("agent_spawn", {
        agent_id: "sub-1",
        task_id: "t1",
        kind: "discover",
      }),
    );

    expect(seen.length).toBe(2);
    expect(store.getState().currentCycle).toBe(1);
    expect(store.getState().agents["sub-1"]).toBeDefined();

    unsub();
    store.dispatch(mkEvent("cycle_start", { cycle: 2, pending_tasks: 0 }));
    expect(seen.length).toBe(2); // no more notifications after unsubscribe
    expect(store.getState().currentCycle).toBe(2);
  });

  it("reset returns to initial state", () => {
    const store = createStore();
    store.dispatch(mkEvent("cycle_start", { cycle: 9, pending_tasks: 1 }));
    expect(store.getState().currentCycle).toBe(9);
    store.reset();
    expect(store.getState().currentCycle).toBeNull();
    expect(store.getState().eventsSeen).toBe(0);
  });
});
