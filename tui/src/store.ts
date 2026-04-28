/**
 * Pure reducer + minimal subscribable store for the Researcher TUI.
 *
 * The store folds the JSONL event stream (or live socket stream) into a
 * single AppState. All reducer operations are pure functions so they are
 * trivially unit-testable; the mutable surface lives in the tiny Store
 * wrapper below.
 *
 * No zustand / redux dependency: we roll our own 30-line store to keep
 * the TUI's `node_modules` footprint small.
 */

import type { Event } from "./transport/types.js";

// ---------- State shapes ----------

export interface AgentEntry {
  agentId: string;
  state: string;
  lastUpdate: string; // ISO timestamp of most recent event touching this agent
}

export interface FactEntry {
  entityId: string;
  entityType: string;
  field: string;
  value: unknown;
  confidence: number;
  sourceUrl: string;
}

export interface LogEntry {
  agentId: string;
  level: "debug" | "info" | "warn" | "error";
  msg: string;
}

export interface VerificationVoteEntry {
  field: string;
  model: string;
  confidence: number;
  disagreement: boolean;
}

export interface AppState {
  runId: string;
  cycles: number;
  currentCycle: number | null;
  lastCycleStart: number | null;
  entitiesTotal: number;
  costUsd: number;
  tokensIn: number;
  tokensOut: number;
  wallS: number;
  agents: Record<string, AgentEntry>;
  recentFacts: FactEntry[]; // bounded to last 10
  recentLogs: LogEntry[]; // bounded to last 20
  runComplete: boolean;
  stopReason: string | null;
  subagentCalls: number;
  lastEventTs: string | null;
  eventsSeen: number;
  entitiesByType: Record<string, number>;
  fieldsBelowConfidence: Record<string, number>;
  coverageRecommendations: string[];
  coverageConflictsOpen: number;
  sourceTypeBreakdown: Record<string, number>;
  sourcePackSources: number;
  sourcePackChunks: number;
  verificationVotes: VerificationVoteEntry[];
  verificationDisagreements: number;
}

export const MAX_RECENT_FACTS = 10;
export const MAX_RECENT_LOGS = 20;

export function initialState(): AppState {
  return {
    runId: "",
    cycles: 0,
    currentCycle: null,
    lastCycleStart: null,
    entitiesTotal: 0,
    costUsd: 0,
    tokensIn: 0,
    tokensOut: 0,
    wallS: 0,
    agents: {},
    recentFacts: [],
    recentLogs: [],
    runComplete: false,
    stopReason: null,
    subagentCalls: 0,
    lastEventTs: null,
    eventsSeen: 0,
    entitiesByType: {},
    fieldsBelowConfidence: {},
    coverageRecommendations: [],
    coverageConflictsOpen: 0,
    sourceTypeBreakdown: {},
    sourcePackSources: 0,
    sourcePackChunks: 0,
    verificationVotes: [],
    verificationDisagreements: 0,
  };
}

// ---------- Reducer ----------

/**
 * Fold a single event into the current state, returning a new state
 * object. The reducer is total: any unrecognized event type yields the
 * state unchanged (modulo the eventsSeen / run_id bookkeeping).
 */
export function reduce(state: AppState, event: Event): AppState {
  const base: AppState = {
    ...state,
    runId: state.runId || event.run_id,
    lastEventTs: event.ts,
    eventsSeen: state.eventsSeen + 1,
  };

  switch (event.type) {
    case "cycle_start": {
      return {
        ...base,
        currentCycle: event.payload.cycle,
        lastCycleStart: event.payload.pending_tasks,
      };
    }

    case "cycle_end": {
      return {
        ...base,
        cycles: Math.max(state.cycles, event.payload.cycle),
        entitiesTotal: event.payload.entities_total,
        costUsd: event.payload.cost_usd,
      };
    }

    case "agent_spawn": {
      const agentId = event.payload.agent_id;
      return {
        ...base,
        agents: {
          ...state.agents,
          [agentId]: {
            agentId,
            state: "spawned",
            lastUpdate: event.ts,
          },
        },
      };
    }

    case "agent_state_change": {
      const agentId = event.payload.agent_id;
      return {
        ...base,
        agents: {
          ...state.agents,
          [agentId]: {
            agentId,
            state: event.payload.new,
            lastUpdate: event.ts,
          },
        },
      };
    }

    case "agent_log": {
      const entry: LogEntry = {
        agentId: event.payload.agent_id,
        level: event.payload.level,
        msg: event.payload.msg,
      };
      const nextLogs = [...state.recentLogs, entry].slice(-MAX_RECENT_LOGS);
      return {
        ...base,
        recentLogs: nextLogs,
      };
    }

    case "fact_written": {
      const entry: FactEntry = {
        entityId: event.payload.entity_id,
        entityType: event.payload.entity_type,
        field: event.payload.field,
        value: event.payload.value,
        confidence: event.payload.confidence,
        sourceUrl: event.payload.source_url,
      };
      const nextFacts = [...state.recentFacts, entry].slice(-MAX_RECENT_FACTS);
      return {
        ...base,
        recentFacts: nextFacts,
      };
    }

    case "cost_update": {
      return {
        ...base,
        costUsd: event.payload.cost_usd_total,
        tokensIn: event.payload.tokens_in_total,
        tokensOut: event.payload.tokens_out_total,
      };
    }

    case "budget_warning": {
      // Log-only; surfaced via recentLogs so the UI can flag it.
      const entry: LogEntry = {
        agentId: "orchestrator",
        level: "warn",
        msg: `budget ${(event.payload.fraction * 100).toFixed(0)}% (${event.payload.cost_usd_total.toFixed(4)}/${event.payload.budget_usd.toFixed(2)})`,
      };
      return {
        ...base,
        recentLogs: [...state.recentLogs, entry].slice(-MAX_RECENT_LOGS),
      };
    }

    case "subagent_call": {
      return {
        ...base,
        subagentCalls: state.subagentCalls + 1,
      };
    }

    case "interrupt_requested": {
      const entry: LogEntry = {
        agentId: "orchestrator",
        level: "warn",
        msg: `interrupt requested: ${event.payload.point}`,
      };
      return {
        ...base,
        recentLogs: [...state.recentLogs, entry].slice(-MAX_RECENT_LOGS),
      };
    }

    case "interrupt_resolved": {
      const entry: LogEntry = {
        agentId: "orchestrator",
        level: event.payload.decision === "continue" ? "info" : "warn",
        msg: `interrupt ${event.payload.point}: ${event.payload.decision}`,
      };
      return {
        ...base,
        recentLogs: [...state.recentLogs, entry].slice(-MAX_RECENT_LOGS),
      };
    }

    case "coverage_report": {
      return {
        ...base,
        entitiesByType: event.payload.entities_by_type,
        fieldsBelowConfidence: event.payload.fields_below_confidence,
        coverageRecommendations: event.payload.next_recommended_seeds,
        coverageConflictsOpen: event.payload.conflicts_open,
        sourceTypeBreakdown: event.payload.source_type_breakdown,
      };
    }

    case "source_pack_loaded": {
      return {
        ...base,
        sourcePackSources: event.payload.sources,
        sourcePackChunks: event.payload.chunks,
      };
    }

    case "verification_vote": {
      const vote: VerificationVoteEntry = {
        field: event.payload.field,
        model: event.payload.model,
        confidence: event.payload.confidence,
        disagreement: event.payload.disagreement,
      };
      return {
        ...base,
        verificationVotes: [...state.verificationVotes, vote].slice(-20),
        verificationDisagreements:
          state.verificationDisagreements + (vote.disagreement ? 1 : 0),
      };
    }

    case "run_complete": {
      return {
        ...base,
        runComplete: true,
        stopReason: event.payload.reason,
        entitiesTotal: event.payload.entities,
        costUsd: event.payload.cost_usd,
        wallS: event.payload.wall_s,
      };
    }

    case "conflict_detected":
    case "conflict_resolved": {
      // Surface conflicts in the log feed so they show up in the UI.
      const entry: LogEntry = {
        agentId: "fusion",
        level: "warn",
        msg: `${event.type} ${event.payload.entity_id}.${event.payload.field}: ${event.payload.reason}`,
      };
      return {
        ...base,
        recentLogs: [...state.recentLogs, entry].slice(-MAX_RECENT_LOGS),
      };
    }

    default: {
      // Exhaustiveness guard: if zod adds a new variant the compiler will
      // flag this branch. At runtime we just pass through.
      return base;
    }
  }
}

/**
 * Fold a whole sequence of events into a fresh state. Convenient for
 * tests and for replay mode.
 */
export function reduceAll(events: Iterable<Event>): AppState {
  let s = initialState();
  for (const e of events) {
    s = reduce(s, e);
  }
  return s;
}

// ---------- Minimal subscribable store ----------

export type Listener = (state: AppState) => void;

export interface Store {
  getState(): AppState;
  subscribe(l: Listener): () => void;
  dispatch(event: Event): void;
  reset(): void;
}

export function createStore(seed: AppState = initialState()): Store {
  let state = seed;
  const listeners = new Set<Listener>();

  return {
    getState: () => state,
    subscribe(l: Listener) {
      listeners.add(l);
      return () => {
        listeners.delete(l);
      };
    },
    dispatch(event: Event) {
      state = reduce(state, event);
      for (const l of listeners) l(state);
    },
    reset() {
      state = initialState();
      for (const l of listeners) l(state);
    },
  };
}
