/**
 * Event schema for the TUI, mirroring researcher/events.py.
 *
 * Keep this in lockstep with the Python side. The JSONL files on disk and
 * the bounded socket queue both emit JSON that matches these zod schemas.
 *
 * Wave 0 ships only the types + parser. Wave 1-F builds the transports
 * (UnixSocketTransport, JsonlReplayTransport) and the App that consumes them.
 */

import { z } from "zod";

// ---------- Payloads ----------

export const CycleStartPayload = z.object({
  cycle: z.number().int(),
  pending_tasks: z.number().int(),
});

export const CycleEndPayload = z.object({
  cycle: z.number().int(),
  claims_written: z.number().int(),
  entities_total: z.number().int(),
  cost_usd: z.number(),
});

export const AgentSpawnPayload = z.object({
  agent_id: z.string(),
  task_id: z.string(),
  kind: z.string(),
});

export const AgentStateChangePayload = z.object({
  agent_id: z.string(),
  old: z.string(),
  new: z.string(),
});

export const AgentLogPayload = z.object({
  agent_id: z.string(),
  level: z.enum(["debug", "info", "warn", "error"]),
  msg: z.string(),
});

export const FactWrittenPayload = z.object({
  entity_id: z.string(),
  entity_type: z.string(),
  field: z.string(),
  value: z.unknown(),
  confidence: z.number(),
  source_url: z.string(),
});

export const ConflictPayload = z.object({
  entity_id: z.string(),
  field: z.string(),
  winning_value: z.unknown().nullable().optional(),
  losing_value: z.unknown().nullable().optional(),
  reason: z.string(),
});

export const CostUpdatePayload = z.object({
  cost_usd_total: z.number(),
  tokens_in_total: z.number().int(),
  tokens_out_total: z.number().int(),
});

export const BudgetWarningPayload = z.object({
  cost_usd_total: z.number(),
  budget_usd: z.number(),
  fraction: z.number(),
});

export const RunCompletePayload = z.object({
  reason: z.enum([
    "budget",
    "plateau",
    "ctrl_c",
    "error",
    "deadline",
    "no_tasks",
    "subagent_cap",
  ]),
  entities: z.number().int(),
  cost_usd: z.number(),
  wall_s: z.number(),
  db_path: z.string(),
});

export const SubagentCallPayload = z.object({
  agent_id: z.string(),
  task_id: z.string(),
  cli_kind: z.enum(["claude_code", "codex"]),
  wall_ms: z.number().int(),
  exit_code: z.number().int().nullable(),
  claims_emitted: z.number().int(),
});

export const CoverageReportPayload = z.object({
  cycle: z.number().int(),
  entities_by_type: z.record(z.string(), z.number().int()),
  fields_below_confidence: z.record(z.string(), z.number().int()),
  confidence_threshold: z.number(),
  conflicts_open: z.number().int(),
  source_type_breakdown: z.record(z.string(), z.number().int()),
  next_recommended_seeds: z.array(z.string()),
});

export const SourcePackPayload = z.object({
  sources: z.number().int(),
  chunks: z.number().int(),
});

export const VerificationVotePayload = z.object({
  entity_id: z.string().nullable().optional(),
  field: z.string(),
  model: z.string(),
  vote: z.unknown(),
  confidence: z.number(),
  disagreement: z.boolean().default(false),
});

export const InterruptRequestedPayload = z.object({
  point: z.string(),
  context: z.record(z.string(), z.unknown()).default({}),
});

export const InterruptResolvedPayload = z.object({
  point: z.string(),
  decision: z.enum(["continue", "abort"]),
});

// ---------- Event envelope ----------

const base = {
  seq: z.number().int(),
  ts: z.string(), // ISO8601
  run_id: z.string(),
};

export const CycleStart = z.object({ type: z.literal("cycle_start"), ...base, payload: CycleStartPayload });
export const CycleEnd = z.object({ type: z.literal("cycle_end"), ...base, payload: CycleEndPayload });
export const AgentSpawn = z.object({ type: z.literal("agent_spawn"), ...base, payload: AgentSpawnPayload });
export const AgentStateChange = z.object({ type: z.literal("agent_state_change"), ...base, payload: AgentStateChangePayload });
export const AgentLog = z.object({ type: z.literal("agent_log"), ...base, payload: AgentLogPayload });
export const FactWritten = z.object({ type: z.literal("fact_written"), ...base, payload: FactWrittenPayload });
export const ConflictDetected = z.object({ type: z.literal("conflict_detected"), ...base, payload: ConflictPayload });
export const ConflictResolved = z.object({ type: z.literal("conflict_resolved"), ...base, payload: ConflictPayload });
export const CostUpdate = z.object({ type: z.literal("cost_update"), ...base, payload: CostUpdatePayload });
export const BudgetWarning = z.object({ type: z.literal("budget_warning"), ...base, payload: BudgetWarningPayload });
export const RunComplete = z.object({ type: z.literal("run_complete"), ...base, payload: RunCompletePayload });
export const SubagentCall = z.object({ type: z.literal("subagent_call"), ...base, payload: SubagentCallPayload });
export const InterruptRequested = z.object({ type: z.literal("interrupt_requested"), ...base, payload: InterruptRequestedPayload });
export const InterruptResolved = z.object({ type: z.literal("interrupt_resolved"), ...base, payload: InterruptResolvedPayload });
export const CoverageReport = z.object({ type: z.literal("coverage_report"), ...base, payload: CoverageReportPayload });
export const SourcePackLoaded = z.object({ type: z.literal("source_pack_loaded"), ...base, payload: SourcePackPayload });
export const VerificationVote = z.object({ type: z.literal("verification_vote"), ...base, payload: VerificationVotePayload });

export const Event = z.discriminatedUnion("type", [
  CycleStart,
  CycleEnd,
  AgentSpawn,
  AgentStateChange,
  AgentLog,
  FactWritten,
  ConflictDetected,
  ConflictResolved,
  CostUpdate,
  BudgetWarning,
  RunComplete,
  SubagentCall,
  InterruptRequested,
  InterruptResolved,
  CoverageReport,
  SourcePackLoaded,
  VerificationVote,
]);

export type Event = z.infer<typeof Event>;
export type CycleStart = z.infer<typeof CycleStart>;
export type CycleEnd = z.infer<typeof CycleEnd>;
export type AgentSpawn = z.infer<typeof AgentSpawn>;
export type AgentStateChange = z.infer<typeof AgentStateChange>;
export type AgentLog = z.infer<typeof AgentLog>;
export type FactWritten = z.infer<typeof FactWritten>;
export type ConflictDetected = z.infer<typeof ConflictDetected>;
export type ConflictResolved = z.infer<typeof ConflictResolved>;
export type CostUpdate = z.infer<typeof CostUpdate>;
export type BudgetWarning = z.infer<typeof BudgetWarning>;
export type RunComplete = z.infer<typeof RunComplete>;
export type SubagentCall = z.infer<typeof SubagentCall>;
export type InterruptRequested = z.infer<typeof InterruptRequested>;
export type InterruptResolved = z.infer<typeof InterruptResolved>;
export type CoverageReport = z.infer<typeof CoverageReport>;
export type SourcePackLoaded = z.infer<typeof SourcePackLoaded>;
export type VerificationVote = z.infer<typeof VerificationVote>;

export function parseEvent(raw: unknown): Event {
  return Event.parse(raw);
}
