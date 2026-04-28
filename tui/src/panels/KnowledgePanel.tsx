/**
 * KnowledgePanel: entity count and a tail of the most recent facts
 * written by the fusion layer. Shows field name, value (stringified),
 * and confidence for each fact.
 */

import React from "react";
import { Box, Text } from "ink";

import type { AppState, FactEntry } from "../store.js";

interface Props {
  state: AppState;
}

function renderValue(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function KnowledgePanel({ state }: Props): React.JSX.Element {
  const facts: FactEntry[] = state.recentFacts;
  const lowFields = Object.entries(state.fieldsBelowConfidence)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  return (
    <Box
      borderStyle="round"
      borderColor="green"
      flexDirection="column"
      paddingX={1}
    >
      <Text bold color="green">
        Knowledge
      </Text>
      <Text>
        entities total: <Text color="yellow">{state.entitiesTotal}</Text>
      </Text>
      <Text>
        open conflicts: <Text color={state.coverageConflictsOpen > 0 ? "red" : "green"}>
          {state.coverageConflictsOpen}
        </Text>
      </Text>
      {(state.sourcePackSources > 0 || state.sourcePackChunks > 0) && (
        <Text>
          sources: <Text color="cyan">{state.sourcePackSources}</Text> / chunks{" "}
          <Text color="cyan">{state.sourcePackChunks}</Text>
        </Text>
      )}
      {state.verificationVotes.length > 0 && (
        <Text>
          council: <Text color="cyan">{state.verificationVotes.length}</Text> votes /{" "}
          <Text color={state.verificationDisagreements > 0 ? "red" : "green"}>
            {state.verificationDisagreements}
          </Text>{" "}
          disagreements
        </Text>
      )}
      <Text dimColor>report: researcher report {state.runId || "<run_id>"}</Text>
      <Text dimColor>evidence: researcher evidence {state.runId || "<run_id>"}</Text>
      {lowFields.length > 0 && (
        <Text>
          low confidence:{" "}
          <Text color="yellow">
            {lowFields.map(([field, count]) => `${field}:${count}`).join(", ")}
          </Text>
        </Text>
      )}
      {state.coverageRecommendations.length > 0 && (
        <>
          <Text dimColor>next seeds</Text>
          {state.coverageRecommendations.slice(0, 3).map((seed, i) => (
            <Text key={`${i}-${seed}`} color="magenta">
              {seed}
            </Text>
          ))}
        </>
      )}
      <Text dimColor>recent facts</Text>
      {facts.length === 0 ? (
        <Text dimColor>-</Text>
      ) : (
        facts.map((f, i) => (
          <Text key={`${i}-${f.entityId}-${f.field}`}>
            <Text color="cyan">{f.entityId}</Text>.
            <Text color="yellow">{f.field}</Text> ={" "}
            <Text>{renderValue(f.value)}</Text>{" "}
            <Text dimColor>(c={f.confidence.toFixed(2)})</Text>
          </Text>
        ))
      )}
    </Box>
  );
}
