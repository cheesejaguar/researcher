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
