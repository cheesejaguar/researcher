/**
 * AgentDagPanel: per-agent state table. One row per agent with its
 * current state and the timestamp of the last event that touched it.
 * Also shows a tail of recent log lines so the operator can see what
 * the agents are actually doing.
 */

import React from "react";
import { Box, Text } from "ink";

import type { AppState } from "../store.js";

interface Props {
  state: AppState;
}

export function AgentDagPanel({ state }: Props): React.JSX.Element {
  const rows = Object.values(state.agents).sort((a, b) =>
    a.agentId.localeCompare(b.agentId),
  );

  return (
    <Box
      borderStyle="round"
      borderColor="magenta"
      flexDirection="column"
      paddingX={1}
    >
      <Text bold color="magenta">
        Agents ({rows.length})
      </Text>
      {rows.length === 0 ? (
        <Text dimColor>no agents yet</Text>
      ) : (
        rows.map((r) => (
          <Text key={r.agentId}>
            {r.agentId}: <Text color="yellow">{r.state}</Text>{" "}
            <Text dimColor>({r.lastUpdate})</Text>
          </Text>
        ))
      )}
      <Box marginTop={1} flexDirection="column">
        <Text bold dimColor>
          recent logs
        </Text>
        {state.recentLogs.length === 0 ? (
          <Text dimColor>-</Text>
        ) : (
          state.recentLogs
            .slice(-5)
            .map((l, i) => (
              <Text key={`${i}-${l.msg}`}>
                <Text
                  color={
                    l.level === "error"
                      ? "red"
                      : l.level === "warn"
                      ? "yellow"
                      : l.level === "info"
                      ? "green"
                      : "gray"
                  }
                >
                  [{l.level}]
                </Text>{" "}
                {l.agentId}: {l.msg}
              </Text>
            ))
        )}
      </Box>
    </Box>
  );
}
