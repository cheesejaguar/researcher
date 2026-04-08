/**
 * StatusBar: single-line (rendered as a bordered strip) footer showing
 * run id, cost, token usage, wall time, and terminal stop reason.
 */

import React from "react";
import { Box, Text } from "ink";

import type { AppState } from "../store.js";

interface Props {
  state: AppState;
}

export function StatusBar({ state }: Props): React.JSX.Element {
  const runId = state.runId || "-";
  const status = state.runComplete
    ? state.stopReason ?? "complete"
    : "running";
  const statusColor = state.runComplete
    ? state.stopReason === "error"
      ? "red"
      : "green"
    : "yellow";

  return (
    <Box
      borderStyle="single"
      borderColor="gray"
      paddingX={1}
      flexDirection="row"
      justifyContent="space-between"
    >
      <Text>
        run <Text color="cyan">{runId}</Text> | status{" "}
        <Text color={statusColor}>{status}</Text>
      </Text>
      <Text>
        cost <Text color="green">${state.costUsd.toFixed(4)}</Text> | tok{" "}
        <Text color="green">
          {state.tokensIn}/{state.tokensOut}
        </Text>{" "}
        | wall <Text color="green">{state.wallS.toFixed(1)}s</Text> | events{" "}
        <Text color="gray">{state.eventsSeen}</Text>
      </Text>
    </Box>
  );
}
