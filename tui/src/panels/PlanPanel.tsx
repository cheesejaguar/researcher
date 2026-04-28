/**
 * PlanPanel: shows the current cycle number, total cycles seen so far,
 * and the pending_tasks count from the most recent cycle_start event.
 */

import React from "react";
import { Box, Text } from "ink";

import type { AppState } from "../store.js";

interface Props {
  state: AppState;
}

export function PlanPanel({ state }: Props): React.JSX.Element {
  const cycle =
    state.currentCycle === null ? "-" : String(state.currentCycle);
  const pending =
    state.lastCycleStart === null ? "-" : String(state.lastCycleStart);
  const entityProgress = `${Math.min(state.entitiesTotal, 100)}/100`;
  const costProgress = `$${state.costUsd.toFixed(2)}/$3.00`;
  const wallProgress = `${state.wallS.toFixed(0)}s/600s`;

  return (
    <Box
      borderStyle="round"
      borderColor="cyan"
      flexDirection="column"
      paddingX={1}
    >
      <Text bold color="cyan">
        Plan
      </Text>
      <Text>
        cycle: <Text color="yellow">{cycle}</Text> / seen{" "}
        <Text color="yellow">{state.cycles}</Text>
      </Text>
      <Text>
        pending tasks: <Text color="yellow">{pending}</Text>
      </Text>
      <Text>
        entities: <Text color="green">{state.entitiesTotal}</Text>
      </Text>
      <Text>
        subagent calls: <Text color="magenta">{state.subagentCalls}</Text>
      </Text>
      <Text dimColor>acceptance</Text>
      <Text>
        entities <Text color="green">{entityProgress}</Text> | cost{" "}
        <Text color={state.costUsd <= 3 ? "green" : "red"}>{costProgress}</Text>
      </Text>
      <Text>
        wall <Text color={state.wallS <= 600 ? "green" : "red"}>{wallProgress}</Text>
      </Text>
    </Box>
  );
}
