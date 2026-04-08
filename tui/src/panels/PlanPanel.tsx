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
    </Box>
  );
}
