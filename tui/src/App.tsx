/**
 * Top-level Ink App for the Researcher cockpit.
 *
 * Wires a store, an async event source (currently JSONL replay), and
 * the four panels: PlanPanel, AgentDagPanel, KnowledgePanel, StatusBar.
 *
 * The event pump is kicked off once in useEffect and dispatches each
 * parsed event into the store. React's useSyncExternalStore keeps the
 * UI in lockstep with the store without introducing extra deps.
 */

import React from "react";
import { Box, useApp } from "ink";

import { PlanPanel } from "./panels/PlanPanel.js";
import { AgentDagPanel } from "./panels/AgentDagPanel.js";
import { KnowledgePanel } from "./panels/KnowledgePanel.js";
import { StatusBar } from "./panels/StatusBar.js";
import { createStore, type AppState, type Store } from "./store.js";
import type { Event } from "./transport/types.js";

export interface AppProps {
  /** Async iterable of events to fold into the store. */
  eventSource: () => AsyncIterable<Event>;
  /** Whether the process should exit when the event stream ends. */
  exitOnComplete?: boolean;
}

function useStoreState(store: Store): AppState {
  return React.useSyncExternalStore(
    React.useCallback((cb) => store.subscribe(cb), [store]),
    React.useCallback(() => store.getState(), [store]),
    React.useCallback(() => store.getState(), [store]),
  );
}

export function App({
  eventSource,
  exitOnComplete = true,
}: AppProps): React.JSX.Element {
  const storeRef = React.useRef<Store | null>(null);
  if (storeRef.current === null) {
    storeRef.current = createStore();
  }
  const store = storeRef.current;
  const state = useStoreState(store);
  const { exit } = useApp();

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        for await (const event of eventSource()) {
          if (cancelled) return;
          store.dispatch(event);
        }
      } catch (err) {
        // Surface the error through the store's log channel if we can;
        // otherwise just print and bail.
        // eslint-disable-next-line no-console
        console.error("[tui] event pump error:", (err as Error).message);
      }
      if (exitOnComplete && !cancelled) {
        // Give React one tick to flush the final state before exiting.
        setTimeout(() => exit(), 50);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eventSource is a closure; we intentionally only run once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Box flexDirection="column">
      <Box flexDirection="row">
        <Box flexDirection="column" marginRight={1} width={32}>
          <PlanPanel state={state} />
          <KnowledgePanel state={state} />
        </Box>
        <Box flexDirection="column" flexGrow={1}>
          <AgentDagPanel state={state} />
        </Box>
      </Box>
      <StatusBar state={state} />
    </Box>
  );
}
