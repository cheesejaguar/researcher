#!/usr/bin/env node
/**
 * researcher-tui: Ink-based cockpit for watching Researcher runs.
 *
 * Usage:
 *   researcher-tui --replay <path/to/events.jsonl>   # replay a finished run
 *   researcher-tui --socket <path/to/sock>           # live attach
 *   researcher-tui --replay <events.jsonl> --socket <sock>
 */

import React from "react";
import { render } from "ink";

import { App } from "./App.js";
import { readJsonlEvents } from "./transport/jsonl.js";
import { readSocketEvents } from "./transport/socket.js";
import type { Event } from "./transport/types.js";

interface CliArgs {
  replay?: string;
  socket?: string;
  help?: boolean;
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--replay") {
      args.replay = argv[++i];
    } else if (a === "--socket") {
      args.socket = argv[++i];
    } else if (a === "-h" || a === "--help") {
      args.help = true;
    }
  }
  return args;
}

function printHelp(): void {
  const msg = [
    "researcher-tui - Ink cockpit for Researcher runs",
    "",
    "Usage:",
    "  researcher-tui --replay <events.jsonl>",
    "  researcher-tui --socket <path>",
    "  researcher-tui --replay <events.jsonl> --socket <path>",
    "",
    "Flags:",
    "  --replay <path>   Replay a pre-recorded JSONL event file",
    "  --socket <path>   Attach to a live orchestrator socket",
    "  -h, --help        Show this message",
  ].join("\n");
  // eslint-disable-next-line no-console
  console.log(msg);
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));

  if (args.help || (!args.replay && !args.socket)) {
    printHelp();
    if (!args.help) process.exitCode = 1;
    return;
  }

  let eventSource: () => AsyncIterable<Event>;
  if (args.replay && args.socket) {
    const replayPath = args.replay;
    const socketPath = args.socket;
    eventSource = async function* replayThenSocket(): AsyncIterable<Event> {
      yield* readJsonlEvents(replayPath);
      yield* readSocketEvents({ socketPath });
    };
  } else if (args.replay) {
    const path = args.replay;
    eventSource = () => readJsonlEvents(path);
  } else if (args.socket) {
    const socketPath = args.socket;
    eventSource = () => readSocketEvents({ socketPath });
  } else {
    printHelp();
    process.exitCode = 1;
    return;
  }

  render(<App eventSource={eventSource} exitOnComplete={true} />);
}

main();
