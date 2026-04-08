/**
 * JSONL transport: read a pre-recorded `events.jsonl` file and yield
 * parsed Events one at a time.
 *
 * Used by --replay mode in the TUI and by the store.test.ts fixture
 * harness. The reader is line-buffered via split("\n") for simplicity;
 * if the files ever get huge we can swap in a streaming line reader
 * without changing the iterator interface.
 */

import { parseEvent, type Event } from "./types.js";

export async function* readJsonlEvents(
  filePath: string,
): AsyncGenerator<Event, void, void> {
  const fs = await import("node:fs/promises");
  let content: string;
  try {
    content = await fs.readFile(filePath, "utf-8");
  } catch (err) {
    throw new Error(
      `failed to read replay file ${filePath}: ${(err as Error).message}`,
    );
  }

  const lines = content.split("\n").filter((l) => l.trim().length > 0);
  for (const line of lines) {
    let raw: unknown;
    try {
      raw = JSON.parse(line);
    } catch {
      // Skip malformed lines silently; a partial JSONL write during a
      // live run is not a fatal error for replay mode.
      continue;
    }
    try {
      yield parseEvent(raw);
    } catch {
      // Unknown event type or schema mismatch: skip this line rather than
      // aborting the entire replay. The TUI should be forward-compatible.
      continue;
    }
  }
}
