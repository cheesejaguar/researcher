/**
 * Unix socket transport placeholder.
 *
 * Wave 1-F ships only the JSONL replay transport. The live-attach
 * transport that consumes the Python side's bounded queue lives here
 * and will be fleshed out in Wave 2. For now we expose the same
 * AsyncGenerator<Event> contract so the App can be pointed at either
 * transport interchangeably.
 *
 * When we implement this for real, the shape will look like:
 *
 *   const socket = net.createConnection(socketPath);
 *   for await (const chunk of socket) {
 *     for (const line of framer.feed(chunk)) {
 *       yield parseEvent(JSON.parse(line));
 *     }
 *   }
 */

import type { Event } from "./types.js";

export interface SocketTransportOptions {
  socketPath: string;
  /** Optional abort signal to cleanly tear down the reader. */
  signal?: AbortSignal;
}

/**
 * Placeholder reader: always throws when invoked. Callers should check
 * the CLI flags and prefer readJsonlEvents until Wave 2 lands.
 */
export async function* readSocketEvents(
  _opts: SocketTransportOptions,
): AsyncGenerator<Event, void, void> {
  throw new Error(
    "socket transport not implemented yet; use --replay <events.jsonl> for now",
  );
  // Unreachable, kept so TypeScript infers the generator type correctly.
  // eslint-disable-next-line @typescript-eslint/no-unreachable
  yield undefined as unknown as Event;
}
