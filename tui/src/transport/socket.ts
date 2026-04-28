import net from "node:net";

import { parseEvent, type Event } from "./types.js";

export interface SocketTransportOptions {
  socketPath: string;
  /** Optional abort signal to cleanly tear down the reader. */
  signal?: AbortSignal;
}

export async function* readSocketEvents(
  opts: SocketTransportOptions,
): AsyncGenerator<Event, void, void> {
  const socket = net.createConnection(opts.socketPath);
  socket.setEncoding("utf8");
  let buffer = "";
  let ended = false;
  let error: Error | null = null;
  const pending: Event[] = [];
  let notify: (() => void) | null = null;

  const wake = (): void => {
    if (notify) {
      notify();
      notify = null;
    }
  };

  const abort = (): void => {
    socket.destroy();
    ended = true;
    wake();
  };

  opts.signal?.addEventListener("abort", abort, { once: true });

  socket.on("data", (chunk: string) => {
    buffer += chunk;
    for (;;) {
      const idx = buffer.indexOf("\n");
      if (idx === -1) break;
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;
      pending.push(parseEvent(JSON.parse(line)));
    }
    wake();
  });
  socket.on("end", () => {
    ended = true;
    wake();
  });
  socket.on("close", () => {
    ended = true;
    wake();
  });
  socket.on("error", (err) => {
    error = err;
    ended = true;
    wake();
  });

  try {
    while (!ended || pending.length > 0) {
      while (pending.length > 0) {
        const next = pending.shift();
        if (next) yield next;
      }
      if (error) throw error;
      if (ended) break;
      await new Promise<void>((resolve) => {
        notify = resolve;
      });
    }
  } finally {
    opts.signal?.removeEventListener("abort", abort);
    socket.destroy();
  }
}
