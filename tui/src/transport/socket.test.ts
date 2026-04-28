import { describe, expect, it } from "vitest";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";

import { readSocketEvents } from "./socket.js";

describe("readSocketEvents", () => {
  it("parses newline-delimited events from a Unix socket", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "researcher-tui-"));
    const socketPath = path.join(dir, "events.sock");
    const event = {
      type: "cycle_start",
      seq: 0,
      ts: "2026-04-08T12:00:00Z",
      run_id: "run-test",
      payload: { cycle: 1, pending_tasks: 2 },
    };
    const server = net.createServer((socket) => {
      socket.write(`${JSON.stringify(event)}\n`);
      socket.end();
    });
    await new Promise<void>((resolve) => server.listen(socketPath, resolve));

    const events = [];
    try {
      for await (const parsed of readSocketEvents({ socketPath })) {
        events.push(parsed);
      }
    } finally {
      server.close();
      fs.rmSync(dir, { recursive: true, force: true });
    }

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("cycle_start");
    expect(events[0].payload.pending_tasks).toBe(2);
  });
});
