/** Deterministic service-worker payload/click tests; no vendor delivery simulated as real. */
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import assert from "node:assert/strict";
const listeners = {};
const displayed = [];
const opened = [];
const pending = [];
const self = {
  addEventListener: (name, fn) => {
    listeners[name] = fn;
  },
  skipWaiting: async () => {},
  location: { origin: "https://workspace.example" },
  registration: {
    showNotification: async (title, options) =>
      displayed.push({ title, options }),
  },
  clients: {
    claim: async () => {},
    matchAll: async () => [],
    openWindow: async (url) => opened.push(url),
  },
};
const source = readFileSync(
  new URL("../public/push-worker.js", import.meta.url),
  "utf8",
);
runInNewContext(source, { self, URL });
const event = { waitUntil: (promise) => pending.push(promise) };
listeners.push({
  ...event,
  data: {
    json: () => ({
      workspace: "Calendar",
      body: "PRIVATE TITLE",
      tag: "ppda-1",
    }),
  },
});
await Promise.all(pending);
assert.equal(displayed[0].title, "PPDA · Event starting");
assert.equal(
  displayed[0].options.body,
  "Open your workspace to view the details.",
);
assert.ok(!JSON.stringify(displayed).includes("PRIVATE"));
assert.equal(displayed[0].options.tag, "ppda-1");
listeners.push({
  ...event,
  data: {
    json: () => {
      throw Error("bad JSON");
    },
  },
});
listeners.notificationclick({
  ...event,
  notification: {
    close: () => {},
    data: { workspace: "https://evil.example/" },
  },
});
await Promise.all(pending);
assert.equal(displayed[1].title, "PPDA · Reminder due");
assert.deepEqual(opened, ["https://workspace.example/?workspace=Reminders"]);
assert.equal(listeners.fetch, undefined);
assert.ok(!source.includes("caches."));
console.log(
  "PASS: service-worker generic payloads, malformed-message fallback, same-origin clicks, no private offline cache.",
);
