import ts from "typescript";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
const source = readFileSync(
  new URL("../src/robot-assets.ts", import.meta.url),
  "utf8",
);
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
});
const {
  intentForRobot,
  robotForBackendIntent,
  routeRobot,
  routeMode,
  routeName,
  MODE_ROBOTS,
  ROBOT_FRAMES,
} = await import(
    "data:text/javascript;base64," + Buffer.from(outputText).toString("base64")
  );
for (const phrase of [
  "calendar",
  "schedule",
  "meeting",
  "appointment",
  "date",
  "tomorrow",
  "today",
  "at 5 PM",
  "event",
])
  assert.equal(intentForRobot(phrase), "calendar");
for (const phrase of [
  "remind me",
  "reminder",
  "remember",
  "task",
  "todo",
  "don't forget",
  "set a reminder",
])
  assert.equal(intentForRobot(phrase), "reminder");
for (const phrase of [
  "summarize",
  "summary",
  "summarizer",
  "summarise",
  "TLDR",
  "shorten this",
  "give me the key points",
  "Summarize this reminder for tomorrow",
])
  assert.equal(intentForRobot(phrase), "summarizer");
assert.equal(intentForRobot("Hello there"), "idle");
assert.equal(robotForBackendIntent("summary"), "summarizer");
assert.equal(robotForBackendIntent("note"), "reminder");
// A released prompt must switch the artwork to the global robot whatever mode
// was selected, and a local answer must not claim the global robot.
assert.equal(routeRobot("global", "Privacy"), MODE_ROBOTS["Global"]);
assert.equal(routeRobot("global", "Default"), MODE_ROBOTS["Global"]);
assert.equal(routeRobot("local", "Privacy"), MODE_ROBOTS["Privacy"]);
assert.equal(routeRobot("local", "Default"), MODE_ROBOTS["Default"]);
assert.equal(routeRobot(undefined, "Global"), MODE_ROBOTS["Global"]);
assert.equal(routeMode("global", "Privacy"), "Global");
assert.equal(routeMode("local", "Privacy"), "Privacy");
assert.equal(routeName("global"), "Global model");
assert.equal(routeName("local"), "Local model");
assert.equal(routeName(undefined), "Local model");
const assets = [
  ...Object.values(MODE_ROBOTS),
  ...Object.values(ROBOT_FRAMES).flat(),
];
assert.equal(assets.length, 18);
for (const asset of assets) {
  const bytes = readFileSync(new URL("../public/" + asset, import.meta.url));
  assert.equal(bytes.subarray(8, 12).toString(), "WEBP");
}
for (const family of ["calendar", "reminder", "summarizer"])
  assert.equal(
    readFileSync(
      new URL(
        "../public/brand/robots/" + family + "-sequence.gif",
        import.meta.url,
      ),
    )
      .subarray(0, 3)
      .toString(),
    "GIF",
  );
const manifest = JSON.parse(
  readFileSync(
    new URL("../public/brand/robots/manifest.json", import.meta.url),
  ),
);
assert.equal(manifest.files.length, 18);
assert.ok(manifest.files.every((file) => file.source_sha256.length === 64));
console.log(
  "PASS: all trigger families, summary precedence, 18 ZIP-derived WebP assets, 3 GIFs and provenance manifest.",
);
