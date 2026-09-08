/** End-to-end browser tests against an isolated real backend + production bundle. */
import { chromium as playwright, expect } from "@playwright/test";
import chromium from "@sparticuz/chromium";
import assert from "node:assert/strict";
import {
  mkdtempSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  rmSync,
  existsSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { brotliDecompressSync } from "node:zlib";
import { execFileSync, spawn } from "node:child_process";
import { createServer, request } from "node:http";
import { randomBytes, generateKeyPairSync } from "node:crypto";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const temp = mkdtempSync(join(tmpdir(), "ppda-e2e-"));
const screenshots = process.env.PPDA_SCREENSHOT_DIR || temp;
mkdirSync(screenshots, { recursive: true });
writeFileSync(
  join(temp, "libraries.tar"),
  brotliDecompressSync(
    readFileSync(
      new URL(
        "../bin/al2023.tar.br",
        import.meta.resolve("@sparticuz/chromium"),
      ),
    ),
  ),
);
execFileSync("tar", ["-xf", join(temp, "libraries.tar"), "-C", temp]);
const vapidKey = generateKeyPairSync("ec", { namedCurve: "prime256v1" });
writeFileSync(
  join(temp, "vapid.pem"),
  vapidKey.privateKey.export({ type: "pkcs8", format: "pem" }),
  { mode: 0o600 },
);
const env = {
  ...process.env,
  DATABASE_URL: "sqlite:///" + join(temp, "e2e.db"),
  JWT_SECRET: randomBytes(48).toString("hex"),
  AES_MASTER_KEY: randomBytes(32).toString("base64"),
  COOKIE_SECURE: "false",
  COOKIE_SAMESITE: "lax",
  COOKIE_PARTITIONED: "false",
  PIPELINE_ENABLED: "false",
  OPENAI_API_KEY: "",
  LOCAL_SUMMARY_MODEL: "extractive", // Explicit offline fixture; neural path has separate tests.
  OLLAMA_URL: "",
  VAPID_PRIVATE_KEY_PATH: join(temp, "vapid.pem"),
  SNIPS_DIR: join(temp, "snips"),
};
execFileSync(join(root, ".venv/bin/alembic"), ["upgrade", "head"], {
  cwd: root,
  env,
  stdio: "pipe",
});
const backend = spawn(
  join(root, ".venv/bin/python"),
  [
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8011",
    "--no-access-log",
  ],
  { cwd: root, env, stdio: ["ignore", "pipe", "pipe"] },
);
let browser, server;
try {
  await new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error("Backend startup timed out")),
      20000,
    );
    backend.stderr.on("data", (chunk) => {
      if (chunk.toString().includes("Uvicorn running")) {
        clearTimeout(timer);
        resolve();
      }
    });
    backend.on("exit", (code) => {
      clearTimeout(timer);
      reject(new Error("Backend exited: " + code));
    });
  });
  const dist = join(root, "frontend/dist/ppda/browser");
  server = createServer((req, res) => {
    if (req.url.startsWith("/api/") || req.url === "/health") {
      const upstream = request(
        {
          hostname: "127.0.0.1",
          port: 8011,
          path: req.url,
          method: req.method,
          headers: req.headers,
        },
        (r) => {
          res.writeHead(r.statusCode, r.headers);
          r.pipe(res);
        },
      );
      upstream.on("error", () => {
        res.writeHead(502);
        res.end();
      });
      req.pipe(upstream);
      return;
    }
    let path = resolve(dist, "." + decodeURIComponent(req.url.split("?")[0]));
    if (!path.startsWith(dist)) {
      res.writeHead(403);
      res.end();
      return;
    }
    if (req.url === "/" || !existsSync(path)) path = join(dist, "index.html");
    const types = {
      ".html": "text/html",
      ".js": "text/javascript",
      ".css": "text/css",
      ".svg": "image/svg+xml",
      ".png": "image/png",
      ".woff2": "font/woff2",
      ".md": "text/plain",
    };
    res.writeHead(200, {
      "Content-Type": types[extname(path)] || "application/octet-stream",
    });
    res.end(readFileSync(path));
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const base = "http://127.0.0.1:" + server.address().port;
  browser = await playwright.launch({
    executablePath: await chromium.executablePath(),
    args: chromium.args,
    headless: true,
    env: {
      ...process.env,
      LD_LIBRARY_PATH:
        join(temp, "lib") + ":" + (process.env.LD_LIBRARY_PATH || ""),
    },
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));

  page.on("dialog", (dialog) => dialog.accept());
  await page.goto(base);
  await page
    .getByRole("button", { name: "New here? Create an account" })
    .click();
  await page.getByLabel("What should we call you?").fill("Sam");
  await page.getByLabel("Email address").fill("sam-smoke@example.com");
  await page
    .getByLabel("Password", { exact: true })
    .fill("a-long-test-password");
  await page.getByRole("checkbox").check();
  await page
    .getByRole("button", { name: "Create account", exact: false })
    .click();
  await page
    .getByRole("heading", { name: "What’s on your mind, Sam?" })
    .waitFor();
  await page.screenshot({
    path: join(screenshots, "workspace-light.png"),
    fullPage: true,
  });
  await expect(page.locator(".robot-welcome .robot-art")).toHaveAttribute(
    "src",
    /default_mode_robot_blue.webp$/,
  );
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.getByRole("button", { name: "Select assistant mode" }).click();
  await page
    .getByRole("button", { name: "Privacy mode Local processing only" })
    .click();
  await expect(page.locator(".robot-welcome .robot-art")).toHaveAttribute(
    "src",
    /privacy_mode_robot_green.webp$/,
  );
  await page.screenshot({
    path: join(screenshots, "privacy-dark.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Select assistant mode" }).click();
  await page
    .getByRole("button", {
      name: "Global mode External LLM · requires consent",
    })
    .click();
  await expect(page.locator(".robot-welcome .robot-art")).toHaveAttribute(
    "src",
    /global_model_robot_soft_red.webp$/,
  );
  await page.getByRole("button", { name: "Select assistant mode" }).click();
  await page
    .getByRole("button", { name: "Privacy mode Local processing only" })
    .click();
  // Hold the real response to verify intermediate frames cannot claim success.
  await page.route(
    "**/api/v1/assistant/command",
    async (route) => {
      const response = await route.fetch();
      await new Promise((r) => setTimeout(r, 1500));
      await route.fulfill({ response });
    },
    { times: 1 },
  );
  await page
    .getByRole("textbox", { name: "Message your assistant" })
    .fill("Remind me to call Rahul tomorrow at 5 pm");
  await page.getByRole("button", { name: "Send message" }).click();
  const activity = page.locator(".assistant-activity");
  await expect(activity.locator("figure")).toHaveAttribute(
    "data-robot-phase",
    "processing",
  );
  await expect(activity.locator(".robot-art")).toHaveAttribute(
    "src",
    /reminder_notepad_appears.webp$/,
  );
  await expect(activity.locator(".robot-art")).not.toHaveAttribute(
    "src",
    /saved|confirmed|summary_ready/,
  );
  await page.getByRole("button", { name: "Review & save reminders" }).click();
  await expect(page.getByRole("dialog").locator("figure")).toHaveAttribute(
    "data-robot-phase",
    "review",
  );
  await expect(
    page.getByRole("dialog").locator(".robot-art"),
  ).not.toHaveAttribute("src", /saved|confirmed/);
  await expect(page.getByLabel("Title", { exact: true })).toHaveValue(
    "call Rahul",
  );
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Add reminder" })
    .click();
  await page.getByRole("heading", { name: "call Rahul" }).waitFor();
  await expect(page.locator(".workspace-art .robot-art")).toHaveAttribute(
    "src",
    /reminder_saved.webp$/,
  );
  await page.screenshot({
    path: join(screenshots, "reminder-saved.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Complete", exact: true }).click();
  await page.getByRole("button", { name: "Undo", exact: true }).waitFor();
  const nav = page.getByRole("navigation", { name: "Main navigation" });
  await nav.getByRole("button", { name: "Notes", exact: true }).click();
  await page.getByRole("button", { name: "Add note" }).click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Add note" })
    .click();
  await page.getByRole("alert").getByText("Please add a title.").waitFor();
  await page.getByLabel("Title", { exact: true }).fill("Encrypted sample note");
  await page
    .getByLabel("Your note")
    .fill("This content must not appear in plaintext in the database.");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Add note" })
    .click();
  await page
    .getByRole("button", { name: "Edit Encrypted sample note" })
    .click();
  await page
    .getByLabel("Title", { exact: true })
    .fill("Updated encrypted note");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Save note" })
    .click();
  await page.reload();
  await nav.getByRole("button", { name: "Notes", exact: true }).click();
  await page.getByRole("heading", { name: "Updated encrypted note" }).waitFor();
  await page.getByRole("button", { name: "Add note" }).click();
  await page.getByLabel("Title", { exact: true }).fill("Failure-path fixture");
  await page.route(
    "**/api/v1/entries/Notes",
    (route) =>
      route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Test save failure" }),
      }),
    { times: 1 },
  );
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Add note" })
    .click();
  await page.getByRole("alert").getByText("Test save failure").waitFor();
  await expect(page.getByRole("dialog").locator("figure")).toHaveAttribute(
    "data-robot-phase",
    "error",
  );
  await expect(
    page.getByRole("dialog").locator(".robot-art"),
  ).not.toHaveAttribute("src", /saved|confirmed/);
  await page.getByRole("button", { name: "Close dialog" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page
    .getByRole("navigation", { name: "Account navigation" })
    .getByRole("button", { name: "Settings" })
    .click();
  await page.getByLabel("Federated learning participation").check();
  await page.getByLabel("Target epsilon").fill("5");
  await page.getByLabel("Reduce motion").check();
  await page
    .getByRole("button", { name: "Save preferences & consent" })
    .click();
  await page
    .getByText("Awaiting 3 validated examples", { exact: true })
    .waitFor();
  await page.getByRole("button", { name: "Verify audit chain" }).click();
  await page.getByText(/entries verified/).waitFor();
  await nav.getByRole("button", { name: "Assistant", exact: true }).click();
  await page.evaluate(() => {
    window.__robotTrace = [];
    const observer = new MutationObserver(() => {
      const f = document.querySelector(".assistant-activity figure");
      const img = f?.querySelector("img");
      if (f && img)
        window.__robotTrace.push({
          phase: f.getAttribute("data-robot-phase"),
          src: img.getAttribute("src"),
        });
    });
    observer.observe(document.querySelector("main"), {
      subtree: true,
      attributes: true,
      childList: true,
    });
  });
  await page
    .getByRole("textbox", { name: "Message your assistant" })
    .fill(
      "Summarize: Apples contain fibre. Apples are fruit. Pears are fruit too.",
    );
  await page.getByRole("button", { name: "Send message" }).click();
  await page.getByText(/• Apples/).waitFor();
  const trace = await page.evaluate(() => window.__robotTrace);
  assert.ok(
    trace.some(
      (frame) =>
        frame.phase === "success" &&
        frame.src.endsWith("summarizer_summary_ready.webp"),
    ),
    JSON.stringify(trace),
  );
  assert.ok(
    trace
      .filter((frame) => frame.phase === "success")
      .every((frame) => frame.src.endsWith("summarizer_summary_ready.webp")),
    "Reduced motion must use only the completion still",
  );
  await expect(page.locator(".message-robot").last()).toHaveAttribute(
    "src",
    /summarizer_summary_ready.webp$/,
  );
  await page.screenshot({
    path: join(screenshots, "summary-ready.png"),
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Confirm label & contribute" })
    .last()
    .click();
  await page.getByRole("button", { name: "Queued locally ✓" }).waitFor();
  // Assistant CRUD remains side-effect-free until its specific confirmation is clicked.
  async function sendCommand(text) {
    await nav.getByRole("button", { name: "Assistant", exact: true }).click();
    await page
      .getByRole("textbox", { name: "Message your assistant" })
      .fill(text);
    await page.getByRole("button", { name: "Send message" }).click();
  }
  const calendarRows = () =>
    page.evaluate(async () => (await fetch("/api/v1/entries/Calendar")).json());
  await sendCommand('Create an event "Browser CRUD review" tomorrow at 10 am');
  await page
    .getByRole("button", { name: "Review & save calendar" })
    .last()
    .click();
  assert.equal(
    (await calendarRows()).filter((i) => i.title === "Browser CRUD review")
      .length,
    0,
  );
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Add event" })
    .click();
  await page
    .getByRole("heading", { name: "Browser CRUD review", exact: true })
    .waitFor();
  const originalEvent = (await calendarRows()).find(
    (i) => i.title === "Browser CRUD review",
  );
  await sendCommand(
    'Reschedule event "Browser CRUD review" to 2099-06-10T09:45',
  );
  await page
    .getByRole("button", { name: "Review & update calendar" })
    .last()
    .click();
  await expect(page.getByLabel("Date and time", { exact: true })).toHaveValue(
    "2099-06-10T09:45",
  );
  assert.equal(
    (await calendarRows()).find((i) => i.id === originalEvent.id).detail,
    originalEvent.detail,
  );
  await page.getByRole("button", { name: "Close dialog" }).click();
  await nav.getByRole("button", { name: "Assistant", exact: true }).click();
  await page
    .getByRole("button", { name: "Review & update calendar" })
    .last()
    .click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Save event" })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  assert.equal(
    (await calendarRows()).find((i) => i.id === originalEvent.id).detail,
    "2099-06-10T09:45",
  );
  await sendCommand('Delete event "Browser CRUD review"');
  await page
    .getByRole("button", { name: "Review & delete calendar" })
    .last()
    .click();
  await expect(page.getByLabel("Title", { exact: true })).toHaveJSProperty(
    "readOnly",
    true,
  );
  assert.ok((await calendarRows()).some((i) => i.id === originalEvent.id));
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Confirm delete event" })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  assert.ok(!(await calendarRows()).some((i) => i.id === originalEvent.id));

  // Mock browser permission/subscription APIs only; exercise real service-worker
  // registration and authenticated enrollment/revocation HTTP. No vendor push sent.
  const browserKey = generateKeyPairSync("ec", { namedCurve: "prime256v1" })
    .publicKey.export({ type: "spki", format: "der" })
    .subarray(-65)
    .toString("base64url");
  await page.evaluate(
    ({ publicKey, auth }) => {
      window.__permissionCalls = 0;
      window.__grantPush = false;
      window.__fakeSubscription = null;
      Object.defineProperty(Notification, "requestPermission", {
        configurable: true,
        value: async () => {
          window.__permissionCalls++;
          return window.__grantPush ? "granted" : "denied";
        },
      });
      PushManager.prototype.getSubscription = async () =>
        window.__fakeSubscription;
      PushManager.prototype.subscribe = async function (options) {
        if (
          !options.userVisibleOnly ||
          options.applicationServerKey.length !== 65
        )
          throw new Error("Bad push enrollment options");
        const value = {
          endpoint: "https://fcm.googleapis.com/fcm/send/browser-smoke",
          expirationTime: null,
          keys: { p256dh: publicKey, auth },
        };
        window.__fakeSubscription = {
          ...value,
          toJSON: () => value,
          unsubscribe: async () => {
            window.__fakeSubscription = null;
            return true;
          },
        };
        return window.__fakeSubscription;
      };
    },
    { publicKey: browserKey, auth: randomBytes(16).toString("base64url") },
  );
  await page
    .getByRole("navigation", { name: "Account navigation" })
    .getByRole("button", { name: "Settings", exact: true })
    .click();
  await page
    .getByRole("heading", { name: "Scheduled browser notifications" })
    .waitFor();
  await expect(
    page.getByRole("button", { name: "Enable browser push" }),
  ).toBeEnabled();
  assert.equal(await page.evaluate(() => window.__permissionCalls), 0);
  await page.getByRole("button", { name: "Enable browser push" }).click();
  await page
    .getByText("Notifications are blocked.", { exact: false })
    .waitFor();
  const subscriptions = () =>
    page.evaluate(
      async () =>
        (await (await fetch("/api/v1/notifications/status")).json())
          .subscriptions,
    );
  assert.equal(await subscriptions(), 0);
  await page.evaluate(() => {
    window.__grantPush = true;
  });
  await page.getByRole("button", { name: "Enable browser push" }).click();
  await page.getByText("Enabled on this browser", { exact: true }).waitFor();
  assert.equal(await subscriptions(), 1);
  assert.ok(
    await page.evaluate(async () =>
      (
        await navigator.serviceWorker.getRegistration("/")
      )?.active?.scriptURL.endsWith("/push-worker.js"),
    ),
  );
  await page.getByRole("button", { name: "Disable on this browser" }).click();
  await expect(
    page.getByRole("button", { name: "Enable browser push" }),
  ).toBeEnabled();
  assert.equal(await subscriptions(), 0);
  await expect(
    page.getByText("Default / Privacy summaries:", { exact: false }),
  ).toContainText("no LLM");
  await nav.getByRole("button", { name: "Assistant", exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: join(screenshots, "mobile-robot.png"),
    fullPage: true,
  });
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  );
  await page
    .getByRole("navigation", { name: "Account navigation" })
    .getByRole("button", { name: "Profile" })
    .click();
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await page
    .getByRole("heading", { name: /Welcome back|Make yourself at home/ })
    .waitFor();
  if (
    await page
      .getByRole("button", { name: "Already have an account? Sign in" })
      .count()
  )
    await page
      .getByRole("button", { name: "Already have an account? Sign in" })
      .click();
  await page
    .getByLabel("Password", { exact: true })
    .fill("a-long-test-password");
  await page.getByRole("button", { name: /^Sign in/ }).click();
  await page
    .getByRole("heading", { name: "What’s on your mind, Sam?" })
    .waitFor();
  assert.deepEqual(errors, []);
  console.log(
    "PASS: real registration/login/logout, encrypted persistent CRUD, consent, local ONNX commands, review-before-save/update/delete, mocked browser push permission/enrollment/revocation with real service-worker registration, summary provenance, learning queue, saved budget settings, audit, themes, contextual robot assets, gated success/error frames, reduced motion and mobile layout.",
  );
} finally {
  if (browser) await browser.close();
  if (server) await new Promise((r) => server.close(r));
  backend.kill("SIGTERM");
  await new Promise((r) => {
    backend.once("exit", r);
    setTimeout(r, 5000);
  });
  rmSync(temp, { recursive: true, force: true });
}
