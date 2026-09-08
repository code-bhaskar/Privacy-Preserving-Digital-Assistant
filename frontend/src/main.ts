import {
  Component,
  Input,
  ChangeDetectorRef,
  provideZonelessChangeDetection,
} from "@angular/core";
import { bootstrapApplication } from "@angular/platform-browser";
import { FormsModule } from "@angular/forms";
import { RobotComponent } from "./robot.component";
import {
  intentForRobot,
  robotForBackendIntent,
  modeRobot,
  ROBOT_FRAMES,
  RobotIntent,
  RobotPhase,
} from "./robot-assets";

@Component({
  selector: "app-icon",
  standalone: true,
  template: `<svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.65"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <path [attr.d]="paths[name] || paths['assistant']" />
  </svg>`,
  styles: [
    `
      :host {
        display: inline-flex;
        width: 21px;
        height: 21px;
        flex-shrink: 0;
      }
      svg {
        width: 100%;
        height: 100%;
      }
    `,
  ],
})
class Icon {
  @Input() name = "assistant";
  paths: Record<string, string> = {
    assistant:
      "M5 6h14a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H8l-5 3V8a2 2 0 0 1 2-2Zm3 6h.01M16 12h.01M9 16h6M12 6V2",
    calendar:
      "M8 2v4m8-4v4M3 10h18M5 4h14a2 2 0 0 1 2 2v14H3V6a2 2 0 0 1 2-2Zm3 10h2m4 0h2m-8 3h2",
    reminders: "M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4",
    notes: "M14 3H5v18h14V8l-5-5Zm0 0v5h5M8 12h8m-8 4h6",
    profile: "M16 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0ZM4 21v-2a8 8 0 0 1 16 0v2",
    settings:
      "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM9 3h6l1 3 3 1 2 5-2 5-3 1-1 3H9l-1-3-3-1-2-5 2-5 3-1 1-3Z",
    shield: "M12 2 3 6v6c0 5 9 10 9 10s9-5 9-10V6l-9-4Zm-4 10 3 3 5-6",
    sun: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM12 2v2m0 16v2M2 12h2m16 0h2M5 5l1 1m12 12 1 1M5 19l1-1M18 6l1-1",
    arrow: "M12 19V5m-6 6 6-6 6 6",
    plus: "M12 5v14M5 12h14",
    logout: "M9 3H3v18h6m7-14 5 5-5 5m-9-5h14",
    check: "m5 12 4 4L19 6",
    spark: "m12 2 3 7 7 3-7 3-3 7-3-7-7-3 7-3 3-7",
  };
}
type Item = {
  id: number;
  title: string;
  detail: string;
  done: boolean;
  fired?: boolean;
  version?: string;
};
type Proposal = {
  kind: string;
  title: string;
  detail: string;
  operation?: "create" | "update" | "delete";
  id?: number;
  version?: string;
  done?: boolean;
};
type Message = {
  robotMode?: string;
  role: string;
  text: string;
  location?: string;
  proposal?: Proposal;
  source?: string;
  intent?: string;
  confidence?: number;
  taught?: boolean;
  explanation?: { token: string; contribution: number }[];
};
type Preferences = {
  assistant: boolean;
  calendar: boolean;
  notes: boolean;
  summary: boolean;
  training: boolean;
  cloud: boolean;
  epsilon: number;
  reduced: boolean;
  timezone: string;
  mode: string;
};
type Learning = {
  state: string;
  queued: number;
  epsilon_spent: number;
  delta_spent: number;
  remaining: number;
  model_version: number;
  history: { id: number; status: string; detail: string }[];
  mechanism: string;
  scope: string;
  pipeline_enabled: boolean;
};
@Component({
  selector: "app-root",
  standalone: true,
  imports: [FormsModule, Icon, RobotComponent],
  templateUrl: "./app.html",
})
class App {
  dark = localStorage.getItem("ppda-theme")
    ? localStorage.getItem("ppda-theme") === "dark"
    : window.matchMedia("(prefers-color-scheme: dark)").matches;
  entered = false;
  checking = true;
  authMode = "login";
  email = "";
  password = "";
  authError = "";
  authBusy = false;
  localConsent = false;
  showPassword = false;
  csrf = "";
  tab = "Assistant";
  mode = "Default";
  modeOpen = false;
  draft = "";
  name = "";
  epsilon = 3;
  training = false;
  cloud = false;
  reduced = false;
  localAssistant = false;
  calendarConsent = false;
  notesConsent = false;
  summaryConsent = false;
  timezone = "Asia/Kolkata";
  toast = "";
  adding = false;
  itemTitle = "";
  itemDetail = "";
  modalError = "";
  editingId: number | null = null;
  editingVersion: string | undefined;
  editingDone = false;
  deleting = false;
  pushBusy = false;
  pushActive = false;
  pushBrowserPresent = false;
  pushConfigured = false;
  pushDetail = "";
  pushSupported =
    window.isSecureContext &&
    "serviceWorker" in navigator &&
    "PushManager" in window &&
    "Notification" in window;
  embedded = window.self !== window.top;
  snipsStatus: {
    available: boolean;
    metrics?: {
      validation_accuracy: number;
      validation_samples: number;
      train_samples: number;
    };
  } | null = null;
  saving = false;
  learningText: string | undefined;
  settingsBusy = false;
  private previousFocus: HTMLElement | null = null;
  robotState = "Ready when you are";
  robotIntent: RobotIntent = "idle";
  robotPhase: RobotPhase = "idle";
  private robotReset: ReturnType<typeof setTimeout> | undefined;
  modeRobot = modeRobot;
  readonly robotFrames = ROBOT_FRAMES;
  get workspaceRobotIntent(): RobotIntent {
    return this.tab === "Calendar" ? "calendar" : "reminder";
  }
  get workspaceRobotPhase(): RobotPhase {
    return this.robotIntent === this.workspaceRobotIntent &&
      this.robotPhase !== "idle"
      ? this.robotPhase
      : "context";
  }
  get workspaceRobotLabel() {
    return this.robotIntent === this.workspaceRobotIntent &&
      this.robotPhase !== "idle"
      ? this.robotState
      : this.tab === "Calendar"
        ? "Make time for what matters"
        : this.tab === "Reminders"
          ? "A little less to remember"
          : "A space for your ideas";
  }
  private setRobot(intent: RobotIntent, phase: RobotPhase, label: string) {
    clearTimeout(this.robotReset);
    this.robotIntent = intent;
    this.robotPhase = phase;
    this.robotState = label;
    this.cdr.markForCheck();
  }
  private robotSucceeded(intent: RobotIntent, label: string) {
    this.setRobot(intent, "success", label);
    this.robotReset = setTimeout(
      () => this.setRobot("idle", "idle", "Ready when you are"),
      2600,
    );
  }
  newConversation() {
    if (this.busy) return;
    this.messages = [];
    this.setRobot("idle", "idle", "Ready when you are");
  }
  messageRobot(message: Message) {
    const intent = robotForBackendIntent(message.intent);
    if (
      message.location === "local backend" &&
      intent !== "idle" &&
      !message.text.startsWith("Please include the text")
    )
      return ROBOT_FRAMES[intent][intent === "summarizer" ? 4 : 3];
    return modeRobot(message.robotMode || "Default");
  }

  busy = false;
  nav = ["Assistant", "Calendar", "Reminders", "Notes"];
  secondary = ["Profile", "Settings"];
  messages: Message[] = [];
  collections: Record<string, Item[]> = {
    Calendar: [],
    Reminders: [],
    Notes: [],
  };
  learning: Learning | null = null;
  auditRows: { id: number; action: string; integrity_hash: string }[] = [];
  auditResult = "";
  currentPassword = "";
  newPassword = "";
  runtime: {
    cloud_configured: boolean;
    local_llm: boolean;
    cloud_model: string;
    summarization_engine: string;
    local_summary: { ready: boolean; engine: string; detail: string };
    boundary: string;
    secure_cookie: boolean;
  } | null = null;
  private notified = new Set<number>();
  private generation = 0;
  constructor(private cdr: ChangeDetectorRef) {
    this.applyTheme();
    void this.restore();
    setInterval(() => {
      if (this.entered) void this.poll();
    }, 15000);
  }
  applyTheme() {
    document.documentElement.dataset["theme"] = this.dark ? "dark" : "light";
    localStorage.setItem("ppda-theme", this.dark ? "dark" : "light");
  }
  toggleTheme() {
    this.dark = !this.dark;
    this.applyTheme();
  }
  async api<T>(
    path: string,
    method = "GET",
    body?: unknown,
    headers: Record<string, string> = {},
  ): Promise<T> {
    const response = await fetch("/api/v1" + path, {
      method,
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        ...headers,
        ...(this.csrf ? { "X-CSRF-Token": this.csrf } : {}),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = await response
      .json()
      .catch(() => ({ detail: "Unexpected server response" }));
    if (!response.ok) {
      if (response.status === 401 && this.entered) this.clearSession();
      let detail = data.detail;
      if (Array.isArray(detail))
        detail = detail.map((d: { msg: string }) => d.msg).join("; ");
      throw new Error(typeof detail === "string" ? detail : "Request failed");
    }
    return data as T;
  }
  error(e: unknown) {
    return e instanceof Error
      ? e.message
      : "Operation failed. Please try again.";
  }
  applyPrefs(p: Preferences) {
    this.localAssistant = p.assistant;
    this.calendarConsent = p.calendar;
    this.notesConsent = p.notes;
    this.summaryConsent = p.summary;
    this.training = p.training;
    this.cloud = p.cloud;
    this.epsilon = p.epsilon;
    this.reduced = p.reduced;
    this.timezone = p.timezone;
    this.mode = p.mode;
  }
  async establish(session: {
    name: string;
    email: string;
    csrf: string;
    preferences: Preferences;
  }) {
    this.name = session.name;
    this.email = session.email;
    this.csrf = session.csrf;
    this.applyPrefs(session.preferences);
    this.entered = true;
    this.password = "";
    this.generation++;
    await this.refresh();
    const destination = new URLSearchParams(location.search).get("workspace");
    if (destination === "Calendar" || destination === "Reminders")
      this.tab = destination;
    this.cdr.markForCheck();
  }
  async restore() {
    try {
      await this.establish(await this.api("/auth/session"));
    } catch {
    } finally {
      this.checking = false;
      this.cdr.markForCheck();
    }
  }
  async authenticate() {
    if (this.authBusy) return;
    this.authBusy = true;
    this.authError = "";
    try {
      const data = await this.api<{
        name: string;
        email: string;
        csrf: string;
        preferences: Preferences;
      }>("/auth/" + this.authMode, "POST", {
        email: this.email,
        password: this.password,
        ...(this.authMode === "register"
          ? { name: this.name, local_consent: this.localConsent }
          : {}),
      });
      await this.establish(data);
    } catch (e) {
      this.authError = this.error(e);
    } finally {
      this.authBusy = false;
      this.cdr.markForCheck();
    }
  }
  async refresh() {
    await Promise.all([this.loadItems(), this.poll()]);
  }
  async loadItems() {
    const generation = this.generation;
    await Promise.all(
      ["Calendar", "Reminders", "Notes"].map(async (kind) => {
        if (!(kind === "Notes" ? this.notesConsent : this.calendarConsent)) {
          this.collections[kind] = [];
          return;
        }
        try {
          const rows = await this.api<Item[]>("/entries/" + kind);
          if (generation === this.generation) this.collections[kind] = rows;
        } catch (e) {
          this.notify(this.error(e));
        }
      }),
    );
    this.cdr.markForCheck();
  }
  async poll() {
    const generation = this.generation;
    try {
      const [learning, runtime] = await Promise.all([
        this.api<Learning>("/learning/status"),
        this.api<NonNullable<typeof this.runtime>>("/runtime"),
      ]);
      if (generation !== this.generation) return;
      this.learning = learning;
      this.runtime = runtime;
      if (this.calendarConsent) {
        const reminders = await this.api<Item[]>("/entries/Reminders");
        if (generation !== this.generation) return;
        this.collections["Reminders"] = reminders;
        for (const item of reminders) {
          if (item.fired && !item.done && !this.notified.has(item.id)) {
            this.notified.add(item.id);
            this.notify("Reminder due: " + item.title);
          }
        }
      }
    } catch (e) {
      if (this.entered) this.notify(this.error(e));
    } finally {
      this.cdr.markForCheck();
    }
  }
  go(t: string) {
    this.tab = t;
    this.adding = false;
    this.modeOpen = false;
    if (t === "Settings") {
      void this.poll();
      void this.loadAudit();
      void this.loadPush();
      void this.loadSnips();
    }
  }
  notify(s: string) {
    this.toast = s;
    this.cdr.markForCheck();
    setTimeout(() => {
      if (this.toast === s) this.toast = "";
      this.cdr.markForCheck();
    }, 6500);
  }
  selectMode(m: string) {
    this.mode = m;
    this.modeOpen = false;
    if (!this.busy) this.setRobot("idle", "idle", "Ready when you are");
  }
  prompt(s: string) {
    this.draft = s;
    document.querySelector<HTMLTextAreaElement>("textarea")?.focus();
  }
  keydown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void this.send();
    }
  }
  async send() {
    const text = this.draft.trim();
    if (!text || this.busy) return;
    this.messages.push({ role: "user", text });
    this.draft = "";
    this.busy = true;
    const generation = this.generation;
    const requestMode = this.mode;
    const visual = intentForRobot(text);
    this.setRobot(
      visual,
      "processing",
      visual === "calendar"
        ? "Clock mode · finding the right time"
        : visual === "reminder"
          ? "Notepad mode · listening"
          : visual === "summarizer"
            ? "Thinking mode · finding the key points"
            : "Listening",
    );
    try {
      const result = await this.api<{
        text: string;
        location: string;
        proposal: Proposal;
        intent: string;
        confidence: number;
        explanation: { token: string; contribution: number }[];
      }>("/assistant/command", "POST", { text, mode: requestMode });
      if (generation !== this.generation) return;
      this.messages.push({
        role: "assistant",
        ...result,
        source: text,
        robotMode: requestMode,
      });
      const actual = robotForBackendIntent(result.intent);
      if (result.location !== "local backend")
        this.setRobot("idle", "idle", "Response ready");
      else if (actual === "summarizer") {
        if (result.text.startsWith("Please include the text"))
          this.setRobot(actual, "error", "Add the text you want summarised");
        else this.robotSucceeded(actual, "Summary ready");
      } else if (actual !== "idle")
        this.setRobot(
          actual,
          "review",
          result.proposal
            ? "Draft ready · review before saving"
            : actual === "calendar"
              ? "Your calendar is ready"
              : "Your workspace is ready",
        );
      else this.setRobot("idle", "idle", "Ready when you are");
    } catch (e) {
      if (generation === this.generation) {
        this.setRobot(visual, "error", "Couldn’t finish · nothing confirmed");
        this.messages.push({
          role: "assistant",
          text: this.error(e),
          location: "Operation not completed",
          robotMode: requestMode,
        });
      }
    } finally {
      this.busy = false;
      this.cdr.markForCheck();
      setTimeout(() => {
        const box = document.querySelector(".conversation");
        if (box) box.scrollTop = box.scrollHeight;
      }, 0);
    }
  }
  review(message: Message) {
    if (!message.proposal) return;
    this.tab = message.proposal.kind;
    const p = message.proposal;
    this.openItem(
      p.id === undefined
        ? undefined
        : {
            id: p.id,
            title: p.title,
            detail: p.detail,
            done: p.done ?? false,
            version: p.version,
          },
    );
    this.deleting = p.operation === "delete";
    this.itemTitle = message.proposal.title;
    this.itemDetail = message.proposal.detail;
    this.learningText =
      p.operation === "create" || !p.operation ? message.source : undefined;
  }
  async teach(message: Message) {
    try {
      await this.api("/learning/examples", "POST", {
        text: message.source,
        label: message.intent,
      });
      message.taught = true;
      this.notify("Your labelled example was encrypted and queued locally.");
      await this.poll();
    } catch (e) {
      this.notify(this.error(e));
    }
  }
  openItem(item?: Item) {
    this.previousFocus = document.activeElement as HTMLElement;
    this.editingId = item?.id ?? null;
    this.editingVersion = item?.version;
    this.editingDone = item?.done ?? false;
    this.deleting = false;
    this.itemTitle = item?.title ?? "";
    this.itemDetail = item?.detail ?? "";
    this.learningText = undefined;
    this.modalError = "";
    this.adding = true;
    this.setRobot(
      this.workspaceRobotIntent,
      "review",
      this.editingId === null
        ? "Draft · review before saving"
        : "Review your changes",
    );
    setTimeout(() => document.getElementById("item-title")?.focus(), 0);
  }
  closeItem() {
    if (this.saving) return;
    this.adding = false;
    this.setRobot("idle", "idle", "Ready when you are");
    this.previousFocus?.focus();
  }
  dialogKeys(event: KeyboardEvent) {
    if (event.key === "Escape") {
      this.closeItem();
      return;
    }
    if (event.key !== "Tab") return;
    const nodes = Array.from(
      (event.currentTarget as HTMLElement).querySelectorAll<HTMLElement>(
        "button,input,textarea",
      ),
    );
    const first = nodes[0],
      last = nodes[nodes.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
  async saveItem() {
    if (this.saving) return;
    if (this.deleting) {
      await this.confirmDelete();
      return;
    }
    if (!this.itemTitle.trim()) {
      this.modalError = "Please add a title.";
      return;
    }
    if (this.tab !== "Notes" && !this.itemDetail) {
      this.modalError = "Please choose a date and time.";
      return;
    }
    this.saving = true;
    this.setRobot(
      this.workspaceRobotIntent,
      "processing",
      "Saving your changes…",
    );
    const kind = this.tab;
    try {
      await this.api(
        "/entries/" +
          kind +
          (this.editingId !== null ? "/" + this.editingId : ""),
        this.editingId !== null ? "PUT" : "POST",
        {
          title: this.itemTitle,
          detail: this.itemDetail,
          done: this.editingDone,
          ...(this.editingVersion
            ? { expected_version: this.editingVersion }
            : {}),
          ...(this.learningText ? { learning_text: this.learningText } : {}),
        },
      );
      this.saving = false;
      this.closeItem();
      this.robotSucceeded(
        kind === "Calendar" ? "calendar" : "reminder",
        kind === "Calendar"
          ? "Event saved"
          : kind === "Reminders"
            ? "Reminder saved"
            : "Note saved",
      );
      this.notify("Saved with AES-256-GCM encryption.");
      await this.loadItems();
      await this.poll();
    } catch (e) {
      this.modalError = this.error(e);
      this.setRobot(
        this.workspaceRobotIntent,
        "error",
        "Not saved · please retry",
      );
    } finally {
      this.saving = false;
      this.cdr.markForCheck();
    }
  }
  async toggleDone(item: Item) {
    try {
      await this.api("/entries/Reminders/" + item.id, "PUT", {
        title: item.title,
        detail: item.detail,
        done: !item.done,
        expected_version: item.version,
      });
      await this.loadItems();
    } catch (e) {
      this.notify(this.error(e));
    }
  }
  async remove(id: number) {
    if (!window.confirm("Delete this item? This cannot be undone.")) return;
    try {
      await this.api(
        "/entries/" + this.tab + "/" + id,
        "DELETE",
        undefined,
        this.collections[this.tab].find((i) => i.id === id)?.version
          ? {
              "If-Match": this.collections[this.tab].find((i) => i.id === id)!
                .version!,
            }
          : {},
      );
      await this.loadItems();
      this.notify("Item deleted.");
    } catch (e) {
      this.notify(this.error(e));
    }
  }
  async confirmDelete() {
    if (this.editingId === null || this.saving) return;
    this.saving = true;
    const kind = this.tab;
    try {
      await this.api(
        "/entries/" + kind + "/" + this.editingId,
        "DELETE",
        undefined,
        this.editingVersion ? { "If-Match": this.editingVersion } : {},
      );
      this.saving = false;
      this.closeItem();
      this.robotSucceeded(
        kind === "Calendar" ? "calendar" : "reminder",
        "Item deleted",
      );
      this.notify("Item deleted. Unsent notifications cancelled.");
      await this.loadItems();
    } catch (e) {
      this.modalError = this.error(e);
      this.setRobot(
        this.workspaceRobotIntent,
        "error",
        "Not deleted · review again",
      );
    } finally {
      this.saving = false;
      this.cdr.markForCheck();
    }
  }
  async loadSnips() {
    const generation = this.generation;
    try {
      const value = await this.api<NonNullable<typeof this.snipsStatus>>(
        "/models/snips/status",
      );
      if (generation === this.generation) this.snipsStatus = value;
    } catch {
      /* Optional public benchmark is not needed for workspace tasks. */
    }
    this.cdr.markForCheck();
  }
  async loadPush() {
    const generation = this.generation;
    try {
      const status = await this.api<{ configured: boolean; detail: string }>(
        "/notifications/status",
      );
      if (generation !== this.generation) return;
      this.pushConfigured = status.configured;
      this.pushDetail = status.detail;
      if (this.pushSupported) {
        const registration = await navigator.serviceWorker.getRegistration("/");
        const sub = await registration?.pushManager.getSubscription();
        const binding = sub
          ? await this.api<{ enabled: boolean }>(
              "/notifications/subscription-status",
              "POST",
              { endpoint: sub.endpoint },
            )
          : { enabled: false };
        if (generation !== this.generation) return;
        this.pushBrowserPresent = !!sub;
        this.pushActive = binding.enabled;
      }
    } catch (e) {
      if (generation === this.generation) this.pushDetail = this.error(e);
    }
    this.cdr.markForCheck();
  }
  async enablePush() {
    if (this.pushBusy) return;
    if (!this.pushSupported || this.embedded) {
      this.notify(
        "Open this site in a standalone HTTPS tab in a browser that supports Web Push.",
      );
      return;
    }
    this.pushBusy = true;
    const generation = this.generation;
    try {
      // Permission is requested from this explicit click, never on page load.
      if ((await Notification.requestPermission()) !== "granted")
        throw new Error(
          "Notifications are blocked. Change this site's browser permission to enable them.",
        );
      const key = await this.api<{ public_key: string }>(
        "/notifications/public-key",
      );
      await navigator.serviceWorker.register("/push-worker.js", {
        scope: "/",
        updateViaCache: "none",
      });
      const registration = await navigator.serviceWorker.ready;
      const base64 = key.public_key.replace(/-/g, "+").replace(/_/g, "/");
      const bytes = Uint8Array.from(
        atob(base64 + "=".repeat((4 - (base64.length % 4)) % 4)),
        (c) => c.charCodeAt(0),
      );
      let subscription = await registration.pushManager.getSubscription();
      subscription ??= await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: bytes,
      });
      if (generation !== this.generation) {
        await subscription.unsubscribe();
        return;
      }
      await this.api("/notifications/subscribe", "POST", subscription.toJSON());
      if (generation !== this.generation) return;
      this.pushActive = true;
      this.pushBrowserPresent = true;
      this.notify(
        "Browser push enabled for saved events and reminders on this device.",
      );
    } catch (e) {
      if (generation === this.generation) this.notify(this.error(e));
    } finally {
      this.pushBusy = false;
      await this.loadPush();
      this.cdr.markForCheck();
    }
  }
  async disablePush() {
    if (this.pushBusy || !this.pushSupported) return;
    this.pushBusy = true;
    try {
      const registration = await navigator.serviceWorker.getRegistration("/");
      const subscription = await registration?.pushManager.getSubscription();
      if (subscription) {
        await this.api("/notifications/unsubscribe", "POST", {
          endpoint: subscription.endpoint,
        });
        await subscription.unsubscribe();
      }
      this.pushActive = false;
      this.pushBrowserPresent = false;
      this.notify(
        "Push disabled on this browser. Other devices are unchanged.",
      );
    } catch (e) {
      this.notify(this.error(e));
    } finally {
      this.pushBusy = false;
      this.cdr.markForCheck();
    }
  }
  async saveSettings() {
    if (this.settingsBusy) return;
    this.settingsBusy = true;
    try {
      const p = await this.api<Preferences>("/settings", "PUT", {
        assistant: this.localAssistant,
        calendar: this.calendarConsent,
        notes: this.notesConsent,
        summary: this.summaryConsent,
        training: this.training,
        cloud: this.cloud,
        epsilon: Number(this.epsilon),
        reduced: this.reduced,
        timezone: this.timezone,
        mode: this.mode,
      });
      this.applyPrefs(p);
      this.notify(
        "Preferences and consent saved. Past privacy expenditure is unchanged.",
      );
      await this.refresh();
      await this.loadAudit();
      await this.loadPush();
    } catch (e) {
      this.notify(this.error(e));
    } finally {
      this.settingsBusy = false;
      this.cdr.markForCheck();
    }
  }
  async saveProfile() {
    try {
      await this.api("/profile", "PATCH", { name: this.name });
      this.notify("Profile saved.");
    } catch (e) {
      this.notify(this.error(e));
    }
  }
  async changePassword() {
    try {
      await this.api("/auth/password", "POST", {
        current_password: this.currentPassword,
        new_password: this.newPassword,
      });
      this.currentPassword = "";
      this.newPassword = "";
      this.clearSession();
      this.notify("Password changed. All sessions revoked; sign in again.");
    } catch (e) {
      this.notify(this.error(e));
    }
  }
  async loadAudit() {
    try {
      this.auditRows = await this.api("/audit");
      this.cdr.markForCheck();
    } catch (e) {
      this.notify(this.error(e));
    }
  }
  async verifyAudit() {
    try {
      const r = await this.api<{
        valid: boolean;
        checked: number;
        limitation: string;
      }>("/audit/verify");
      this.auditResult = r.valid
        ? `${r.checked} entries verified. ${r.limitation}`
        : "Audit verification failed. Contact the operator.";
      this.cdr.markForCheck();
    } catch (e) {
      this.notify(this.error(e));
    }
  }
  async runLearning() {
    try {
      const r = await this.api<{ detail: string }>("/learning/run", "POST");
      this.notify(r.detail);
      await this.poll();
    } catch (e) {
      this.notify(this.error(e));
    }
  }
  get noun() {
    return this.tab === "Calendar"
      ? "event"
      : this.tab === "Reminders"
        ? "reminder"
        : "note";
  }
  get dateLabel() {
    return new Intl.DateTimeFormat("en", {
      weekday: "long",
      month: "long",
      day: "numeric",
      timeZone: this.timezone,
    }).format(new Date());
  }
  clearSession() {
    this.setRobot("idle", "idle", "Ready when you are");
    this.generation++;
    this.adding = false;
    this.itemTitle = "";
    this.itemDetail = "";
    this.learningText = undefined;
    this.draft = "";
    this.runtime = null;
    this.snipsStatus = null;
    this.pushActive = false;
    this.pushBrowserPresent = false;
    this.auditResult = "";
    this.entered = false;
    this.csrf = "";
    this.messages = [];
    this.collections = { Calendar: [], Reminders: [], Notes: [] };
    this.learning = null;
    this.auditRows = [];
    this.password = "";
    this.currentPassword = "";
    this.newPassword = "";
    this.notified.clear();
    this.tab = "Assistant";
    this.cdr.markForCheck();
  }
  async logout() {
    try {
      await this.api("/auth/logout", "POST");
      this.clearSession();
      this.notify("Signed out. This session is revoked.");
    } catch (e) {
      this.notify(this.error(e));
    }
  }
}
bootstrapApplication(App, {
  providers: [provideZonelessChangeDetection()],
}).catch(console.error);
