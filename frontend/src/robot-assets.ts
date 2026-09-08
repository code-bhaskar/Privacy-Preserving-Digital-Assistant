/** Supplied named ZIP images. All paths are same-origin, optimised WebP copies. */
export type RobotIntent = "idle" | "calendar" | "reminder" | "summarizer";
export type RobotPhase =
  "idle" | "context" | "processing" | "review" | "success" | "error";
const ROOT = "brand/robots/";
export const MODE_ROBOTS: Record<string, string> = {
  Default: ROOT + "default_mode_robot_blue.webp",
  Privacy: ROOT + "privacy_mode_robot_green.webp",
  Global: ROOT + "global_model_robot_soft_red.webp",
};
export const ROBOT_FRAMES: Record<Exclude<RobotIntent, "idle">, string[]> = {
  calendar: [
    "normal_robot",
    "detecting_intent",
    "clock_face",
    "mode",
    "confirmed",
  ].map((name) => ROOT + "calendar_" + name + ".webp"),
  reminder: [
    "normal_robot",
    "detecting_intent",
    "notepad_appears",
    "writing_task",
    "saved",
  ].map((name) => ROOT + "reminder_" + name + ".webp"),
  summarizer: [
    "normal_robot",
    "thinking_begins",
    "deep_processing",
    "information_compression",
    "summary_ready",
  ].map((name) => ROOT + "summarizer_" + name + ".webp"),
};
export function intentForRobot(text: string): RobotIntent {
  // The task prefix wins over words inside a document to be summarised.
  if (/\b(summari[sz](?:e|er)|summary|tldr|shorten)\b|key points/i.test(text))
    return "summarizer";
  if (/\b(remind(?:er)?|remember|task|todo|forget)\b/i.test(text))
    return "reminder";
  if (
    /\b(calendar|schedule|meeting|appointment|date|tomorrow|today|event)\b|\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?/i.test(
      text,
    )
  )
    return "calendar";
  return "idle";
}
export function robotForBackendIntent(intent?: string): RobotIntent {
  if (intent === "summary") return "summarizer";
  if (intent === "reminder" || intent === "note") return "reminder";
  return intent === "calendar" ? "calendar" : "idle";
}
export function modeRobot(mode: string): string {
  return MODE_ROBOTS[mode] || MODE_ROBOTS["Default"];
}
