/* No fetch handler, offline cache, authentication token or private workspace data. */
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) =>
  event.waitUntil(self.clients.claim()),
);
self.addEventListener("push", (event) => {
  let data = {};
  try {
    data = event.data?.json() || {};
  } catch {
    /* Safe generic fallback. */
  }
  const workspace = data.workspace === "Calendar" ? "Calendar" : "Reminders";
  event.waitUntil(
    self.registration.showNotification(
      workspace === "Calendar"
        ? "PPDA · Event starting"
        : "PPDA · Reminder due",
      {
        body: "Open your workspace to view the details.",
        icon: "/brand/mark.svg",
        tag: typeof data.tag === "string" ? data.tag.slice(0, 80) : "ppda-due",
        data: { workspace },
        renotify: false,
      },
    ),
  );
});
self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const workspace =
    event.notification.data?.workspace === "Calendar"
      ? "Calendar"
      : "Reminders";
  const url = new URL("/?workspace=" + workspace, self.location.origin).href;
  event.waitUntil(
    (async () => {
      const clients = await self.clients.matchAll({
        type: "window",
        includeUncontrolled: true,
      });
      for (const client of clients) {
        if (new URL(client.url).origin === self.location.origin) {
          const navigated = await client.navigate(url);
          if (navigated) return navigated.focus();
        }
      }
      return self.clients.openWindow(url);
    })(),
  );
});
