# Scheduled push, assistant actions and SNIPS

## Browser push setup

1. Install backend requirements; run `.venv/bin/alembic upgrade head`.
2. Run `.venv/bin/python scripts/init_vapid.py`. It creates `.runtime/vapid-private.pem`
   with mode 0600 and never replaces an existing key. Back it up with the other local secrets.
   Set `VAPID_SUBJECT=mailto:your-operator-contact` in the private backend environment;
   the sample contact is a development placeholder. `VAPID_PRIVATE_KEY_PATH` overrides the path.
3. Keep the backend running. Serve the frontend and API under one HTTPS origin. The
   root `/push-worker.js` must be served as JavaScript, not an SPA fallback. Also serve
   `/manifest.webmanifest` and the bundled icon files; iOS requires a supported OS and
   installation to the Home Screen before push can be enabled.
4. Open the workspace in a **standalone browser tab**, not the embedded preview. Sign in,
   save Calendar/Reminders processing consent and the correct IANA timezone in Settings.
5. Click **Enable browser push** and grant the browser permission. Permission is never
   requested on page load. Repeat enrollment for every desired browser/device.
6. Create an event or reminder a few minutes ahead, confirm Save, then close its tab to
   try delivery. Whether delivery works with the entire browser stopped depends on the
   browser/OS. Clicking the notification opens the right workspace; login remains required.

Push enrollment is a separate, explicit opt-in from cloud-LLM processing. **Push uses
Google/Apple/Mozilla/Microsoft browser push infrastructure even in Privacy mode.** The
push provider can observe routing/timing/network metadata. Event titles, notes, prompts
and user identifiers are not included in the encrypted Web Push payload. Lock-screen
text is generic. Subscription URLs/keys are stored with per-user AES-GCM encryption;
provider destinations are allowlisted and redirects disabled. Up to ten browser subscriptions
per account are allowed. The service worker does not cache private pages, API responses,
authentication tokens or user content.

### Timing and cancellation semantics

- A one-second due scanner handles saved Calendar and Reminders entries in UTC, using
  the account's saved timezone for naive date/time input. Read/edit views convert scheduled
  instants into the current saved timezone. A separate sender avoids blocking due scanning
  on network calls. Unsaved chat drafts are never scheduled.
- A SQLite outbox persists work across restarts. `(entry, subscription, due instant)` is
  unique. Provider acceptance is recorded, not falsely described as confirmed device delivery.
- Retries use bounded backoff and at most five attempts. Expired browser endpoints (404/410)
  are removed. Interrupted sends recover after a lease. A stable notification tag helps
  coalesce retries, but an ambiguous network failure can still cause duplicate delivery.
- Rescheduling cancels the old queued delivery and schedules the new time. Completing or
  deleting an item, disabling a browser subscription or revoking Calendar/Reminders consent
  prevents pending delivery. Revoking category consent removes all its subscriptions.
- Already accepted or in-flight pushes **cannot be recalled**. Messages expire one hour
  after their due instant; long outages will not produce arbitrarily old alerts. Existing
  entries already marked fired before enrollment/migration are not retroactively pushed.
- **Not an exact-time alarm or emergency notification service.** Backend uptime, clock
  accuracy, queue load, connectivity, permission changes, battery policies, Do Not Disturb,
  browser support and provider availability affect actual display time. Closing a tab is
  supported by the service worker; guaranteed delivery while the backend is stopped is not.
- Disable on this browser to stop this device; other enrolled devices are unchanged.
  Generic notifications may still arrive after logout until disabled. A shared browser's
  existing subscription cannot be reassigned to a different account without unsubscribing.

Automated tests use mocked vendor sends and browser permission/subscription APIs, while
exercising real SQLite transactions, HTTP enrollment and service-worker registration.
Real provider-to-device delivery must be tested with an enrolled device on its deployment.

## Assistant CRUD commands

Use **Default or Privacy** mode. Examples:

```
Create an event "Design sync" tomorrow at 10 am
Remind me to call Rahul tomorrow at 5 pm
List my calendar events
Show my reminders
Reschedule event "Design sync" to 2099-06-10T09:45
Update reminder #12 title to "Call Dad"
Complete reminder #12
Reopen reminder #12
Delete event "Design sync"
Delete reminder #12
```

IDs shown by listing are examples; use IDs from your own workspace. Quoted titles and
explicit IDs are preferred. Partial titles only work when there is a unique match.
Ambiguous targets prompt for an ID; missing or foreign-account targets never produce a
mutation. Bulk delete/update is deliberately unsupported. This is conservative English
pattern/date parsing, not a general conversational tool-planning LLM. Unparsed changes
remain editable in the draft and are reported as such. Review all dates before saving.

Create, update and delete return **proposals only**. The review dialog must be confirmed.
Delete shows a read-only copy and an irreversible-action warning. Version fingerprints
reject a stale review with HTTP 409, including when the saved timezone changed. The
normal owner-scoped authenticated CRUD APIs apply the confirmed operation. The local
classifier's prediction alone can never perform an action.

These are the app's calendar events—not creation of Google/Outlook calendar collections
or external calendar synchronisation. Global mode remains direct, consented OpenAI chat
and has no workspace tool access; use Default/Privacy for actual workspace changes.

## Which LLM summarises?

Local summaries now default to **FLAN-T5-small**, with T5-small and DistilBART-CNN
alternatives. Inference runs offline on the backend CPU after the operator installs the
optional dependencies and stages pretrained weights. Missing assets fail explicitly;
there is no cloud or extractive fallback. The old sentence-ranking mode remains available
only through explicit `LOCAL_SUMMARY_MODEL=extractive` configuration.

Global uses the backend-configured OpenAI model, default **gpt-4o-mini**, with saved
cloud consent and an API key. Optional Ollama is separate local general chat. See
[local neural summary setup and validation limitations](LOCAL_SUMMARIES.md).

## SNIPS training and verification

A small trained public artifact and its measured report are bundled in `models/snips/`.
Reproduce with:

```bash
.venv/bin/python scripts/train_snips.py
# Optional alternate deployment directory:
.venv/bin/python scripts/train_snips.py --output /your/model/directory
# Set SNIPS_DIR to that directory on the backend if overridden.
```

The downloader pins an upstream commit, uses HTTPS, and can fall back to `gh api` when
raw GitHub access is unavailable. It never downloads private user queues. Raw benchmark
files are git-ignored. Exact train/validation overlaps are removed from training;
feature/hyperparameter fitting never uses validation labels. Metrics, source-file hashes,
ONNX integrity hash and export parity are recorded in `models/snips/metrics.json`.

Measured on 13,773 training / 700 validation examples: **97.86% accuracy**, **0.97847
macro-F1**. See the adjacent model README for citation, original benchmark scope and
limitations. These are public central-training results—not private-user FL/DP results.
The original seven labels are kept separate from the five-label workspace task model.
Settings shows this distinction; authenticated `/api/v1/models/snips/status` and
`POST /api/v1/models/snips/predict` (`{"text":"weather forecast"}`) expose benchmark-only
inference. It runs locally and never executes calendar/reminder actions.
