# PPDA — Privacy-Preserving Digital Assistant

A login-first Angular workspace with a real FastAPI backend, encrypted calendar/reminder/note storage, local ONNX intent inference and summarisation, consent-controlled cloud integration, and a single supervised federated-learning pipeline.

## What is implemented

- Registration, login, logout and password change; bcrypt with a non-truncating password policy.
- Signed JWTs in HttpOnly/SameSite cookies, CSRF tokens, persistent session revocation and persistent login throttling.
- Backend ownership checks and category-specific processing consent.
- AES-256-GCM encrypted workspace text and queued training examples, with per-user derived keys, fresh nonces and AAD.
- Persistent CRUD, real local ONNX Runtime classification, token-occlusion explanations, local neural summaries (FLAN-T5-small default; optional model installation required), and reviewed assistant create/update/delete proposals.
- Optional OpenAI adapter (Global mode) and loopback Ollama adapter (local conversation). No cloud keys are shipped or exposed to the browser.
- Opt-in scheduled browser Web Push for calendar events and reminders: one-second due checks, encrypted subscriptions, durable retries and cancellation. Delivery is best effort, not an exact-time alarm.
- A separately trained, bundled seven-label SNIPS ONNX benchmark: 97.86% validation accuracy on 700 examples; public central training, not a private FL/DP result.
- Append-only, SHA-256 chained and HMAC-authenticated audit records, with a verification UI.
- Real independent client OS workers, pairwise X25519/HKDF/ChaCha20 masks and all-participant aggregation.
- Client-local Gaussian noise, L2 clipping, persistent **conservative sequential** privacy accounting, minimum-cohort enforcement and guarded model publication.
- Light/dark themes, robot branding, accessible forms, settings and reduced-motion controls.

## Important boundaries

**Local means this backend host, not the browser.** To keep text on your own device, run the backend on that device. In a remotely hosted preview, its operator can access plaintext during processing and holds the storage keys. This is not end-to-end encryption against that operator.

The FL implementation is a **single-host research implementation**, using three real opted-in account queues and three independent OS workers. It does not establish physically separate-device isolation, sybil resistance or malicious-server security. It is not Bonawitz/Shamir dropout recovery: **any dropout aborts the round** without opening recovery secrets.

The implemented mechanism is **client-local Gaussian DP with basic composition**, not the originally proposed distributed-central-DP/RDP mechanism. This conservative choice costs more utility. Finite-precision Gaussian sampling is not independently audited; do not describe this as a certified production privacy guarantee. Model candidates that fail a public-seed regression gate are rejected, while their budget remains spent. Historical benchmark claims from the earlier overview do not apply to this new implementation.

## Run locally

Linux, Python 3.11+ and Node 20.19+ (or compatible newer releases) are required. SQLite is the tested database. **Use one backend process per database**; startup rejects a concurrent process. PostgreSQL, multi-replica deployment and Windows client-worker support are future work.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/init_env.py
.venv/bin/alembic upgrade head
cd frontend && npm ci && cd ..
```

`init_env.py` creates fresh secrets with mode `0600` and does not print or replace them. Back up `.env` securely: replacing its AES key makes existing data unreadable. Runtime data and secrets are git-ignored.

Terminal 1, for an **HTTP loopback-only development session**:

```bash
COOKIE_SECURE=false COOKIE_SAMESITE=lax COOKIE_PARTITIONED=false .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

Terminal 2:

```bash
cd frontend
npm start -- --port 4200
```

Open `http://localhost:4200` and create an account. There are **no preloaded accounts, passwords or private records**.

For an embedded HTTPS preview, set `COOKIE_SAMESITE=none` and `COOKIE_PARTITIONED=true` so supporting browsers can use a partitioned third-party session cookie. CSRF and same-origin checks remain required. For an ordinary same-origin deployment use `COOKIE_SAMESITE=lax` and `COOKIE_PARTITIONED=false`.

For an HTTPS sandbox/reverse-proxy preview, keep `COOKIE_SECURE=true` and bind the backend to `0.0.0.0`. Browser API calls use relative URLs through the Angular proxy, which preserves Host for origin checking. Permissive dev-server hosts are for previews, not a hardened production deployment. Deploy behind HTTPS; never disable Secure cookies on a public HTTP host.

## Using the connected assistant

1. Create an account. Optionally grant local processing permissions during registration; change categories individually in Settings.
2. Try Privacy mode: `Remind me to call Rahul tomorrow at 5 pm`. Review the draft and save it. The operation is not performed just because the model predicted an intent.
3. Try `Summarize: ...` with several sentences. The configured local model runs offline after installation; Global uses configured OpenAI, default `gpt-4o-mini`. See [local model setup](docs/LOCAL_SUMMARIES.md).
4. Open Calendar, Reminders or Notes to create/edit/delete encrypted records. Changes survive reload and logout.
5. Save preferences explicitly. Choosing a theme applies immediately; processing consent and budgets change only after a successful server save.
6. Inspect and verify your privacy activity in Settings.
7. Try `List my reminders`, `Update reminder #12 title to "Call Dad"`, or `Delete event "Design sync"` using your own IDs/titles. Confirm each change in the review dialog.
8. For browser push, run `.venv/bin/python scripts/init_vapid.py`, configure your VAPID operator contact, then enable it in Settings from a standalone HTTPS tab. This optional feature uses a browser push provider even in Privacy mode.

See [push setup, assistant commands and model provenance](docs/NOTIFICATIONS_AND_COMMANDS.md) for setup, cancellation/delivery limits and SNIPS reproduction.

### Global and Default modes

Set `OPENAI_API_KEY` securely in the backend environment and optionally change `OPENAI_MODEL`, then restart the backend. Global mode requires saved cloud consent. Only the current prompt is sent; workspace records and prior conversation are not attached. Provider failures are reported, not replaced with fabricated answers.

Default handles task requests locally. Opted-in general chat may use OpenAI if configured; a conservative sensitive-keyword guard keeps recognised sensitive prompts local, **but it is not a universal privacy classifier**. Use Privacy mode for an absolute no-cloud-LLM routing rule. Separately enabled browser push still uses its browser vendor’s notification service. Explicit Global mode can transmit sensitive text, so use it deliberately.

For broader local conversation, run Ollama yourself and configure `OLLAMA_URL=http://127.0.0.1:11434` and `OLLAMA_MODEL` to an installed model. No LLM weights are downloaded by the app. Without Ollama, local mode supports the task classifier, configured local summaries and limited deterministic greetings—not general conversation. FLAN-T5-small is the default summariser; T5-small and DistilBART-CNN are selectable. Missing model assets produce an explicit setup error, not cloud or extractive fallback.

## Integrated learning

1. Each of **three actual participating accounts** explicitly enables learning and saves settings.
2. Each contributes at least **three confirmed intent examples** using “Correct intent → Confirm label & contribute”, or by reviewing/saving assistant-generated task drafts after opting in.
3. The single pipeline automatically checks eligibility every ten seconds (or on a queued example/settings change). It never fabricates participants.
4. Encrypted queues are supplied to separate client subprocesses. Each trains a small softmax model locally, clips its delta, adds its own Gaussian noise, quantises and masks it. Only masked vectors are returned to the supervisor.
5. All participants must finish. Any timeout, dropout, malformed update or preference-version change aborts before aggregation/publication. Every reservation stays charged to avoid retry accounting loopholes.
6. The shared candidate is tested against the public seed regression set. It is activated only if the gate passes; otherwise the previous model stays active. This score is **not a held-out accuracy benchmark**.
7. Consumed encrypted examples are removed after the attempt; the ledger and audit remain. Future local predictions use the active shared version through ONNX Runtime.

Privacy parameters: client replacement adjacency, clipping norm `0.1`, per-attempt ε `0.5`, δ `1e-6`, lifetime δ cap `1e-5`, user epsilon target `1–10`. No sampling amplification is claimed. Setting a lower target cannot undo past expenditure. Opt-out removes unused examples; changes during a running round conservatively abort it. The client/account protection unit does not automatically cover a person across multiple accounts/devices.

This updates the shared **assistant intent model**, not a third-party conversational LLM. Strong local DP at small cohort sizes often destroys utility; rejection is a valid, honest outcome.

## Verification

```bash
.venv/bin/python -m pytest -q
cd frontend
npm run build
npm run test:smoke
npm run test:push
npm run test:assets
```

Additional tests cover reviewed assistant CRUD, ambiguous/foreign targets, stale-edit and timezone safeguards, due-time dispatch, retry/recovery, cancellation and encrypted push subscriptions. The browser smoke exercises service-worker registration plus mocked permission/subscription APIs, not actual vendor-to-device delivery. SNIPS training/export is tested with an offline fixture; its actual public-data report is bundled in `models/snips/metrics.json`.

Backend tests cover auth, cookie/CSRF/origin enforcement, password policy, IDOR checks, ciphertext/AAD, append-only audit and privacy ledger, ONNX serving, local/cloud boundaries, clipping/noise, pairwise cancellation, real three-process training, dropout abort and persistent accounting.

The Linux browser smoke test launches its **own isolated database, secrets, backend and static server** (backend port 8011), then exercises real registration, persistence, assistant drafts, summaries, learning opt-in, audit, logout/login, themes and mobile layout. It does not modify the live workspace or call a paid provider. Provider integration is mocked in unit tests; a live cloud call requires operator configuration and explicit consent.

## Repository map

- `app/`: FastAPI APIs, crypto, schemas, storage, ONNX/local inference, consent.
- `fl/`: client workers, Gaussian mechanism, masking protocol and integrated supervisor.
- `migrations/`: Alembic schema plus append-only triggers.
- `frontend/`: Angular workspace and browser smoke test.
- `tests/`: isolated backend/security/protocol tests.
- `docs/PRD.md`: product specification and implementation delta.
- `docs/SECURITY_AND_DP.md`: threat model, accounting contract and limitations.
- `frontend/public/brand/`: robot SVG mark, light/dark wordmarks and mascot.

No third-party security audit, external audit-chain anchoring, email verification/reset, hardware keystore, poisoning defenses or physical-device FL enrollment is provided in this release.

## Supplied robot images

The website now uses the **18 named images from the two ZIPs on GitHub**, mapped to Default/Privacy/Global modes and calendar/reminder/summarisation contexts. Three reference GIFs are exported; live animations use individually controlled WebP frames so a “saved” frame cannot play before a successful API response. See [Robot artwork and placement](docs/ROBOT_ASSETS.md) for provenance, rebuild instructions and tests.

## Local neural summary setup

See [LOCAL_SUMMARIES.md](docs/LOCAL_SUMMARIES.md) for the optional CPU dependencies, model download, offline deployment, input limits and validation caveats. Angular is the frontend; FastAPI is the backend. Neural weights were not downloaded in the restricted sandbox, and real neural output quality is not claimed as tested.
