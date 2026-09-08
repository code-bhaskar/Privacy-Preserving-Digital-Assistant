# PPDA — Product Requirements Document

**Version:** 1.1 · **Date:** 9 September 2026  
**Product:** Privacy-Preserving Digital Assistant  
**Status:** Connected single-host implementation; production hardening remains

## 1. Purpose

A calm, minimal personal workspace for conversation, calendars, reminders and notes. PPDA helps people organise their day while giving them explicit control over where their information is processed and whether they participate in collaborative learning.

**Positioning:** A little help. A lot of privacy.

Privacy is a mode-specific, deployment-dependent property, not an unconditional promise. The browser sends commands to the application backend. For on-device processing, that backend and local model runtime must operate inside the user's trusted device boundary. A remotely hosted backend receives submitted text even when it does not call an external AI provider.

## 2. Implementation status and source of truth

The checkout initially contained only a README and mascot. This implementation now includes the Angular frontend, FastAPI backend, Alembic schema and real supervised client workers. The extensive earlier overview's historic test counts, benchmark figures and protocol claims are not evidence for this new code.

### Connected and tested

- Real registration/login/logout/password change, bcrypt and persistent JWT sessions in HttpOnly cookies; CSRF/origin checks and persistent rate limits.
- Owner-scoped calendar/reminder/note CRUD with AES-256-GCM encrypted text, per-user key derivation, purpose AAD and fresh nonces.
- Category-specific consent, persisted settings and audit verification.
- Local ONNX Runtime softmax intent classification, token occlusion, configured local neural summarisation (FLAN-T5-small by default, operator-installed weights) and reviewed create/update/delete task drafts.
- Optional consent-controlled OpenAI adapter and loopback Ollama adapter. Provider keys/local LLM weights are not shipped; no paid provider call was performed during testing.
- Scheduled opt-in browser Web Push for events/reminders, with encrypted subscriptions, durable retry/cancellation and generic lock-screen content; best-effort delivery, not an exact alarm.
- Separate bundled SNIPS public intent benchmark: 97.86% validation accuracy on 700 examples, central training with no FL/DP accuracy claim.
- Three real opted-in user queues handled by three separate OS workers in the integrated pipeline; no simulated participants.
- L2 clipping, client-local Gaussian noise, pairwise X25519/HKDF/ChaCha20 masking, append-only lifetime budget ledger, all-participant aggregation and guarded model publication.
- Responsive light/dark workspace, saved appearance, robot branding and reduced motion.

### Deliberate deviations from the original target

| Area               | Implemented now                                                                                           | Still future work                                                              |
| ------------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Storage            | SQLite with serialized transactions and Alembic; one backend process per database enforced by a file lock | PostgreSQL and multi-replica deployment                                        |
| ML                 | 645-weight linear softmax model trained in NumPy and served through ONNX Runtime                          | Original PyTorch IntentNet and its benchmark reproduction                      |
| DP                 | Conservative client-local Gaussian mechanism with basic sequential composition                            | Validated distributed-central-noise mechanism and matched RDP accounting       |
| Secure aggregation | All-member pairwise masks; any dropout aborts with no recovery-key release                                | Bonawitz self masks, Shamir recovery, authenticated physical-device enrollment |
| Client boundary    | Real OS workers on one trusted host                                                                       | Separate-device isolation and sybil-resistant identities                       |
| Assets             | 18 supplied ZIP images, state-controlled clock/notepad/thinking sequences and 3 exported GIFs             | Additional custom animation art beyond the supplied ZIP frames                 |

### Limits that must remain visible

“Local” means the backend host. A remote preview receives browser text and holds encryption keys. Client processes do not protect against that host's administrator. Cryptography and finite-precision Gaussian sampling are research-grade and unaudited. Small-cohort local DP can destroy model utility; rejecting a noisy candidate is valid and still spends reserved budget. Account creation is not privacy-preserving enrollment. Email verification/recovery, production keystores, independent audit anchoring and poisoning defenses remain unimplemented. Browser push is opt-in and uses browser-vendor infrastructure even in Privacy mode; provider-to-device delivery still requires a real enrolled-device acceptance test.

See [Security and DP notes](SECURITY_AND_DP.md) for the exact implemented mechanism and threat model. This status section takes precedence over future-target descriptions below.

## 3. Audience and primary journeys

### Users

- Individuals managing everyday plans and notes.
- Privacy-conscious users who want local processing and clear cloud boundaries.
- Opted-in participants contributing to a compatible shared assistant model.

### Primary journeys

1. Register or sign in → enter workspace → select a processing mode.
2. Ask to schedule an event → review parsed title/time → confirm → view calendar entry.
3. Create a reminder → review due time → receive supported notification → mark complete.
4. Write a note → save encrypted content → optionally summarise selected content.
5. Select Privacy Mode → complete a supported local task without cloud fallback.
6. Enable learning → set budget → contribute only to compatible, adequately sized cohorts.
7. Open Settings → inspect privacy expenditure, consent and processing history.

## 4. Information architecture and design

### Layout

Persistent left navigation: Assistant, Calendar, Reminders, Notes; Profile and Settings anchored below. Main header contains mode selector, transparent runtime status and theme control. Content sits within a readable maximum width. The composer remains easy to reach without covering conversation content.

### Visual system

- Light: off-white background, white cards, slate text, muted lavender and sage accents.
- Dark: charcoal/navy background, raised dark cards, pale text and restrained cyan accents.
- Typography: locally bundled DM Sans for UI, Manrope for headings.
- Generous spacing, subtle borders, rounded corners, no heavy neon surfaces.
- Robot mark: rounded head, antenna, paired glowing eyes; no third-party logos.
- Mascot: existing silver/cyan robot. Keep silhouette, materials and proportions consistent.
- Functional navigation uses simple matching SVG line icons rather than busy miniature illustrations.
- Support keyboard navigation, visible focus, readable labels, semantic forms and reduced motion.
- Target WCAG 2.2 AA. Complete contrast and assistive-technology audits before release; a successful build is not accessibility certification.

### Theme behavior

First visit should respect system preference; explicit Light/Dark selection overrides it and persists locally. Theme choice must not require an account or a remote request. All dialogs, controls, illustrations and empty/error states need both themes.

## 5. Functional requirements

Priority: **P0** required before a secure release; **P1** completes the first product release; **P2** optional expansion.

| ID       | Priority | Requirement                                  | Acceptance criterion                                                                          |
| -------- | -------- | -------------------------------------------- | --------------------------------------------------------------------------------------------- |
| AUTH-01  | P0       | Login-first protected workspace              | Unauthenticated protected API calls return 401; navigation guards are not the sole control.   |
| AUTH-02  | P0       | Registration, login, logout, password change | Credentials are verified server-side; logout invalidates the session across restart.          |
| AUTH-03  | P0       | Ownership checks                             | A user cannot read, change or delete another user's records.                                  |
| CON-01   | P0       | Category-specific consent                    | Missing or revoked consent blocks the applicable operation and records a privacy-safe denial. |
| MODE-01  | P0       | Privacy Mode                                 | Local processing only; model/network failure never triggers a cloud fallback.                 |
| MODE-02  | P0       | Global Mode                                  | An actual provider adapter requires informed cloud consent and exposes provider provenance.   |
| MODE-03  | P0       | Default Mode                                 | Deterministic local-first policy; cloud only for eligible tasks with valid consent.           |
| AST-01   | P1       | Intent/entity extraction                     | Supported requests map to validated predefined actions; ambiguity requires clarification.     |
| CAL-01   | P1       | Calendar CRUD                                | Create/read/update/delete events with timezone-aware time handling and owner isolation.       |
| REM-01   | P1       | Reminder CRUD and delivery                   | Store due time and status; distinguish firing a record from delivering a notification.        |
| NOTE-01  | P1       | Notes CRUD                                   | Encrypt title and content; support safe user-scoped search and editing.                       |
| SUM-01   | P1       | Local summarisation                          | Summarise selected content locally; never silently include it in cloud context.               |
| PROF-01  | P1       | Profile management                           | Validate profile updates; require reauthentication for sensitive changes.                     |
| AUD-01   | P0       | Privacy-safe audit history                   | Record actions without sensitive plaintext and verify chain integrity with documented limits. |
| SET-01   | P1       | Settings                                     | Change theme, motion, default mode, consent, learning preference and target budget.           |
| FL-01    | P0       | Genuine integrated FL                        | Opted-in client workers train locally and submit only protected updates via the protocol.     |
| DP-01    | P0       | Client-level differential privacy            | Clip, noise, account and enforce budget before releasing protected aggregates.                |
| MODEL-01 | P0       | Safe model rollout                           | Validate label schema, version, dimensions and quality; reject incompatible artifacts.        |
| UI-01    | P1       | Responsive robot workspace                   | Navigation and forms work on mobile/desktop, in light/dark modes and by keyboard.             |

## 6. Mode semantics

### Privacy

Authenticate → check consent → local intent/summarisation → validate action → obtain confirmation where appropriate → perform CRUD → audit metadata → queue eligible learning examples only with independent opt-in.

The compact intent classifier is not a general conversational LLM. Unsupported queries must produce an honest limitation. Optional Ollama/llama.cpp integration can provide local conversation after hardware requirements are defined.

### Global

Backend calls a configured external LLM using server-held credentials. Only the explicitly selected prompt/context is transmitted. Display provider identity and explain that the provider can read transmitted content. The application backend is not an end-to-end encrypted channel to the model. Private notes/calendar/history are not attached automatically.

### Default

Use deterministic routing based on supported intent, sensitivity, consent and runtime readiness. Cloud capability does not override denied consent. Never send user text to a cloud model just to decide whether it is private.

### Tool execution

Treat model output as untrusted. Validate tool schemas, owner identity, field lengths and time ranges. Require confirmation for destructive or externally consequential actions. Prompt injection must not override permissions or consent.

## 7. Security and storage requirements

### Authentication

- bcrypt with an explicit UTF-8 byte-length policy; reject unsupported password length rather than silently truncating.
- HS256 JWT validation with constrained algorithms, expiry and session identifiers; short-lived access tokens.
- Browser authentication via Secure, HttpOnly, SameSite cookies with CSRF defenses for state-changing requests.
- Persistent session revocation, shared login throttling and generic authentication errors.
- Dummy-hash verification for nonexistent users, without promising perfect timing indistinguishability.
- Derive owner identity from the authenticated session, not request-body user IDs.
- Never log passwords, tokens, reset secrets or full request content.

### Encryption

AES-256-GCM with fresh 96-bit nonces and versioned envelopes. Encrypt calendar titles, reminder text, note titles/bodies, message content and optional persisted conversation history. Bind owner and resource context using AAD. Existing ciphertext formats, if imported later, need tested migration support.

Production key management must separate keys from encrypted data and support rotation. Validate keys at startup; keep `.env` and secrets out of source control. Encryption does not protect content in a compromised running backend. Document remaining unencrypted metadata such as timing or record counts.

Do not persist decrypted ORM state accidentally. Encrypted-note search should initially decrypt an authorised user's bounded result set locally; avoid inventing searchable encryption.

### Auditing

Append-only database triggers plus SHA-256 linkage. Denied actions and security changes must be traceable without prompt leakage. An unanchored chain cannot detect every consistent full-chain rewrite or truncation; external checkpoints/anchoring are required for that stronger claim. Database administrators can bypass ordinary trigger-based controls.

### Deployment

HTTPS at a reverse proxy, same-origin relative browser URLs, restricted CORS, no exposed provider keys. Refuse startup on absent secrets or unapplied schema migrations. Rate limiting and revocation must work across replicas.

## 8. Integrated federated learning and differential privacy

### Distinguish the two global models

- **External global LLM:** conversational provider used in Global Mode.
- **Shared federated assistant model:** compatible model trained across consenting clients.

Federated contributions do not fine-tune a third-party provider automatically. A seven-class benchmark artifact must never replace an eight-class assistant artifact merely because both are ONNX files.

### Single product pipeline

Eligible interaction → explicit correction/validated label → encrypted client-local queue → background local training → L2 clip delta → distributed calibrated noise → quantisation → cryptographic masking → cohort secure aggregation → candidate evaluation → versioned publication → local activation.

The supervisor coordinates work, but private examples and individual plaintext updates stay within client workers. Same-host processes demonstrate separation of execution, not security from a privileged host. Minimum cohort size and threshold checks prevent single-user releases. No simulated clients or fake model-progress claims in the product path.

### Cryptography (target; see §2 for implemented subset)

Evaluate the existing proposed X25519/HKDF, ChaCha20 mask generation, AES-GCM sealed shares and Shamir recovery design before adopting it. Authenticate protocol clients and bind submissions to client, round and phase. Protect control-plane endpoints separately from browser user APIs. Enforce reveal invariants and reject malformed/replayed vectors. PRG masks provide computational, not information-theoretic, security. Use reviewed components where feasible; a working test suite does not replace a cryptographic audit.

### DP accounting contract

**Current mechanism:** fixed-participation client-local Gaussian noise with replacement sensitivity `2C`, per-round ε=0.5 and δ=10^-6, `C=0.1`, basic lifetime composition and δ cap 10^-5. No subsampling amplification or RDP claim. Reservations are never refunded after failure/rejection. Preferences are versioned; changing them during an active round aborts before aggregate release. Consumed examples are deleted after attempts. The protection is conditional on the public participation transcript; enrollment/participation existence is not hidden.

The stronger distributed-central-DP target below remains future work; its accountant must match its final sampling/dropout mechanism.

- Protection unit: one client contribution, composed across rounds; not automatically one person across multiple devices.
- Define adjacency, actual sampling mechanism, population assumptions, honest-noise contributors and collusion assumptions.
- Clip complete model deltas to an L2 bound S.
- Calibrate noise using a validated accountant matched to the actual sampling scheme, not an unrelated Poisson formula.
- Budget target epsilon and delta over a defined accounting horizon. Persist accumulated expenditure; UI values alone are not a guarantee.
- Evaluate finite precision, clipping, modular arithmetic and quantisation jointly. Quantisation as post-processing does not inherently destroy a valid prior DP guarantee, but per-client distributed quantisation and finite arithmetic require mechanism-specific analysis.
- Require enough surviving honest noise contributors; abort before release if assumptions fail. Do not lower noise silently after dropout.
- Conservatively reserve budget before participation and reconcile releases transactionally. Interrupted rounds must not accidentally reset expenditure or allow repeated disclosures.
- Do not calibrate clipping from private norms without an appropriate privacy-safe procedure.

### User settings

Implemented UI range: epsilon 1–10, step 0.5; this is a research configuration, not a certified production range. The fixed lifetime delta cap can prevent further rounds even when epsilon remains. Before release, validate supported presets, delta and a conservative default against the intended population and use case. Show target budget, accounted spend, remaining budget, accounting period and participation state.

Changing a setting affects future rounds only. If the new target is below already spent privacy, stop further participation and explain why. Turning learning off stops future contributions; it does not undo prior releases. A round uses a shared validated privacy configuration; admit only clients whose remaining budget permits it.

### Honest runtime states

Disabled · Awaiting eligible examples · Waiting for cohort · Training locally · Protecting update · Aggregating · Evaluating model · Up to date · Budget exhausted · Round aborted.

Settings replaces the separate demo FL tab. Technical diagnostics may be available under an advanced section, with appropriate access control.

## 9. Mascot behavior

Keep 70–80% of the robot unchanged. Prefer lightweight SVG/CSS overlays on the existing illustration rather than visually inconsistent regenerated frames.

| Intent                | Sequence                                                          |
| --------------------- | ----------------------------------------------------------------- |
| Calendar/schedule     | Idle → clock ring → calendar tiles → confirmation → idle          |
| Reminder/task         | Idle → attentive → notepad → writing → checkmark → idle           |
| Summarisation         | Idle → scanning → neural rings → compressed cards → ready → idle  |
| Failure/clarification | Attentive expression + restrained amber cue; no success checkmark |

Keywords can preview a state; the validated action and API result control success/failure. Pause nonessential motion offscreen and respect reduced-motion preferences. The supplied frame sets are now integrated as state-controlled WebP animations, with three exported reference GIFs. See ROBOT_ASSETS.md.

## 10. Technical architecture

| Layer               | Technology                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------- |
| UI                  | Angular 20, TypeScript, standalone components, RxJS; minimal custom CSS, optionally Bootstrap utilities |
| API                 | Python 3.11+, FastAPI, Uvicorn, Pydantic                                                                |
| Data                | PostgreSQL, SQLAlchemy, Alembic; isolated SQLite tests                                                  |
| Local ML            | PyTorch training; ONNX Runtime serving; optional TF-IDF fallback                                        |
| Optional local chat | Ollama or llama.cpp adapter                                                                             |
| Cloud               | Provider-neutral backend LLM adapter; explicit configured provider                                      |
| Security            | bcrypt, JWT sessions, AES-GCM, persistent revocation, CSRF                                              |
| FL/DP               | Authenticated client protocol, secure aggregation, validated DP accounting, model registry              |
| Background work     | Supervised client workers, persistent jobs; APScheduler for due reminders                               |
| Packaging           | Docker Compose and HTTPS reverse proxy                                                                  |

Proposed backend layers: routers → controllers → services → repositories → ORM models. Cross-cutting modules handle auth, consent, crypto, validation and auditing. Keep all browser URLs relative; proxy to backend services internally.

### Proposed persistent entities

Users, sessions, consents, calendar events, reminders, notes, optional conversations, audit entries, user settings, training jobs, privacy ledger, federated rounds and model versions. Client-local examples remain in local encrypted storage, not the coordinator database. Server round metadata must not include raw examples or plaintext individual deltas.

## 11. Release verification

### Security gates

- Cross-user CRUD tests for every resource.
- Password length, hashing, invalid/expired/revoked token and restart-persistence tests.
- CSRF, login throttle and cookie policy tests.
- Encrypt/decrypt/AAD tampering tests and database inspection proving sensitive fields remain ciphertext.
- Privacy-mode network tests covering success, failure, fallback and summarisation.
- No sensitive content in application logs or analytics.

### FL/DP gates

- Mask cancellation, dropout recovery, threshold and reveal-invariant tests.
- Authenticated round-state enforcement, replay and malformed-payload tests.
- Clipping bounds, known accountant reference values, composed expenditure and exhaustion tests.
- Abort safety for missing noise contributors, restart/retry safety and no single-client aggregate release.
- Model schema/version checks, quality evaluation and rollback.
- Independent security review before claims of production-grade cryptographic privacy.

### UI gates

- Entry → workspace → all navigation views.
- Light/dark persistence and reduced motion.
- Keyboard mode selection, dialogs, error messages and forms.
- CRUD flows, empty states, loading, failure and permission-denied states.
- No confirmation animations on failed operations.
- Mobile viewport and assistive-technology review.

No historical model accuracy, latency or DP numbers are acceptance evidence until reproduced against the actual implementation and hardware. Track inference and UI latency, completion rate, privacy-mode cloud requests (must be zero), aggregate quality and privacy expenditure in controlled tests without collecting raw user content.

## 12. Delivery phases

1. **Delivered foundation:** PRD, branding and themed Angular workspace.
2. **Delivered connected foundation:** backend, migrations, accounts, consent, encrypted CRUD, audit and protected entry; independent security review still required.
3. **Delivered assistant integration:** ONNX intent inference, local summaries, optional provider adapters, routing, confirmations, reviewed updates/deletions and scheduled best-effort browser push. Broader LLM capabilities require runtime/provider configuration.
4. **Delivered research learning integration:** consent-controlled examples, real client workers, conservative local DP, persistent budgets, all-member masking, quality-gated rollout and settings. Formal numerical/protocol review and distributed-device deployment remain future work.
5. **Release hardening:** accessibility, additional mascot polish, expanded end-to-end tests, threat-model review and deployment.

## 13. Explicit non-goals and open decisions

No claims of malicious-server resistance, Byzantine robustness, perfect anonymity, universal IDOR immunity or guaranteed mobile-model performance without evidence. Hardware enclaves, homomorphic encryption, PIR and model unlearning are deferred.

Before production, decide: trusted deployment boundary; external provider and retention terms; local conversational model/hardware; account recovery and email delivery; cohort size and participant authentication; DP adjacency/sampling/accounting horizon; retention policy; key management and backup; notification channels.

**Definition of done:** a genuinely authenticated, encrypted assistant with clearly enforced mode boundaries, honest processing provenance, validated privacy-preserving learning and usable light/dark interfaces—not simply a UI that displays those claims.


## Scheduled push and assistant CRUD update (2026-09-09)

Implemented opt-in browser Web Push with encrypted subscriptions, durable timed queues,
retries, revocation and generic lock-screen content. Delivery is best effort and requires
backend uptime and supported browser/OS settings; it uses browser vendor infrastructure
even in Privacy mode. Assistant Default/Privacy commands can list and prepare create,
update, complete/reopen and delete proposals for events/reminders. Explicit confirmation,
owner scoping and stale-version checks remain mandatory in the UI.

Local summarisation now uses configurable FLAN-T5-small / T5-small / DistilBART-CNN after optional model installation. No silent fallback is allowed. Global uses configured OpenAI, default
gpt-4o-mini. The separate bundled SNIPS public benchmark achieved 97.86% validation
accuracy on 700 examples after overlap removal; this is central public-data training,
not a private-user federated or DP accuracy claim. See NOTIFICATIONS_AND_COMMANDS.md
and models/snips/metrics.json for provenance, examples and delivery limitations.

Neural summary setup and offline/runtime test boundaries are documented in LOCAL_SUMMARIES.md. Real neural weights could not be downloaded in the restricted implementation sandbox.
