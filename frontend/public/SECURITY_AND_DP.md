# Security and DP implementation notes

## Deployment boundary

The browser submits text to FastAPI. Local model execution, encryption/decryption and training happen on the backend host. In a hosted preview, that is a remote host. Transport encryption does not prevent that host from seeing plaintext. The storage key is loaded from `.env`/environment; compromise of the running host is outside the protection boundary.

The implemented local intent model has 128 hashed features plus a bias and five labels, with 645 trainable weights. NumPy trains it; a dimension-checked in-memory ONNX graph serves it with one CPU thread. This is a new small model, not the previously described 68k-parameter IntentNet or a general LLM. Public seed training/regression examples bootstrap it. No generalisation accuracy or phone-latency claim is made.

## Accounts and data

bcrypt rejects passwords under 12 characters or above 72 UTF-8 bytes rather than truncating. Session JWTs use HS256 with issuer, audience, required expiry and a random session ID. The database stores expiry/revocation across restart. HttpOnly cookies prevent direct JS token reads; CSRF tokens are returned to the signed-in UI and kept in memory. Embedded HTTPS previews can opt into `SameSite=None; Secure; Partitioned` cookies; normal same-origin deployments default to Lax. SameSite=None or Partitioned without Secure is rejected at startup. State-changing endpoints require CSRF tokens, except initial login/registration, which enforce same-origin browser requests. Request bodies are bounded even without Content-Length.

Rate limits are persistent: five attempts per normalized email and forty per direct peer in five minutes. Behind a proxy, peer limits are intentionally shared instead of trusting spoofable forwarding headers. Production should add a trusted edge limiter with a correctly configured proxy chain. Session revocation and budget state are not in-memory blocklists.

Records are selected using authenticated ownership, and request models reject extra fields. Workspace title/content and local training examples are AES-256-GCM ciphertext. HKDF derives distinct per-user keys; AAD binds user, purpose/type and ciphertext format. Account name/email, record kind, due timestamps, statuses, consent and accounting metadata remain readable in the database. Never claim whole-database encryption or zero metadata leakage.

Conversation messages are not persisted unless a user explicitly supplies an input as a training example or saves a task/note. Cloud requests include only the current prompt; no history or workspace context is attached. Source text is not logged in audit reasons.

Audit records use per-user SHA-256 linkage and a keyed HMAC over each digest. SQL triggers reject UPDATE/DELETE. This detects changed records without the signing key, but not a privileged deletion of the whole DB or a consistent tail truncation: there is no independently stored checkpoint. Ledger entries are also append-only; DB administrators and key holders remain trusted.

## Federated protocol actually implemented

This is a single-host, honest-participant research pipeline. The coordinator has no ordinary code path that receives an individual unmasked model delta. Each child worker receives only its intended encrypted examples, public shared weights and its user key over its private stdin pipe. It returns its X25519 public key and a masked uint32 vector. Worker environment variables do not include API/JWT/master secrets. All processes still share a privileged host; that host can inspect memory or files. No physical-device security boundary is established.

For each round, three real opted-in account queues are selected. Each needs at least three confirmed examples; up to twenty are consumed. X25519 derives pair secrets; HKDF binds the round nonce; ChaCha20 produces pairwise pseudorandom uint32 masks. Signs are opposite for each pair. Addition modulo 2^32 cancels masks only in the sum of all participants. At least two noncolluding participants and an honest coordinator are assumed. This provides computational masking, not a one-time-pad information-theoretic guarantee.

There are no self masks, Shamir shares, remote client enrollment, Byzantine filtering, malicious-server consistency proofs or dropout recovery. **Any missing member aborts the entire round.** The code never exposes recovery keys. Private OS pipes bind the messages to the worker processes started by the supervisor; this is not a distributed-network authentication scheme.

A database transaction reserves budgets and marks examples used before workers start. A process-level file lock rejects a second backend against the same SQLite database. Transactions use BEGIN IMMEDIATE to serialize audit tails, eligibility checks and reservations. Restart marks interrupted rounds aborted without refunding expenditure. Every preference save increments a server-controlled version; a mismatch before aggregate release aborts that round. Revoking a local data category purges all unused queued examples conservatively. Used examples are deleted after the attempt or during restart recovery.

## Differential privacy contract

**Implemented:** client-local Gaussian mechanism, followed by deterministic quantisation/masking, with conservative basic sequential composition. **Not implemented:** subsampled RDP or distributed-central Gaussian calibration.

Adjacency replaces one client's complete local training dataset, for a fixed public participation/cohort transcript. Participation itself, sample availability, enrollment timing and the existence of a user account are not protected by this guarantee. One person with several accounts is not automatically one protected unit. Do not interpret this as user anonymity or participation-hiding DP.

For a model delta Δ, clip to `clip(Δ, C)` with `C=0.1`. Two clipped deltas differ by at most `2C` in L2. Each worker adds independent Gaussian noise with standard deviation

```
sigma = 1.01 × (2C) × sqrt(2 ln(1.25 / delta_round)) / epsilon_round
```

The sufficient classical Gaussian bound (Dwork & Roth, The Algorithmic Foundations of Differential Privacy, Theorem A.1) is used only for `0 < epsilon_round <= 1`. Current constants are ε_round=0.5 and δ_round=10^-6. There is no subsampling amplification claim and no dependence on other clients honestly adding their noise for this local mechanism. Clipping/noise are applied to the whole client update, not to individual example gradients.

Each participant independently perturbs its update before aggregation, so losing participants does not silently reduce an assumed distributed noise total. The separate masking protocol still aborts on dropout for confidentiality/correctness. This local-DP design imposes substantially greater noise than central-DP secure aggregation and will often reject candidates at this small scale.

Quantisation uses scale 100,000 and saturates each already-noised vector to ±floor((2^30−1)/n) before conversion to uint32. This is deterministic post-processing of each local DP output and avoids signed overflow of the aggregate sum under the supported cohort bound. It is not correct to say that any quantisation automatically destroys DP. The exact finite-precision sampling and arithmetic implementation nevertheless need security review.

The sampler uses OS-backed randomness through Python SystemRandom.normalvariate, not a deterministic NumPy seed. It is floating-point sampling, not an audited discrete Gaussian implementation. Consequently the ideal mathematical mechanism and the research implementation must not be conflated with a certified numerical DP guarantee.

**Ledger:** ε_total=sum ε_round, δ_total=sum δ_round for the account's lifetime. A round reserves the full cost before training. Rejection, dropout, timeout, preference changes and retries do not refund it. This may overcount but avoids undercounting releases. Every new participation must fit both the user epsilon target and fixed δ_total<=10^-5. Epsilon changes apply only to future eligibility; setting a target below already-spent epsilon blocks future participation. Recreating accounts is not a privacy-accounting reset for the same person's data; identity linking/enrollment is a deployment research problem, not solved here.

Candidate quality scores and publication/rejection decisions use the noised aggregate and public seed data, so they are post-processing of the protected updates. The release does not report raw local training accuracy, losses, gradient norms or private-data-dependent clipping calibration.

## Model release

The shared candidate is the old public model plus the averaged protected aggregate. It must be finite, dimension compatible and pass the public-seed regression gate (at least 0.75 and no more than 0.025 below the previous model). Rejected candidates are not activated or persisted as individual vectors. This gate is a safety regression check, **not a held-out benchmark** or poisoning defense.

ONNX graphs are regenerated in memory from validated dimensions for the active version. No arbitrary downloaded ONNX file replaces the assistant model. The external OpenAI LLM is a separate service and is not trained by this pipeline.

## Validation and unimplemented hardening

Tests cover account/session behavior, CSRF/origin rules, owner isolation, ciphertext/AAD, audit triggers, ONNX parity/shape checks, clipping/calibration, real mask cancellation, real three-process training, persistent budgets, opt-out, and dropout-abort behavior. Browser tests use an isolated real backend/database rather than response mocks. Provider calls are mocked; no paid live-provider test was performed.

A test pass does not prove cryptographic protocol security or DP. Independent review, a formally specified discrete/noise implementation, sybil-resistant physical-device enrollment, key rotation/keystore integration, externally anchored audits, email verification/recovery, production transport hardening and large-cohort privacy–utility evaluation remain required for production claims.


## Optional browser notification egress

Browser push is an independent opt-in, including in Privacy mode. Browser vendors receive
routing/timing/network metadata; encrypted payloads contain only generic due notifications,
not titles, notes, prompts or user identifiers. Subscription material is encrypted per user;
provider destinations are allowlisted and redirects refused. The service worker does not
cache private data or tokens. Revocation cancels queued jobs, but cannot recall a push already
accepted or in flight. See NOTIFICATIONS_AND_COMMANDS.md for timing, retry and delivery limits.
SNIPS artifacts are trained on public data and kept separate from the private-user FL model.
Their benchmark accuracy is not a DP or federated-training result.
