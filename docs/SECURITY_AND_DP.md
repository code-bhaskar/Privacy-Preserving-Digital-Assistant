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

**Two stages share that protocol.** `LEARNING_STAGE` selects which object is
federated, and nothing else about the round changes:

| | `softmax` (default) | `lora` |
|---|---|---|
| Released per client | the full delta over `(F+1)×L` = 645 weights | a rank-`r` adapter, `r×L` = 20 numbers at `r=4` |
| Base weights | moved by the aggregate | frozen; the adapter is merged only if the gate passes |
| ε, δ per round | 0.5, 10⁻⁶ | 0.5, 10⁻⁶ (identical ledger charge) |
| Clip norm `C` | 0.1 | `LORA_CLIP_NORM`, default 0.1 |
| Masking, cohort, consent re-check, gate, publication table | the same code | the same code |

The adapter stage releases fewer coordinates, and the Gaussian mechanism noises
each released coordinate, so less noise reaches the served model: measured over
200 simulated rounds at the shipped defaults, the merged perturbation norm is
31.5 for the full matrix against 5.5 for a rank-4 adapter, a ratio of 0.174 that
tracks the predicted `sqrt(20/645) = 0.176`. Publication rates in the same
simulation were 0% and 98%. Reproduce with
`scripts/measure_dp_stages.py`; `tests/test_lora_fl.py` pins the ratio. This is a
utility difference at identical privacy cost, **not** a stronger guarantee, and
not an accuracy claim: at ε=0.5 with three clients a published adapter is still
noise-dominated, and the gate asks only whether the served model broke.

The projection `A` (shape `(F+1, r)`) is public and derived from a committed seed
by hashing, so clients and server agree on it without transmitting it and it
cannot drift between NumPy versions. Raising the rank extends `A` rather than
reshuffling it, so an adapter published at rank `r` remains meaningful at `r+1`.

A database transaction reserves budgets and marks examples used before workers start. A process-level file lock rejects a second backend against the same SQLite database. Transactions use BEGIN IMMEDIATE to serialize audit tails, eligibility checks and reservations. Restart marks interrupted rounds aborted without refunding expenditure. Every preference save increments a server-controlled version; a mismatch before aggregate release aborts that round. Revoking a local data category purges all unused queued examples conservatively. Used examples are deleted after the attempt or during restart recovery.

## Differential privacy contract

**Implemented:** client-local Gaussian mechanism, followed by deterministic quantisation/masking, with conservative basic sequential composition. **Not implemented:** subsampled RDP or distributed-central Gaussian calibration.

Adjacency replaces one client's complete local training dataset, for a fixed public participation/cohort transcript. Participation itself, sample availability, enrollment timing and the existence of a user account are not protected by this guarantee. One person with several accounts is not automatically one protected unit. Do not interpret this as user anonymity or participation-hiding DP.

For a model delta Δ, clip to `clip(Δ, C)` with `C=0.1` in the default stage and
`C=LORA_CLIP_NORM` in the adapter stage. `C` is a parameter of the mechanism, not
a constant, and the noise is always calibrated to the same `C` that was used for
clipping; changing one without the other would break the sensitivity bound, which
is why both are validated at startup (`Settings.lora_parameters`). Two clipped
deltas differ by at most `2C` in L2. Each worker adds independent Gaussian noise with standard deviation

```
sigma = 1.01 × (2C) × sqrt(2 ln(1.25 / delta_round)) / epsilon_round
```

The sufficient classical Gaussian bound (Dwork & Roth, The Algorithmic Foundations of Differential Privacy, Theorem A.1) is used only for `0 < epsilon_round <= 1`. Current constants are ε_round=0.5 and δ_round=10^-6. There is no subsampling amplification claim and no dependence on other clients honestly adding their noise for this local mechanism. Clipping/noise are applied to the whole client update, not to individual example gradients.

Each participant independently perturbs its update before aggregation, so losing participants does not silently reduce an assumed distributed noise total. The separate masking protocol still aborts on dropout for confidentiality/correctness. This local-DP design imposes substantially greater noise than central-DP secure aggregation and will often reject candidates at this small scale. The adapter stage reduces the *dimension* of the release, which is the one lever that improves the merged signal-to-noise ratio without spending more ε: raising the clip raises signal and noise together and leaves the ratio unchanged, while a larger cohort divides the noise by `sqrt(n)`.

Quantisation uses scale 100,000 and saturates each already-noised vector to ±floor((2^30−1)/n) before conversion to uint32. This is deterministic post-processing of each local DP output and avoids signed overflow of the aggregate sum under the supported cohort bound. It is not correct to say that any quantisation automatically destroys DP. The exact finite-precision sampling and arithmetic implementation nevertheless need security review.

The sampler uses OS-backed randomness through Python SystemRandom.normalvariate, not a deterministic NumPy seed. It is floating-point sampling, not an audited discrete Gaussian implementation. Consequently the ideal mathematical mechanism and the research implementation must not be conflated with a certified numerical DP guarantee.

**Ledger:** ε_total=sum ε_round, δ_total=sum δ_round for the account's lifetime. A round reserves the full cost before training. Rejection, dropout, timeout, preference changes and retries do not refund it. This may overcount but avoids undercounting releases. Every new participation must fit both the user epsilon target and fixed δ_total<=10^-5. Epsilon changes apply only to future eligibility; setting a target below already-spent epsilon blocks future participation. Recreating accounts is not a privacy-accounting reset for the same person's data; identity linking/enrollment is a deployment research problem, not solved here.

Candidate quality scores and publication/rejection decisions use the noised aggregate and public seed data, so they are post-processing of the protected updates. The release does not report raw local training accuracy, losses, gradient norms or private-data-dependent clipping calibration.

## Optional local LLM intent classification

When `OLLAMA_URL`/`OLLAMA_MODEL` are configured, `app/llm_intent.py` asks that
loopback runtime for a label from a fixed set
(`calendar|reminder|note|summary|chat|out_of_scope`) with `format: json` and
`temperature: 0`. Its reading outranks the bundled 128-feature softmax **and is
the task that gets executed**: when deterministic parsing finds no task, the
adopted label decides which review dialog is prepared, so a paraphrase such as
"ping me about the dentist visit" becomes a reminder draft instead of small talk.
The response reports this as `intent_source: "local-llm"`. The trust boundary is
still deliberately narrow:

- **Loopback only.** A non-loopback URL is refused before any request is made, so
  intent classification can never leave the host even if misconfigured.
- **Fixed output space.** A label outside the set, an unparseable body, a
  transport error or a timeout all return `None` and the softmax label stands.
  Nothing is coerced into a valid-looking label.
- **Never authoritative, never a write.** Deterministic task evidence still wins
  (a record ID, an explicit `Summarize:` prefix, a workspace keyword), and no
  label writes a record: create/update/delete still require the review dialog and
  confirmation against an item version hash. A misread therefore costs the user a
  draft they dismiss, not a change to their data.
- **`summary` is never adopted.** Summarisation needs source text after
  `Summarize:`, which only the deterministic signal guarantees, so an LLM label
  cannot invent a summary request.
- **Category consent gates adoption.** If the label's category is not consented,
  the message is answered as conversation and the reply says so, rather than
  turning a guessed label into a 403.
- **Screened for execution too.** The same instruction-override screen that
  disqualifies the label for routing disqualifies it for the executed task
  (`capability.screened_llm_label` is the single source of truth), and the
  response reports `llm_intent_used: null` so the discard is visible.
- **Observable.** A silent fallback is an invisible one, so
  `llm_intent.probe()` reports whether the runtime is reachable, whether the
  configured model has actually been pulled, and which models are available.
  `/api/v1/runtime` surfaces it; the probe is cached for 30 s, bounded at 3 s and
  never raises.
- **Injection screened.** A message matching an instruction-override pattern
  disqualifies the LLM label, because a model reading the user's own text can be
  steered by it. The screen is a heuristic, not a proof; the confirmation step is
  what actually bounds the damage.

Cost: one extra loopback inference per message when a runtime is configured.

`scripts/fake_ollama.py` is a **simulator** that speaks the same loopback API for
demos and tests. It is a deterministic keyword classifier with no weights and no
generalisation. It advertises a model name in the `simulated-` family, which
`probe()` and `engine_label()` report as a simulator, so a settings page or a
screenshot cannot present it as a language model result.

## Prompt escalation accounting

Out-of-capability requests may be released to the global model. That release is a
different mechanism from the training-round Gaussian mechanism above, and it is
accounted separately.

**Mechanism.** `fl/text_dp.py` first applies deterministic de-identification
(e-mail addresses, URLs, phone-like and long digit runs, IP addresses,
`key: value` credentials). Words outside the committed public vocabulary
(`fl/public_vocabulary.txt`, rebuilt from this repository's public documentation
by `scripts/build_public_vocabulary.py`) are replaced by `[redacted]`, because
the mechanism cannot preserve a symbol outside its output space; rare words and
proper nouns therefore disappear. Remaining content words go through k-ary
randomised response over that vocabulary: the true word is emitted with
probability `e^ε / (e^ε + k - 1)`, otherwise a uniformly random different
vocabulary word. Function words are passed through verbatim as a documented
utility choice, so sentence shape and word order are **not** protected. Because
the output space is derived from this repository's public documentation, a word
that appears in that documentation is preservable by the mechanism; that is a
property of any public vocabulary, not a leak of user data. Regenerating the list
changes `k` and therefore the retention probability, which is why the asset
carries a SHA-256 digest and `scripts/build_public_vocabulary.py --check`
verifies it.

**What is guaranteed.** Each protected token's release is `ε_token`-LDP. The
default `ESCALATION_TOKEN_EPSILON=10` with the 2,824-word public vocabulary
retains about 89% of protected words. **This is a per-token statement.** Under
basic sequential composition a prompt with `n` protected tokens is at most
`n × ε_token`-private; the API returns that bound as `composition_bound` and the
UI displays it. It is not enforced as a cap, and it must not be quoted as the
prompt's epsilon. The provider's reply is not a DP release at all. Most of the
practical protection here comes from redaction, not from the noise.

**Ledger.** Each *successful* escalation charges the account
`EPSILON_ESCALATION = 0.25` and `DELTA_ESCALATION = 0` (k-RR is pure ε-DP, so no
δ is invented). Escalation rows carry `round_id = NULL`, which migration `0003`
permits; the append-only triggers are reinstated by that migration. This charge
is a **policy budget on the number of escalated releases** an account may make —
with the default ε target of 3.0, twelve escalations. It is deliberately *not*
the composed text-DP loss, and the two numbers must not be conflated. A failed
provider call is not charged, because nothing was published; a blocked escalation
is never charged.

**Refusals.** Escalation does not happen in Privacy mode, without cloud consent,
without a configured provider, when the sensitive-keyword guard matches in
Default mode, or when the budget cannot cover the release. In Default mode a
refusal degrades to a local explanation; in explicit Global mode it returns
403/503/428 so an operator misconfiguration is not hidden behind a local answer.

## Model release

The shared candidate is the old public model plus the averaged protected aggregate. In the adapter stage the candidate is `W0 + A @ B̄`, where `B̄` is the averaged protected adapter and the merge is deterministic post-processing of an already-private release. Either way it must be finite, dimension compatible and pass the public-seed regression gate (at least 0.75 and no more than 0.025 below the previous model). Rejected candidates are not activated or persisted as individual vectors. This gate is a safety regression check, **not a held-out benchmark** or poisoning defense.

An accepted adapter is merged into a new `model_versions` row, so the ONNX serving path, the API and the browser are unchanged by which stage ran; the adapter itself is also recorded in `lora_adapters` with its base model, merged model, gate score and acceptance flag. A **rejected** adapter keeps its row with `accepted = false` and no merged model, because the budget for it was spent and the refusal must stay auditable. Adapter rows hold the aggregated, already-noised vector and no example text. Rounds carry a `stage` column so one history tail shows which mechanism produced each round.

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
