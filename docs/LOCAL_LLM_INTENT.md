# Local LLM intent classification, and the two federated stages

This is the operator guide for the part of the system that answers: *"what did
the user ask for, who decided, and what was learned from it?"*

```
your message
   │
   ├─ 1. normaliser (offline)          app/normalizer.py
   ├─ 2. ONNX softmax classifier       app/local_model.py + app/onnx_model.py
   ├─ 3. local LLM classifier          app/llm_intent.py   ← Ollama on loopback
   ├─ 4. capability router             app/capability.py
   ├─ 5. executed task + review draft  app/assistant_actions.py
   │        └─ you confirm ──────────► encrypted training example
   │
   └─ 6. federated round               fl/pipeline.py  or  fl/lora/pipeline.py
            client-local Gaussian DP → pairwise masking → aggregation
            → public-seed gate → published as the new global shared model

   out-of-capability only:
   └─ token-level DP (fl/text_dp.py) → cloud global model, charged to your ledger
```

Two different things are called "the global model" here, and they are separate:

* **The aggregated shared model** — `model_versions`, produced by federated
  rounds from your own confirmed examples. It never leaves the host. This is the
  "send updates to the global model" of federated learning.
* **The cloud model** — an external provider (OpenAI by default) that only ever
  receives a de-identified, token-DP-perturbed prompt when you are out of local
  capability and consent is granted. It is not trained by this system.

---

## 1. Install a local LLM runtime

The classifier talks to an [Ollama](https://ollama.com) runtime on **loopback
only**. A non-loopback URL is refused before any request is made: intent
classification must never leave the host.

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &            # or: systemctl enable --now ollama

# pick ONE small model (approximate sizes; verify with `ollama list`)
ollama pull qwen2.5:0.5b  # ~400 MB,  fastest, fine for 6-way classification
ollama pull qwen2.5:1.5b  # ~1.0 GB,  good balance on 4 GB RAM
ollama pull llama3.2:1b   # ~1.3 GB
ollama pull llama3.2:3b   # ~2.0 GB,  best of the small set, wants 8 GB RAM
```

For a six-label classification task a 0.5B–3B model is plenty; you are not asking
it to write prose. If the host has 3–4 GB of RAM, start with `qwen2.5:0.5b` and
only move up if labels are wrong.

## 2. Point the backend at it

In `.env` (never in the browser, never committed):

```bash
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:1.5b
```

Then restart the backend. The same runtime also powers offline chat in Privacy
mode, so this one setting enables both.

## 3. Verify it is really being used

A silent fallback is an invisible one, so there are three independent checks.

**a. The API.** `GET /api/v1/runtime` (signed in) returns:

```json
{
  "intent_classifier": "Local LLM (qwen2.5:1.5b) with ONNX softmax fallback",
  "local_llm": true,
  "learning_stage": "softmax",
  "llm_runtime": {
    "configured": true,
    "loopback": true,
    "reachable": true,
    "model": "qwen2.5:1.5b",
    "model_present": true,
    "available_models": ["qwen2.5:1.5b"],
    "simulator": false,
    "detail": "Local LLM “qwen2.5:1.5b” is reachable on loopback and is classifying intent; its label outranks the softmax model."
  }
}
```

`model_present: false` means Ollama is running but the model was never pulled —
the `detail` string then contains the exact `ollama pull` command.
`reachable: false` means the runtime is down and every request fell back to the
softmax model.

**b. The UI.** Settings → *Runtime and learning* shows the same sentence, plus
which federated stage is active and its real parameters.

**c. A message.** Send `ping me about the dentist visit` — no reminder keyword in
it. With a working runtime the reply is a **reminder draft** and the response
carries:

```json
"intent": "reminder",
"intent_source": "local-llm",
"llm_intent": "reminder",
"llm_intent_used": "reminder",
"model_intent": "chat",
"confidence": 0.69,
"router": { "evidence": "local LLM classification (softmax said “chat” at 0.69)" }
```

Without a runtime, `llm_intent` is `null`, `intent_source` is `"default"` and the
message is answered as conversation. That difference is the whole point: the LLM
label **selects the executed task**, it is not merely displayed next to it.

### What the LLM label can and cannot do

| It can | It cannot |
|---|---|
| Choose which review dialog is prepared | Write, update or delete a record |
| Mark a request out of scope so it escalates | Override deterministic evidence (a record ID, `Summarize:`) |
| Be adopted as the task (`intent_source: "local-llm"`) | Be adopted as `summary` — that needs source text |
| Be discarded by the injection screen (`llm_intent_used: null`) | Be used at all if its category consent is off |

Nothing is saved until you confirm the draft, and a confirmed write is checked
against the item's version hash.

## 4. Choose the federated stage

`LEARNING_STAGE` selects what a round federates. Everything else — cohort rules,
consent re-checks, the lifetime ε/δ ledger, X25519/HKDF/ChaCha20 pairwise
masking, the public-seed publication gate, the `model_versions` table that serves
predictions — is the same code either way.

```bash
LEARNING_STAGE=softmax   # default: federate the full 645-weight shared matrix
LEARNING_STAGE=lora      # freeze that matrix; federate a rank-r adapter instead
```

The `lora` stage learns `W = W0 + A @ B`: `W0` (the published model) is frozen,
`A` is a public projection derived from a committed seed, and `B` is a rank-`r`
adapter initialised to zero — LoRA (Hu et al., 2021) applied to the classifier
head. A client trains only `B` on its own decrypted examples and releases those
`r × L` numbers.

```bash
LORA_RANK=4              # released vector = rank × labels = 20 numbers
LORA_CLIP_NORM=0.1       # sensitivity bound; the Gaussian scale is calibrated to it
LORA_LOCAL_STEPS=60      # client-side steps before the release
LORA_LEARNING_RATE=0.5
```

**Why release an adapter instead of the matrix.** The Gaussian mechanism adds
noise to *every released coordinate*, and all of it reaches the served model.
Releasing 20 coordinates instead of 645 therefore injects about
`sqrt(20/645) ≈ 0.176` of the noise energy, at the *same* ε, δ, clip and ledger
charge. Measured over 200 simulated rounds at the shipped defaults:

| stage | released | σ/coord | merged noise norm | vs model norm | published |
|---|---|---|---|---|---|
| full matrix | 645 | 2.14 | 31.5 | 1.9× | 0% |
| adapter rank 1 | 5 | 2.14 | 2.7 | 0.2× | 100% |
| adapter rank 4 | 20 | 2.14 | 5.5 | 0.3× | 98% |
| adapter rank 8 | 40 | 2.14 | 7.9 | 0.5× | 94% |
| adapter rank 16 | 80 | 2.14 | 11.0 | 0.6× | 71% |

Reproduce with:

```bash
.venv/bin/python scripts/measure_dp_stages.py --trials 200
```

**Read that table honestly.** It is a simulation of the *mechanism's noise*
merged into the public seed model — no private data, no accuracy claim. The
publication gate asks "did this break the served model?", not "did this improve
it?". At ε=0.5 with three clients a published round is still noise-dominated
(σ=2.14 per coordinate against a clipped signal norm of at most 0.1), and the
`stage_detail.note` field in `/api/v1/learning/status` says so in the running
system. The levers are cohort size (noise ÷ √n), ε (more spend, less noise) and
rank (dimension); the clip is *not* a lever, because raising it raises signal and
noise together.

A round leaves an auditable row in `lora_adapters`: the aggregated (already
noised) adapter, the base model, the model it merged into, the gate score, and
whether it was accepted. A rejected adapter keeps its row with `accepted = false`
and no merged model, because the budget was spent. Rounds also carry a `stage`
column, so one history tail shows which mechanism produced each round.

**No causal-LLM adapter stage.** Ollama's API cannot hot-load a PEFT adapter, so
a LoRA round over llama/qwen weights would publish an artefact nothing here could
serve. `fl/lora/backend.py` documents the interface an operator would implement
if they had local encoder weights *and* an in-process serving path.

## 5. Demo the whole loop without downloading a model

`scripts/fake_ollama.py` speaks the same loopback API. **It is not a language
model** — it is a deterministic keyword classifier over the six labels, with no
weights and no generalisation. It exists so the routing, the FL round, the DP
mechanism and the aggregation can be demonstrated and tested on a machine with
no model, and so CI can exercise the real HTTP client code.

```bash
.venv/bin/python scripts/fake_ollama.py --port 11435
```

```bash
OLLAMA_URL=http://127.0.0.1:11435
OLLAMA_MODEL=simulated-local-llm:demo
```

It binds loopback only (a non-loopback `--host` is refused), prints a banner
saying it is a simulator, and answers chat with "Simulated runtime: …". The model
name is in the `simulated-` family, which `probe()` and `engine_label()` report
as a simulator, so a screenshot of the settings page cannot be mistaken for an
LLM result. Never quote a run against it as a language-model measurement.

A complete offline demonstration:

```bash
# terminal 1 — simulated runtime
.venv/bin/python scripts/fake_ollama.py --port 11435

# terminal 2 — backend, with LEARNING_STAGE=lora in .env
COOKIE_SECURE=false COOKIE_SAMESITE=lax COOKIE_PARTITIONED=false \
  .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

# terminal 3 — frontend
cd frontend && npm start -- --port 4200
```

1. Create an account, grant assistant + calendar + notes consent and enable
   *training* in Settings.
2. Send `ping me about the dentist visit`. The reply is a reminder draft, and the
   explanation panel says the task was selected by the local LLM.
3. Give it a time in the review dialog and save. The example is queued encrypted,
   with `source: "local-llm"` inside the ciphertext.
4. Repeat with two more accounts (three eligible clients is the minimum; a cohort
   is never padded or simulated).
5. Settings → *Check training eligibility*. The round spawns three real OS client
   workers; the history row then reads, for example:

   `Round 1 · published · lora — Rank-4 adapter over a frozen base: 20 released
   coordinates instead of 645. Public seed regression score 0.975; merged into
   model v2. Gaussian sigma 2.14 per coordinate at eps=0.5, clip=0.1. Budget
   consumed.`

6. Settings shows ε spent 0.5 and δ spent 10⁻⁶ against your lifetime target, and
   the audit log carries `FL_BUDGET_RESERVED` and `FL_LORA_PUBLISHED`.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `intent_classifier` says "ONNX Runtime · 128-feature softmax intent model" | `OLLAMA_URL`/`OLLAMA_MODEL` empty | set both in `.env`, restart |
| "configured LLM runtime is not loopback, so it is ignored" | remote or LAN URL | intent classification is loopback-only by design |
| `reachable: false`, "did not answer" | runtime not running | `ollama serve` |
| `model_present: false` | model never pulled | the `detail` string contains the `ollama pull` command |
| Classification is slow | large model on CPU | use `qwen2.5:0.5b`/`llama3.2:1b`; the 20 s timeout falls back to ONNX |
| `llm_intent_used: null` | injection screen fired | expected for "ignore previous instructions…" style text |
| LLM says reminder, reply is chat | category consent off | the reply's notes say so; enable it in Settings |
| Rounds never start | fewer than three eligible clients, or budget exhausted | check `state` in `/api/v1/learning/status` |
| Round `rejected` | noisy candidate failed the public-seed gate | normal at ε=0.5; budget is intentionally not refunded |
| Round `aborted` | client dropout, consent revoked mid-round, or bad vector dimensions | any dropout aborts; there is no recovery-secret release |

## 7. Limits, stated plainly

* **Local means the backend host, not the browser.** In a remotely hosted
  preview, its operator can read plaintext during processing and holds the
  storage keys.
* **The LLM sees your plaintext.** That is the point of a local runtime and the
  reason it is pinned to loopback; it is also why a remote URL is refused rather
  than warned about.
* The injection screen is a heuristic, not a proof. The confirmation dialog is
  what actually bounds the damage.
* The FL implementation is single-host with three real OS workers. It does not
  establish physical-device isolation, sybil resistance or malicious-server
  security, and any dropout aborts the round.
* The DP mechanism is client-local Gaussian with basic sequential composition —
  not distributed-central DP, not RDP, and not an audited discrete Gaussian
  sampler. Do not describe it as a certified production guarantee.
* The simulator is not a model, and the public-seed gate is not a benchmark.

See [SECURITY_AND_DP.md](SECURITY_AND_DP.md) for the mechanism contracts and
[NOTIFICATIONS_AND_COMMANDS.md](NOTIFICATIONS_AND_COMMANDS.md) for the SNIPS
benchmark and command reference.
