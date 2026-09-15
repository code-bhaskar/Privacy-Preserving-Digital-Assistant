#!/usr/bin/env python3
"""A simulated loopback LLM runtime, for demos and tests only.

WHY THIS EXISTS
    The local LLM intent classifier (`app/llm_intent.py`) is pinned to a
    loopback runtime and falls back silently to the bundled ONNX softmax model
    whenever that runtime is absent. That is correct behaviour, but it makes the
    LLM path impossible to demonstrate on a machine with no model downloaded, and
    impossible to test in CI. This script speaks just enough of the Ollama HTTP
    API on 127.0.0.1 to exercise the real client code end to end:

        LLM classification -> task routing -> review dialog -> confirmed example
        -> federated round -> Gaussian DP -> secure aggregation -> global model

WHAT IT IS NOT
    It is not a language model. It has no weights, no tokenizer and no
    generalisation: it is a deterministic keyword classifier over the six labels
    in `app/llm_intent.py`, plus a fixed reply for conversation. It cannot answer
    questions, and it will produce confident labels for text a real model would
    handle differently. Never present a run against this script as an LLM result,
    and never point a production deployment at it.

    To keep that mistake hard to make, the model name it advertises starts with
    `simulated-`, `app/llm_intent.probe()` says so in the operator-facing detail
    string, and every chat reply begins with "Simulated runtime".

USAGE
    .venv/bin/python scripts/fake_ollama.py --port 11435

    then in .env:
        OLLAMA_URL=http://127.0.0.1:11435
        OLLAMA_MODEL=simulated-local-llm:demo

    Port 11435 is the default so a real Ollama on 11434 is never shadowed.
    Binds loopback only; a non-loopback --host is refused.

Stdlib only: no dependencies beyond the Python this repository already requires.
"""

import argparse
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "simulated-local-llm:demo"
LOOPBACK = ("127.0.0.1", "localhost", "::1")
CLASSIFIER_MARKER = "Classify the user's request into exactly one label"

# Ordered: the first matching rule wins, so specific task evidence is tested
# before the general conversation and out-of-scope rules.
RULES = (
    (
        "summary",
        r"\b(summari[sz](?:e|es|ed|ing|er)|summary|summarise|tldr|tl;dr|key points|"
        r"condense|shorten|main points)\b",
    ),
    (
        "reminder",
        r"\b(remind|reminder|remainders?|remindar|todo|to-do|task|don'?t forget|"
        r"do not forget|ping me|nudge me|follow up|medicine|groceries|"
        r"remember to)\b",
    ),
    (
        "calendar",
        r"\b(calendar|calender|schedule|meeting|appointment|event|standup|"
        r"interview|sync|lunch with|book(?:ing)? (?:a|the))\b",
    ),
    (
        "note",
        r"\b(note|notes|notebook|jot|write (?:this )?down|save (?:this|a) "
        r"(?:thought|idea)|shopping list)\b",
    ),
    (
        "out_of_scope",
        r"\b(capital of|population of|president of|who (?:is|was|wrote)|weather|"
        r"forecast|news|stock|share price|history of|meaning of|definition of|"
        r"recipe for|distance between|translate|interpret|explain how|why (?:is|"
        r"does|do)|write (?:a|me) (?:poem|story|essay|email|code)|program|"
        r"debug|compile[rs]?|photosynthesis|algorithm|calculate|compute|solve|"
        r"recommend|suggest|compare|analyse|analyze|paraphrase)\b",
    ),
    (
        "chat",
        r"^\s*(?:hi|hello|hey|yo|namaste|good\s+(?:morning|afternoon|evening|"
        r"night)|thanks?|thank\s+you|bye|goodbye|ok|okay|sure|great|nice)\b[\s!.]*$"
        r"|\b(what can you do|who are you|what do you do|how do you work|"
        r"help me get started|tell me about yourself)\b",
    ),
)

CHAT_REPLY = (
    "Simulated runtime: I am a deterministic keyword stand-in for a local LLM, "
    "used to demonstrate routing without downloading a model. I cannot answer "
    "this. Install Ollama and pull a real model to get real answers."
)


def classify(prompt):
    """Extract the user's request from the classifier prompt and label it."""
    body = prompt
    marker = re.search(r"Request:\n(.*)", prompt, re.S)
    if marker:
        body = marker.group(1)
    body = re.sub(r"\n*Reply with the JSON object only\.\s*$", "", body).strip()
    lowered = body.lower()
    for label, pattern in RULES:
        if re.search(pattern, lowered, re.I):
            return label, body
    # A request that reached the classifier but matched no rule is out of scope,
    # which is the honest default: the local stack does calendar, reminders,
    # notes, summaries and small talk only.
    return "out_of_scope", body


class Handler(BaseHTTPRequestHandler):
    server_version = "PPDASimulatedRuntime/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[simulated-runtime] " + (fmt % args) + "\n")

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > 1_000_000:
            raise ValueError("Request too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        if self.path.rstrip("/") in ("/api/tags", "/api/tags/"):
            self._send(
                {
                    "models": [
                        {
                            "name": MODEL,
                            "model": MODEL,
                            "modified_at": "2026-01-01T00:00:00Z",
                            "size": 0,
                            "digest": "simulated",
                            "details": {
                                "family": "simulated",
                                "parameter_size": "0",
                                "quantization_level": "none",
                            },
                        }
                    ]
                }
            )
        elif self.path.rstrip("/") == "/api/version":
            self._send({"version": "0.0.0-simulated"})
        elif self.path.rstrip("/") in ("", "/"):
            self._send(
                {
                    "simulator": True,
                    "warning": "Not a language model. Deterministic keyword classifier for demos and tests.",
                    "model": MODEL,
                }
            )
        else:
            self._send({"error": "unknown route"}, status=404)

    def do_POST(self):
        if self.path.rstrip("/") != "/api/generate":
            self._send({"error": "unknown route"}, status=404)
            return
        try:
            request = self._read()
        except Exception:
            self._send({"error": "invalid json"}, status=400)
            return
        prompt = str(request.get("prompt") or "")
        if CLASSIFIER_MARKER in prompt:
            label, body = classify(prompt)
            response = json.dumps({"label": label})
            self.log_message("classify %r -> %s", body[:60], label)
        else:
            response = CHAT_REPLY
            self.log_message("chat %r -> canned reply", prompt[:60])
        self._send(
            {
                "model": str(request.get("model") or MODEL),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "response": response,
                "done": True,
                "done_reason": "stop",
                "total_duration": 1_000_000,
                "eval_count": 1,
                "simulator": True,
            }
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    args = parser.parse_args(argv)
    if args.host not in LOOPBACK:
        parser.error(
            "refusing to bind %r: this simulator must stay on loopback, like the "
            "real runtime the application accepts" % args.host
        )
    banner = f"""
==============================================================================
 SIMULATED LOCAL LLM RUNTIME - NOT A LANGUAGE MODEL
 listening on http://{args.host}:{args.port}   model: {MODEL}

 Deterministic keyword classifier over the six labels in app/llm_intent.py.
 For demos and tests only. Never describe its output as an LLM result.

 Point the backend at it with:
   OLLAMA_URL=http://{args.host}:{args.port}
   OLLAMA_MODEL={MODEL}
 Ctrl-C to stop.
==============================================================================
"""
    print(banner, flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSimulated runtime stopped.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
