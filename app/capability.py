"""Local-first capability router.

Every request is offered to the local stack first. This module answers one
question: *can this deployment actually do the work?* If it can, nothing ever
leaves the host. If it cannot, the caller may escalate to the global model after
applying `fl.text_dp` — the decision itself never sends anything anywhere.

The local stack can:
  * create / update / delete / list calendar events, reminders and notes,
  * run the configured local summariser,
  * answer deterministic greetings and capability questions,
  * hold a general conversation *only* if a loopback Ollama runtime is configured.

The bundled 128-feature intent classifier is reported and explained, but it is
deliberately not the sole gate: it is a small public-seed model and confidently
labels out-of-domain questions as tasks (measured: "what is the capital of
France" -> calendar 0.68). Deterministic task evidence therefore outranks it, and
explicit out-of-scope signals outrank a high-confidence task label. Staying local
is the default whenever the evidence is ambiguous.
"""

import re

from . import assistant_actions

# A task label must be at least this confident to be attempted without
# deterministic confirmation. Below it, and with no task pattern, the request is
# treated as out of scope.
TASK_CONFIDENCE = 0.55
TASK_LABELS = ("calendar", "reminder", "note", "summary")

SUMMARY = re.compile(
    r"\b(summari[sz](?:e|ed|er|ing)|summary|summarise|tldr|tl;dr|shorten|condense)\b|key points",
    re.I,
)
GREETING = re.compile(
    r"^\s*(?:hi|hello|hey|yo|namaste|good\s+(?:morning|afternoon|evening|night)|"
    r"thanks?|thank\s+you|bye|goodbye|ok|okay|sure|great|nice)\b[\s!.]*$",
    re.I,
)
CAPABILITY_QUESTION = re.compile(
    r"\b(?:what can you do|who are you|what do you do|how do you work|"
    r"what are you capable of|help me get started|what are my options)\b",
    re.I,
)
INTERROGATIVE = re.compile(
    r"^\s*(?:please\s+)?(?:what|whats|what's|who|whom|whose|why|how|when|where|"
    r"which|is|are|does|do|can|could|would|should|tell me about|explain|define|"
    r"describe|list the|name the)\b",
    re.I,
)
GENERATIVE = re.compile(
    r"\b(?:write|compose|draft|translate|interpret|convert|calculate|compute|solve|"
    r"code|program|debug|recommend|suggest|brainstorm|compare|contrast|analyse|"
    r"analyze|paraphrase|rewrite|generate|invent|imagine|critique|review this|"
    r"give me (?:a |an |some )?(?:idea|poem|story|joke|recipe|plan|advice))\b",
    re.I,
)
KNOWLEDGE = re.compile(
    r"\b(?:capital of|population of|president of|weather|forecast|news|score|"
    r"stock|share price|history of|meaning of|definition of|recipe for|who is|"
    r"who was|what is the|when was|distance between|best (?:way|place|phone|"
    r"laptop|movie|book)|how (?:do|does|can|to|much|many))\b",
    re.I,
)

# Reasons are shown to the user, so they name the missing capability plainly.
OUT_OF_SCOPE = (
    (KNOWLEDGE, "general knowledge or live information"),
    (GENERATIVE, "open-ended writing, analysis or code generation"),
    (INTERROGATIVE, "an open question the local task model cannot answer"),
)


class Route(dict):
    """Dictionary with attribute access, so it serialises straight into JSON."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _signals(text):
    return [reason for pattern, reason in OUT_OF_SCOPE if pattern.search(text)]


def local_task(text):
    """Deterministic evidence that a supported local task was requested."""
    if SUMMARY.search(text):
        return "summary"
    kind = assistant_actions.infer_kind(text)
    if kind:
        return {"Calendar": "calendar", "Reminders": "reminder", "Notes": "note"}[kind]
    return None


# An LLM reading the user's own text can be steered by that text, so an
# instruction-override attempt disqualifies the LLM label and falls back to the
# deterministic and ONNX evidence. This is a screen, not a proof.
INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|the\s+|your\s+)?"
    r"(?:previous|prior|above|earlier|system)\s*(?:instructions?|rules?|prompts?|messages?)?"
    r"|\bsystem\s+prompt\b|\byou\s+are\s+now\b|\bnew\s+instructions?\b"
    r"|\bact\s+as\s+(?:a|an|if)\b|\bpretend\s+(?:to\s+be|you\s+are)\b"
    r"|\b(?:classify|label|route)\s+this\s+as\b",
    re.I,
)


def screened_llm_label(text, label):
    """The local LLM's label after the instruction-override screen.

    One source of truth, used both by `assess` and by the caller that decides
    which task to execute, so a label disqualified for routing cannot sneak back
    in as the executed intent. Returns None when there is no label or when the
    text looks like an attempt to steer the classifier. This is a screen, not a
    proof: it catches the phrasings that actually appear, not all of them.
    """
    if not label:
        return None
    return None if INJECTION.search(text) else label


def assess(
    text,
    model_label,
    confidence,
    *,
    summary_ready=True,
    local_llm=False,
    task=None,
    llm_label=None,
):
    """Return the routing decision for one request. No I/O, no side effects.

    `task` lets the caller pass deterministic evidence it already resolved, such
    as a workspace kind recovered from a record ID ("Rename #12 to ..."), which
    no keyword in the sentence reveals.

    `llm_label` is the local LLM's reading, when a runtime is configured. It
    outranks the bundled softmax, which was measured guessing confidently on
    out-of-domain text, but it never outranks deterministic task evidence and it
    is discarded entirely if the text looks like an instruction-override attempt.
    """
    task = task or local_task(text)
    llm_label = screened_llm_label(text, llm_label)
    if task:
        return Route(
            capable=True,
            handler=task,
            capability=f"local.{task}",
            reason=(
                "Local intent model recognised a supported task; handled entirely "
                "on this device."
            ),
            evidence="deterministic task pattern"
            + (
                ""
                if model_label == task
                else f" (classifier said “{model_label}” at {confidence:.2f})"
            ),
            signals=[],
        )
    if GREETING.search(text) or CAPABILITY_QUESTION.search(text):
        return Route(
            capable=True,
            handler="chat",
            capability="local.greeting",
            reason="Greeting or capability question answered locally.",
            evidence="deterministic greeting pattern",
            signals=[],
        )
    if llm_label == "out_of_scope":
        return Route(
            capable=False,
            handler="global",
            capability="out-of-scope",
            reason=(
                "The local LLM read this as outside the supported tasks. The "
                "local stack handles calendar, reminders, notes and local "
                "summaries."
            ),
            evidence="local LLM classification",
            signals=["local LLM: out_of_scope"],
        )
    if llm_label in TASK_LABELS:
        return Route(
            capable=True,
            handler=llm_label,
            capability=f"local.{llm_label}",
            reason=(
                f"The local LLM classified this as “{llm_label}”; handled on "
                "this device."
            ),
            evidence=(
                "local LLM classification"
                + (
                    ""
                    if model_label == llm_label
                    else f" (softmax said “{model_label}” at {confidence:.2f})"
                )
            ),
            signals=[],
        )
    signals = _signals(text)
    if signals:
        return Route(
            capable=False,
            handler="global",
            capability="out-of-scope",
            reason=(
                f"Outside local capability: {signals[0]}. The local stack only "
                "handles calendar, reminders, notes and local summaries."
            ),
            evidence="out-of-scope signal",
            signals=signals,
        )
    if model_label in TASK_LABELS and confidence >= TASK_CONFIDENCE:
        return Route(
            capable=True,
            handler=model_label,
            capability=f"local.{model_label}",
            reason=(
                f"Local intent model classified this as “{model_label}” with "
                f"confidence {confidence:.2f}; attempting it locally."
            ),
            evidence="classifier confidence",
            signals=[],
        )
    if local_llm:
        return Route(
            capable=True,
            handler="chat",
            capability="local.llm",
            reason=(
                "The local LLM read this as conversation, and a loopback runtime "
                "is configured, so it stays on-device."
                if llm_label == "chat"
                else "A loopback local LLM is configured, so general chat stays on-device."
            ),
            evidence=(
                "local LLM classification" if llm_label == "chat" else "configured local LLM runtime"
            ),
            signals=[],
        )
    return Route(
        capable=False,
        handler="global",
        capability="out-of-scope",
        reason=(
            "General conversation is outside the local task model, and no local "
            "LLM runtime is configured on this host."
        ),
        evidence=(
            f"classifier said “{model_label}” at {confidence:.2f}, below the "
            f"{TASK_CONFIDENCE:.2f} task threshold"
        ),
        signals=[],
    )
