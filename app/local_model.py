"""Small real local softmax intent model. No network code; no general-LLM claims."""

import hashlib
import re
from collections import Counter
import numpy as np

LABELS = ["calendar", "reminder", "note", "summary", "chat"]
FEATURES = 128
SHAPE = (FEATURES + 1, len(LABELS))
SEEDS = {
    "calendar": [
        "schedule a meeting tomorrow",
        "what is on my calendar today",
        "book an appointment",
        "plan an event next monday",
        "show my meetings",
        "add a calendar event",
        "schedule lunch with a friend",
        "meeting at five pm",
    ],
    "reminder": [
        "remind me to call mom tomorrow",
        "set a reminder at six",
        "remember to take medicine",
        "do not forget the groceries",
        "create a task",
        "show my reminders",
        "add a todo",
        "remind me to drink water",
    ],
    "note": [
        "write a note about my idea",
        "save a thought",
        "create a note",
        "show my notes",
        "take notes for the project",
        "write down this idea",
        "save this in my notebook",
        "note the shopping list",
    ],
    "summary": [
        "summarize this document",
        "summarise the following messages",
        "give me the key points",
        "tldr this text",
        "shorten this paragraph",
        "make a summary",
        "condense these messages",
        "extract the main points",
    ],
    "chat": [
        "hello how are you",
        "good morning",
        "thanks for your help",
        "what can you do",
        "tell me about yourself",
        "hi there",
        "help me get started",
        "nice to meet you",
    ],
}


def features(text: str):
    vector = np.zeros(FEATURES + 1)
    tokens = re.findall(r"[\w']+", text.lower())
    for token in tokens:
        index = (
            int.from_bytes(
                hashlib.blake2b(token.encode(), digest_size=8).digest(), "little"
            )
            % FEATURES
        )
        vector[index] += 1
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    vector[-1] = 1
    return vector


def probabilities(weights, text):
    logits = features(text) @ weights
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


def train(weights, examples, steps=35, lr=0.25):
    x = np.stack([features(t) for t, _ in examples])
    y = np.eye(len(LABELS))[[LABELS.index(label) for _, label in examples]]
    w = np.array(weights, dtype=float, copy=True)
    for _ in range(steps):
        p = x @ w
        p -= p.max(axis=1, keepdims=True)
        p = np.exp(p)
        p /= p.sum(axis=1, keepdims=True)
        w -= lr * (x.T @ (p - y) / len(x) + 0.001 * w)
    return w


def seed_weights():
    data = [(text, label) for label, texts in SEEDS.items() for text in texts]
    return train(np.zeros(SHAPE), data, steps=550, lr=0.8)


def sanity_score(weights):
    # Public seed regression gate, explicitly NOT a held-out accuracy benchmark.
    data = [(t, label) for label, texts in SEEDS.items() for t in texts]
    return sum(
        LABELS[int(probabilities(weights, t).argmax())] == label for t, label in data
    ) / len(data)


def classify(weights, text):
    from .onnx_model import probabilities as served_probabilities

    p = served_probabilities(weights, text)
    i = int(p.argmax())
    terms = re.findall(r"[\w']+", text)[:35]
    saliency = []
    for n, term in enumerate(terms):
        without = " ".join(terms[:n] + terms[n + 1 :])
        saliency.append(
            {
                "token": term,
                "contribution": float(p[i] - served_probabilities(weights, without)[i]),
            }
        )
    saliency.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    return LABELS[i], float(p[i]), saliency[:5]


def summary_body(text):
    return re.sub(
        r"^(summari[sz]e|summary|tldr|shorten this|give me the key points)\s*[:\-]?\s*",
        "",
        text.strip(),
        flags=re.I,
    )


def summarize(text):
    # Explicit legacy extractive mode, not a neural-model fallback.
    body = summary_body(text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", body) if s.strip()]
    if not sentences:
        return "Please include the text you want summarised."
    stop = set(
        "the a an is are was were to of and or in for this that it on with as at be by".split()
    )
    words = lambda s: [t for t in re.findall(r"\w+", s.lower()) if t not in stop]
    freq = Counter(t for sentence in sentences for t in words(sentence))
    score = lambda i: (
        sum(freq[t] for t in words(sentences[i]))
        / max(1, len(words(sentences[i]))) ** 0.5
    )
    keep = sorted(sorted(range(len(sentences)), key=score, reverse=True)[:3])
    return "\n".join("• " + sentences[i] for i in keep)
