"""Token-level differential privacy for prompts escalated to the global model.

What is implemented
-------------------
1. Deterministic de-identification: e-mail addresses, URLs, phone-like and long
   digit runs, IP addresses and `key: value` secrets are replaced before any
   probabilistic step. Redaction is deterministic, so it is *not* a DP
   mechanism; it is an exposure-reduction step whose output feeds the mechanism.
2. A k-ary randomised response (k-RR) over a **public, committed vocabulary**
   (`fl/public_vocabulary.txt`, rebuilt from this repository's public
   documentation). Each protected token is emitted unchanged with probability
   ``p = e^eps / (e^eps + k - 1)`` and otherwise replaced by a uniformly random
   different vocabulary word. That release is ``eps``-differentially private for
   the token: any two possible input words induce output distributions whose
   ratio is at most ``e^eps``.
3. Words outside the public vocabulary cannot be preserved by the mechanism;
   they are replaced by a fixed ``[redacted]`` symbol. This is more protective
   than a random substitution, but it does reveal the *position and count* of
   rare words. Proper nouns, place names and unusual terms therefore disappear.

What is NOT claimed
-------------------
- The guarantee is **per token**, not per document. Under basic sequential
  composition a prompt with ``n`` protected tokens is at most ``n * eps``-DP, and
  the caller must display that bound rather than imply the whole prompt is
  ``eps``-private. `composed_epsilon()` returns it.
- Function words (see `FUNCTION_WORDS`) are passed through verbatim as a utility
  decision. Their presence, order and the sentence shape are therefore *not*
  protected.
- The provider's **reply** is not a DP release and is returned unmodified.
- Numeric literals that do not match a redaction pattern are passed through, so
  arithmetic requests stay answerable. Quantities are therefore not protected.
- Sampling uses OS-backed randomness through `secrets.SystemRandom`, like the
  Gaussian mechanism in `fl/privacy.py`. It is a research implementation, not an
  audited discrete sampler.
"""

import math
import re
import secrets
from functools import lru_cache
from pathlib import Path

VOCABULARY_PATH = Path(__file__).resolve().parent / "public_vocabulary.txt"
REDACTED = "[redacted]"
TOKEN_EPSILON = 10.0

# Passed through unchanged. Documented as unprotected; see module docstring.
FUNCTION_WORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because
    been before being below between both but by can cannot could did do does
    doing down during each few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just me more
    most my myself no nor not of off on once only or other our ours ourselves
    out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would
    you your yours yourself yourselves please could would may might must shall
    tell give make need want help
    """.split()
)

REDACTION_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "email"),
    (re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I), "url"),
    (
        re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,5}[\s-]?\d{3,5}(?:[\s-]?\d{2,4})?\b"),
        "phone-or-id",
    ),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "address"),
    (
        re.compile(
            r"\b(?:password|passwd|secret|token|otp|pin|cvv|aadhaar|pan|ssn|api[_ -]?key)\b\s*[:=]?\s*\S+",
            re.I,
        ),
        "credential",
    ),
]

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


@lru_cache(maxsize=1)
def vocabulary():
    if not VOCABULARY_PATH.exists():
        raise RuntimeError(
            "fl/public_vocabulary.txt is missing. Run scripts/build_public_vocabulary.py."
        )
    words = tuple(
        line.strip()
        for line in VOCABULARY_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    if len(words) < 100:
        raise ValueError("Public vocabulary is too small to be a usable output space")
    if len(set(words)) != len(words):
        raise ValueError("Public vocabulary must not contain duplicates")
    return words


@lru_cache(maxsize=1)
def _rank():
    return {word: position for position, word in enumerate(vocabulary())}


def retention_probability(epsilon, size=None):
    """Probability that a protected token survives the mechanism unchanged."""
    size = size or len(vocabulary())
    weight = math.exp(epsilon)
    return weight / (weight + size - 1)


def composed_epsilon(epsilon, protected_tokens):
    """Basic-composition bound for a whole prompt. Displayed, never implied away."""
    return epsilon * max(0, protected_tokens)


def redact(text):
    """Deterministic de-identification. Returns (text, [{type, count}])."""
    found = {}
    result = text
    for pattern, label in REDACTION_PATTERNS:
        result, count = pattern.subn(REDACTED, result)
        if count:
            found[label] = found.get(label, 0) + count
    return result, [
        {"type": label, "count": count} for label, count in sorted(found.items())
    ]


def _perturb_word(word, words, epsilon, rng):
    """k-RR release for one token. Returns (released, changed)."""
    ranks = _rank()
    lowered = word.lower()
    position = ranks.get(lowered)
    if position is None:
        # Out of the public output space: cannot be preserved, so it is dropped.
        return REDACTED, True
    size = len(words)
    weight = math.exp(epsilon)
    if rng.random() < weight / (weight + size - 1):
        return word, False
    index = rng.randrange(size - 1)
    return words[index if index < position else index + 1], True


def perturb(text, epsilon_token=TOKEN_EPSILON):
    """De-identify then apply token-level k-RR. Never returns the raw prompt.

    The returned report contains the original words, because it is shown only to
    the requesting account on its own device/session; only `text` is transmitted.
    """
    if not 0 < epsilon_token <= 20:
        raise ValueError("Token epsilon must be in (0, 20]")
    words = vocabulary()
    rng = secrets.SystemRandom()
    deidentified, redactions = redact(text)
    released = []
    changes = []
    protected = 0
    for chunk in deidentified.split():
        match = WORD.fullmatch(chunk.strip(".,;:!?()\"'"))
        if not match:
            released.append(chunk)
            continue
        token = match.group()
        prefix = chunk[: match.start()]
        suffix = chunk[match.end() :]
        if token.lower() in FUNCTION_WORDS:
            released.append(chunk)
            continue
        protected += 1
        replacement, changed = _perturb_word(token, words, epsilon_token, rng)
        if changed:
            changes.append({"from": token, "to": replacement})
        released.append(prefix + replacement + suffix)
    return {
        "text": " ".join(released),
        "redactions": redactions,
        "perturbed": changes,
        "protected_tokens": protected,
        "vocabulary_size": len(words),
        "epsilon_token": epsilon_token,
        "retention_probability": retention_probability(epsilon_token, len(words)),
        "composed_epsilon": composed_epsilon(epsilon_token, protected),
        "mechanism": mechanism_label(epsilon_token, len(words)),
    }


def mechanism_label(epsilon_token=TOKEN_EPSILON, size=None):
    size = size or len(vocabulary())
    return (
        f"Token-level {epsilon_token:g}-LDP k-RR over {size} public words "
        f"+ deterministic PII redaction"
    )
