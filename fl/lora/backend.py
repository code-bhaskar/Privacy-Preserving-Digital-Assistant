"""The trainable object of the `lora` stage: a frozen base plus a low-rank adapter.

Parameterisation
----------------
The shared intent model is a matrix ``W0`` of shape ``(F+1, L)`` (129 x 5 in this
repository). This stage never moves ``W0``. It learns

    W = W0 + A @ B        A: (F+1, r) public projection, fixed and deterministic
                          B: (r, L)   trainable adapter, initialised to zero

which is LoRA (Hu et al., 2021) applied to the classifier head: a frozen base, a
rank-``r`` update, and ``B = 0`` reproducing the base model exactly. ``A`` is
derived from a committed seed by hashing, not from a random-number generator, so
every client process and the server reconstruct the identical matrix without it
ever being transmitted, and the value cannot drift between NumPy versions.

What is released, and why that matters for privacy
--------------------------------------------------
The client releases ``B`` — ``r * L`` numbers — not the ``(F+1) * L`` merged
matrix. The Gaussian mechanism in `fl/privacy.py` adds noise per coordinate, so
the released object's dimension decides how much noise energy reaches the served
model:

    full-matrix stage: (F+1)*L = 665 noisy coordinates, all of them served
    rank-4 adapter   : r*L     =  20 noisy coordinates, merged through A

For the same epsilon, delta and clip norm, the merged perturbation of the
adapter stage carries roughly ``sqrt(r*L / ((F+1)*L))`` of the noise energy —
about 5.8x less at rank 4 — which is the whole utility argument for federating
an adapter instead of a matrix. `tests/test_lora_fl.py` measures that ratio
rather than asserting it from theory alone.

This is a dimension-reduction benefit, not a stronger guarantee: the mechanism,
its epsilon and its ledger charge are identical to the default stage.

What is deliberately not here
-----------------------------
No causal-LLM (llama/qwen) backend. Ollama's API cannot hot-load a PEFT adapter,
so a LoRA round over GGUF weights would publish an artefact that nothing in this
system could serve, and this repository has no in-process LLM runtime to serve it
from. The functions below are the backend interface; an operator with local
encoder weights and an in-process serving path can add one without touching the
client, the protocol, the ledger or the publication gate.
"""

import hashlib
import math

import numpy as np

from app.local_model import LABELS, SHAPE, features, sanity_score

# Committed, public. Changing it invalidates every published adapter, because the
# server and the clients would no longer agree on the projection.
PROJECTION_SEED = "ppda-lora-projection-v1"
FEATURES_PLUS_ONE, LABEL_COUNT = SHAPE


def _unit_uniform(*parts):
    """Deterministic uniform in [0, 1) from a hash. No RNG state, no version drift."""
    digest = hashlib.blake2b(
        "|".join(str(part) for part in parts).encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") / float(1 << 64)


def projection(rank):
    """Public A of shape (F+1, rank), approximately orthonormal columns.

    Two hashed uniforms per entry through Box-Muller give exact standard normals
    from a committed seed. Columns are normalised to unit length so that a
    clipped adapter delta of norm C merges into a matrix perturbation of
    comparable norm, which keeps the clip constant interpretable.
    """
    if not 1 <= int(rank) <= 32:
        raise ValueError("Adapter rank must be between 1 and 32")
    columns = np.empty((FEATURES_PLUS_ONE, int(rank)), dtype=np.float64)
    for j in range(int(rank)):
        for i in range(FEATURES_PLUS_ONE):
            u1 = max(_unit_uniform(PROJECTION_SEED, i, j, 0), 1e-12)
            u2 = _unit_uniform(PROJECTION_SEED, i, j, 1)
            columns[i, j] = math.sqrt(-2.0 * math.log(u1)) * math.cos(
                2.0 * math.pi * u2
            )
        norm = np.linalg.norm(columns[:, j])
        if not norm or not np.isfinite(norm):
            raise ValueError("Degenerate projection column")
        columns[:, j] /= norm
    return columns


def vector_length(rank):
    """Length of the released, noised vector. This is what the server validates."""
    return int(rank) * LABEL_COUNT


def adapter_shape(rank):
    return (int(rank), LABEL_COUNT)


def initialize(rank):
    """B = 0: a fresh adapter is exactly the base model, as in LoRA."""
    return np.zeros(adapter_shape(rank))


def merge(base, adapter, rank):
    """Materialise the served matrix. Pure post-processing; no privacy cost."""
    adapter = np.asarray(adapter, dtype=np.float64).reshape(adapter_shape(rank))
    if not np.isfinite(adapter).all():
        raise ValueError("Nonfinite adapter")
    merged = np.asarray(base, dtype=np.float64).reshape(SHAPE) + projection(rank) @ adapter
    if merged.shape != SHAPE or not np.isfinite(merged).all():
        raise ValueError("Merged model is not servable")
    return merged


def probabilities_with(base, adapter, rank, text):
    """Inference through the merged matrix, via the same ONNX session used in
    production. Kept here so a client can sanity-check its own adapter locally."""
    from app.onnx_model import probabilities as served

    return served(merge(base, adapter, rank), text)


def train(base, adapter, examples, rank, steps=60, lr=0.5):
    """Local training of B only, on the client's own decrypted examples.

    Cross-entropy over the merged logits, gradient taken with respect to B:
    with ``z = X @ A`` of shape ``(N, r)``, ``dL/dB = z.T @ (softmax - Y) / N``.
    ``W0`` is read-only here, which is what makes the released object small.
    """
    if not examples:
        raise ValueError("A client needs at least one example to train on")
    if not 1 <= int(steps) <= 2000 or not 0 < float(lr) <= 4:
        raise ValueError("Unsupported local training parameters")
    A = projection(rank)
    base = np.asarray(base, dtype=np.float64).reshape(SHAPE)
    b = np.array(adapter, dtype=np.float64, copy=True).reshape(adapter_shape(rank))
    x = np.stack([features(text) for text, _ in examples])
    y = np.eye(LABEL_COUNT)[[LABELS.index(label) for _, label in examples]]
    base_logits = x @ base
    z = x @ A
    for _ in range(int(steps)):
        logits = base_logits + z @ b
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        error = (p - y) / len(x)
        b -= float(lr) * (z.T @ error + 0.001 * b)
    if not np.isfinite(b).all():
        raise ValueError("Local training diverged")
    return b


def score(base, adapter, rank):
    """Public-seed regression score of the merged model.

    Explicitly NOT a held-out accuracy benchmark: the seed sentences are in this
    repository, so this is a "did the noisy update break the served model" gate,
    the same rule the default stage applies.
    """
    return sanity_score(merge(base, adapter, rank))


def label_of(base, adapter, rank, text):
    """Label of the merged model, for client-side self-checks and tests."""
    p = probabilities_with(base, adapter, rank, text)
    return LABELS[int(np.asarray(p).argmax())]


def signal_norm(adapter, rank):
    """Frobenius norm of the merged perturbation a clean adapter would cause."""
    return float(np.linalg.norm(projection(rank) @ np.asarray(adapter, dtype=np.float64)))
