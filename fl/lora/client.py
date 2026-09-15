"""A real independent OS worker for the low-rank stage.

Same contract as `fl/client.py`: plaintext examples and the locally trained
adapter never reach stdout, only the masked, quantised, noise-added vector does.
The base matrix arrives read-only, so nothing here can alter the published model.

This module must not import `app.config` or `app.security`: the supervisor starts
it with a minimal environment that carries no secrets, and the per-user key is
passed in over the private pipe instead.
"""

import base64
import json
import sys

import numpy as np
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.local_model import SHAPE
from .backend import adapter_shape, initialize, merge, train, vector_length
from ..privacy import privatize, quantize
from ..protocol import mask


def decrypt_examples(config):
    """Recover the client's own plaintext locally. Identical envelope to the
    default stage: AES-256-GCM under the per-user key, AAD binds the row to this
    account and purpose, so a ciphertext cannot be replayed across users."""
    key = base64.b64decode(config["key"])
    aad = f"{config['uid']}:training:v1".encode()
    out = []
    for blob in config["examples"]:
        if not isinstance(blob, str) or not blob.startswith("v1."):
            raise ValueError("Unsupported ciphertext version")
        encrypted = base64.b64decode(blob[3:], validate=True)
        value = json.loads(AESGCM(key).decrypt(encrypted[:12], encrypted[12:], aad))
        text, label = value["text"], value["label"]
        if not isinstance(text, str) or not text.strip() or not isinstance(label, str):
            raise ValueError("Malformed training example")
        out.append((text, label))
    return out


def main():
    config = json.loads(sys.stdin.readline())
    rank = int(config["rank"])
    if vector_length(rank) != rank * adapter_shape(rank)[1]:
        raise ValueError("Inconsistent adapter shape")

    secret = X25519PrivateKey.generate()
    print(
        json.dumps({"public_key": secret.public_key().public_bytes_raw().hex()}),
        flush=True,
    )
    peers = json.loads(sys.stdin.readline())["peers"]
    if len(peers) < 3 or len(peers) != len(set(peers)):
        raise ValueError("Unsafe cohort")

    examples = decrypt_examples(config)
    base = np.array(config["base"], dtype=np.float64).reshape(SHAPE)
    start = initialize(rank)
    trained = train(
        base,
        start,
        examples,
        rank,
        steps=int(config.get("steps", 60)),
        lr=float(config.get("lr", 0.5)),
    )
    # The adapter starts at zero, so the delta is the adapter; it is still
    # computed as a difference so a warm-started adapter would remain correct.
    delta = trained - start
    protected = privatize(
        delta,
        float(config["epsilon"]),
        float(config["delta"]),
        float(config.get("clip_norm", 0.1)),
    )
    quantized = quantize(protected, len(peers))
    masked = mask(quantized, secret, peers, int(config["index"]), config["nonce"])
    print(json.dumps({"masked": masked.tobytes().hex()}), flush=True)


if __name__ == "__main__":
    main()
