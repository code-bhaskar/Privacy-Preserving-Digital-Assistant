"""A real independent OS worker. Plaintext training/deltas never go to stdout."""

import base64
import json
import sys
import numpy as np
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from app.local_model import train, SHAPE
from .privacy import privatize, quantize
from .protocol import mask


def main():
    config = json.loads(sys.stdin.readline())
    secret = X25519PrivateKey.generate()
    print(
        json.dumps({"public_key": secret.public_key().public_bytes_raw().hex()}),
        flush=True,
    )
    peers = json.loads(sys.stdin.readline())["peers"]
    if len(peers) < 3 or len(peers) != len(set(peers)):
        raise ValueError("Unsafe cohort")
    key = base64.b64decode(config["key"])
    local_data = []
    for blob in config["examples"]:
        encrypted = base64.b64decode(blob[3:])
        value = json.loads(
            AESGCM(key).decrypt(
                encrypted[:12], encrypted[12:], f"{config['uid']}:training:v1".encode()
            )
        )
        local_data.append((value["text"], value["label"]))
    weights = np.array(config["weights"]).reshape(SHAPE)
    delta = train(weights, local_data) - weights
    protected = privatize(delta, config["epsilon"], config["delta"])
    quantized = quantize(protected, len(peers))
    masked = mask(quantized, secret, peers, config["index"], config["nonce"])
    print(json.dumps({"masked": masked.tobytes().hex()}), flush=True)


if __name__ == "__main__":
    main()
