"""All-participant pairwise masking; dropout ABORTS, no recovery secret release.

Honest clients/coordinator, at least two noncolluding clients. PRG security is
computational. Transport here is private OS pipes on one trusted host. This is
not a distributed-device authentication protocol or a Bonawitz implementation.
"""

import numpy as np
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


def pair_mask(private_key, public_hex, round_nonce, length):
    shared = private_key.exchange(
        X25519PublicKey.from_public_bytes(bytes.fromhex(public_hex))
    )
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=bytes.fromhex(round_nonce),
        info=b"ppda-secagg-v1",
    ).derive(shared)
    stream = (
        Cipher(algorithms.ChaCha20(key, b"\0" * 16), mode=None)
        .encryptor()
        .update(b"\0" * (length * 4))
    )
    return np.frombuffer(stream, dtype="<u4")


def mask(vector, private_key, peers, own_index, nonce):
    result = vector.copy().astype(np.uint32)
    for peer_index, public in enumerate(peers):
        if own_index == peer_index:
            continue
        stream = pair_mask(private_key, public, nonce, len(result))
        result = result + stream if own_index < peer_index else result - stream
    return result.astype("<u4")


def aggregate(masked, expected):
    if expected < 3 or len(masked) != expected:
        raise ValueError("All cohort members must contribute; dropout aborts the round")
    if len({len(v) for v in masked}) != 1:
        raise ValueError("Mismatched vector dimensions")
    return np.sum(np.stack(masked), axis=0, dtype=np.uint32)
