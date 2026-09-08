"""Create a persistent VAPID signing key; never print it or replace an existing key."""

import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

root = Path(__file__).resolve().parent.parent
path = root / ".runtime/vapid-private.pem"
path.parent.mkdir(mode=0o700, exist_ok=True)
try:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    print("Existing VAPID key retained.")
else:
    key = ec.generate_private_key(ec.SECP256R1())
    with os.fdopen(fd, "wb") as f:
        f.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    print(
        "Created local VAPID key (0600). Set VAPID_SUBJECT to an operator contact before production use."
    )
