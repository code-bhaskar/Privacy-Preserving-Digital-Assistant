import base64
import hashlib
import hmac
import json
import os

import bcrypt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from .config import MASTER_KEY, settings


def user_key(uid: int) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=f"ppda-user-v1:{uid}".encode(),
    ).derive(MASTER_KEY)


def encrypt(uid: int, purpose: str, value: dict) -> str:
    nonce = os.urandom(12)
    body = json.dumps(value, ensure_ascii=False).encode()
    ciphertext = AESGCM(user_key(uid)).encrypt(
        nonce, body, f"{uid}:{purpose}:v1".encode()
    )
    return "v1." + base64.b64encode(nonce + ciphertext).decode()


def decrypt(uid: int, purpose: str, value: str) -> dict:
    if not value.startswith("v1."):
        raise ValueError("Unsupported ciphertext version")
    data = base64.b64decode(value[3:], validate=True)
    return json.loads(
        AESGCM(user_key(uid)).decrypt(
            data[:12], data[12:], f"{uid}:{purpose}:v1".encode()
        )
    )


def valid_password(password: str):
    if not 12 <= len(password) or len(password.encode("utf-8")) > 72:
        raise ValueError("Use at least 12 characters and at most 72 UTF-8 bytes.")


def password_hash(password: str) -> str:
    valid_password(password)
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, encoded: str) -> bool:
    raw = password.encode("utf-8")
    # Long inputs are rejected without silent bcrypt truncation.
    if len(raw) > 72:
        raw = b"invalid-overlong-password"
        bcrypt.checkpw(raw, encoded.encode())
        return False
    return bcrypt.checkpw(raw, encoded.encode())


def keyed(value: str) -> str:
    return hmac.new(
        settings.jwt_secret.encode(), value.encode(), hashlib.sha256
    ).hexdigest()


DUMMY_HASH = bcrypt.hashpw(
    os.urandom(32).hex().encode(), bcrypt.gensalt(rounds=12)
).decode()
