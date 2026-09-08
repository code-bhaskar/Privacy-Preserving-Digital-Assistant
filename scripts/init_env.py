"""Create local secrets once. Never print or overwrite existing keys."""

import base64
import os
from pathlib import Path

root = Path(__file__).resolve().parent.parent
(root / ".runtime").mkdir(mode=0o700, exist_ok=True)
try:
    fd = os.open(root / ".env", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    print("Existing .env retained; no keys changed.")
else:
    with os.fdopen(fd, "w") as stream:
        stream.write("JWT_SECRET=" + os.urandom(48).hex() + "\n")
        stream.write(
            "AES_MASTER_KEY=" + base64.b64encode(os.urandom(32)).decode() + "\n"
        )
        stream.write("COOKIE_SECURE=true\nPIPELINE_ENABLED=true\n")
    print("Local secrets created in ignored .env (mode 0600).")
