"""Real Web Push. Minimal lock-screen payloads; durable, best-effort timed delivery."""

import base64
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from pywebpush import WebPushException, webpush
from sqlalchemy import delete, select, update

from .config import settings
from .database import (
    entries,
    push_outbox,
    push_subscriptions,
    record,
    transaction,
    users,
)
from .preferences import prefs
from .security import decrypt


class NoRedirectSession(requests.Session):
    def request(self, *args, **kwargs):
        kwargs["allow_redirects"] = False
        return super().request(*args, **kwargs)


def public_key():
    path = Path(settings.vapid_private_key_path)
    if not path.is_file():
        return None
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(
        key.curve, ec.SECP256R1
    ):
        raise ValueError("VAPID key must be P-256")
    return (
        base64.urlsafe_b64encode(
            key.public_key().public_bytes(
                serialization.Encoding.X962,
                serialization.PublicFormat.UncompressedPoint,
            )
        )
        .rstrip(b"=")
        .decode()
    )


def validate_subscription(value):
    # Subscription URLs are server request destinations: no arbitrary URLs/SSRF.
    endpoint = value["endpoint"]
    parsed = urlparse(endpoint)
    host = parsed.hostname or ""
    allowed = (
        host
        in (
            "fcm.googleapis.com",
            "web.push.apple.com",
            "wns2-par02p.notify.windows.com",
        )
        or host.endswith(".push.services.mozilla.com")
        or host.endswith(".notify.windows.com")
    )
    if (
        parsed.scheme != "https"
        or not allowed
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or parsed.fragment
    ):
        raise ValueError("Unsupported browser push provider endpoint")
    for name, length in [("p256dh", 65), ("auth", 16)]:
        raw = value["keys"][name]
        if not re.fullmatch(r"[A-Za-z0-9_-]+={0,2}", raw):
            raise ValueError("Invalid subscription key")
        decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
        if len(decoded) != length:
            raise ValueError("Invalid subscription key size")
        if name == "p256dh":
            ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), decoded)


def cancel_entry(conn, entry_id):
    conn.execute(
        update(push_outbox)
        .where(
            push_outbox.c.entry_id == entry_id,
            push_outbox.c.status.in_(["pending", "sending"]),
        )
        .values(status="cancelled")
    )


def enqueue_due(now=None):
    now = now or datetime.now(timezone.utc)
    with transaction() as conn:
        rows = (
            conn.execute(
                select(entries).where(
                    entries.c.kind.in_(["Calendar", "Reminders"]),
                    entries.c.done == False,
                    entries.c.fired == False,
                )
            )
            .mappings()
            .all()
        )
        for row in rows:
            if not row["due_at"] or datetime.fromisoformat(row["due_at"]) > now:
                continue
            user = (
                conn.execute(select(users).where(users.c.id == row["user_id"]))
                .mappings()
                .one()
            )
            if not prefs(user)["calendar"]:
                continue
            conn.execute(
                update(entries).where(entries.c.id == row["id"]).values(fired=True)
            )
            record(
                conn,
                user["id"],
                "REMINDER_FIRED" if row["kind"] == "Reminders" else "EVENT_STARTED",
            )
            # Do not send arbitrarily old reminders after a long server outage.
            expiry = datetime.fromisoformat(row["due_at"]).timestamp() + 3600
            if expiry <= now.timestamp():
                continue
            subs = (
                conn.execute(
                    select(push_subscriptions).where(
                        push_subscriptions.c.user_id == user["id"]
                    )
                )
                .mappings()
                .all()
            )
            for sub in subs:
                existing = conn.execute(
                    select(push_outbox.c.id).where(
                        push_outbox.c.entry_id == row["id"],
                        push_outbox.c.subscription_id == sub["id"],
                        push_outbox.c.due_at == row["due_at"],
                    )
                ).first()
                if not existing:
                    conn.execute(
                        push_outbox.insert().values(
                            entry_id=row["id"],
                            subscription_id=sub["id"],
                            due_at=row["due_at"],
                            status="pending",
                            attempts=0,
                            next_attempt=now.timestamp(),
                            expires_at=expiry,
                        )
                    )


def deliver_batch(now=None, limit=20):
    timestamp = time.time() if now is None else now
    with transaction() as conn:
        # Recover an interrupted send after its lease; tag-based notifications deduplicate retries.
        conn.execute(
            update(push_outbox)
            .where(
                push_outbox.c.status == "sending",
                push_outbox.c.next_attempt <= timestamp,
            )
            .values(status="pending")
        )
        jobs = (
            conn.execute(
                select(push_outbox)
                .where(
                    push_outbox.c.status == "pending",
                    push_outbox.c.next_attempt <= timestamp,
                )
                .order_by(push_outbox.c.id)
                .limit(limit)
            )
            .mappings()
            .all()
        )
    for job in jobs:
        timestamp = time.time() if now is None else now
        with transaction() as conn:
            current = (
                conn.execute(
                    select(push_outbox).where(
                        push_outbox.c.id == job["id"], push_outbox.c.status == "pending"
                    )
                )
                .mappings()
                .first()
            )
            if not current:
                continue
            row = (
                conn.execute(select(entries).where(entries.c.id == job["entry_id"]))
                .mappings()
                .first()
            )
            sub = (
                conn.execute(
                    select(push_subscriptions).where(
                        push_subscriptions.c.id == job["subscription_id"]
                    )
                )
                .mappings()
                .first()
            )
            user = (
                conn.execute(select(users).where(users.c.id == row["user_id"]))
                .mappings()
                .first()
                if row
                else None
            )
            valid = (
                row
                and sub
                and user
                and sub["user_id"] == row["user_id"]
                and prefs(user)["calendar"]
                and not row["done"]
                and row["due_at"] == job["due_at"]
                and job["expires_at"] > timestamp
            )
            if not valid:
                conn.execute(
                    update(push_outbox)
                    .where(push_outbox.c.id == job["id"])
                    .values(status="cancelled")
                )
                continue
            count = current["attempts"] + 1
            conn.execute(
                update(push_outbox)
                .where(push_outbox.c.id == job["id"])
                .values(status="sending", attempts=count, next_attempt=timestamp + 60)
            )
        # No DB transaction held during external network I/O. Revocation cannot recall an in-flight push.
        status = "sent"
        try:
            subscription = decrypt(user["id"], "push-subscription", sub["ciphertext"])
            validate_subscription(subscription)
            session = NoRedirectSession()
            try:
                webpush(
                    subscription_info=subscription,
                    data=json.dumps(
                        {
                            "title": "PPDA · Reminder due"
                            if row["kind"] == "Reminders"
                            else "PPDA · Event starting",
                            "body": "Open your workspace to view the details.",
                            "tag": f"ppda-{job['id']}",
                            "workspace": row["kind"],
                        }
                    ),
                    vapid_private_key=settings.vapid_private_key_path,
                    vapid_claims={"sub": settings.vapid_subject},
                    ttl=max(1, int(job["expires_at"] - timestamp)),
                    headers={"Urgency": "high"},
                    timeout=10,
                    requests_session=session,
                )
            finally:
                session.close()
        except WebPushException as exc:
            code = exc.response.status_code if exc.response is not None else None
            if code in (404, 410):
                with transaction() as conn:
                    conn.execute(
                        delete(push_subscriptions).where(
                            push_subscriptions.c.id == sub["id"]
                        )
                    )
                    record(conn, user["id"], "PUSH_SUBSCRIPTION_EXPIRED")
                continue
            status = "failed" if count >= 5 else "pending"
        except Exception:
            status = "failed" if count >= 5 else "pending"
        with transaction() as conn:
            # A concurrent edit/disable may have cancelled or removed this job.
            conn.execute(
                update(push_outbox)
                .where(push_outbox.c.id == job["id"], push_outbox.c.status == "sending")
                .values(
                    status=status,
                    next_attempt=timestamp + min(900, 15 * 4 ** (count - 1)),
                )
            )
            record(
                conn,
                user["id"],
                "PUSH_ACCEPTED_BY_PROVIDER"
                if status == "sent"
                else "PUSH_RETRY_OR_FAILURE",
            )
