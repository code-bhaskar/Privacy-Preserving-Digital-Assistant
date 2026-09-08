import base64
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from pywebpush import WebPushException
from sqlalchemy import delete, select, update

from app import notifications
from app.config import settings
from app.database import engine, push_outbox, push_subscriptions, transaction
from app.security import decrypt

P = "/api/v1"
DUE = datetime(2099, 6, 1, 12, tzinfo=timezone.utc)


def b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def subscription():
    public = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        )
    )
    return {
        "endpoint": "https://fcm.googleapis.com/fcm/send/" + uuid.uuid4().hex,
        "keys": {"p256dh": b64(public), "auth": b64(b"t" * 16)},
    }


@pytest.fixture(autouse=True)
def isolated_push(monkeypatch, tmp_path):
    key = ec.generate_private_key(ec.SECP256R1())
    path = tmp_path / "vapid.pem"
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    monkeypatch.setattr(settings, "vapid_private_key_path", str(path))
    calls = []
    monkeypatch.setattr(notifications, "webpush", lambda **kwargs: calls.append(kwargs))
    with transaction() as conn:
        conn.execute(delete(push_outbox))
        conn.execute(delete(push_subscriptions))
    yield calls
    with transaction() as conn:
        conn.execute(delete(push_outbox))
        conn.execute(delete(push_subscriptions))


def scheduled(client, kind="Reminders", done=False):
    result = client.post(
        P + "/entries/" + kind,
        json={
            "title": "PRIVATE medical appointment",
            "detail": DUE.isoformat(),
            "done": done,
        },
    )
    assert result.status_code == 201, result.text
    return result.json()


def subscribe(client):
    value = subscription()
    r = client.post(P + "/notifications/subscribe", json=value)
    assert r.status_code == 201, r.text
    return value


def jobs():
    with engine.connect() as conn:
        return conn.execute(select(push_outbox)).mappings().all()


def test_encrypted_opt_in_auth_keys_owner_and_ssrf(client, signup):
    assert client.get(P + "/notifications/status").status_code == 401
    user = signup(consent=False)
    assert (
        client.post(P + "/notifications/subscribe", json=subscription()).status_code
        == 403
    )
    user = signup()
    sub = subscribe(client)
    assert (
        len(
            base64.urlsafe_b64decode(
                client.get(P + "/notifications/public-key").json()["public_key"] + "="
            )
        )
        == 65
    )
    with engine.connect() as conn:
        row = conn.execute(select(push_subscriptions)).mappings().one()
    assert (
        sub["endpoint"] not in row["ciphertext"]
        and sub["keys"]["auth"] not in row["ciphertext"]
    )
    assert decrypt(user["id"], "push-subscription", row["ciphertext"]) == sub
    for endpoint in [
        "http://fcm.googleapis.com/fcm/send/x",
        "https://127.0.0.1/x",
        "https://fcm.googleapis.com.evil.example/x",
        "https://evil@fcm.googleapis.com/x",
        "https://fcm.googleapis.com:8000/x",
    ]:
        assert (
            client.post(
                P + "/notifications/subscribe", json={**sub, "endpoint": endpoint}
            ).status_code
            == 422
        )
    assert (
        client.post(
            P + "/notifications/subscribe",
            json={**sub, "keys": {"auth": "bad", "p256dh": "bad"}},
        ).status_code
        == 422
    )
    signup()
    assert client.post(P + "/notifications/subscribe", json=sub).status_code == 409
    assert not client.post(
        P + "/notifications/subscription-status", json={"endpoint": sub["endpoint"]}
    ).json()["enabled"]
    client.post(P + "/notifications/unsubscribe", json={"endpoint": sub["endpoint"]})
    with engine.connect() as conn:
        assert conn.execute(select(push_subscriptions.c.id)).first()


def test_due_time_durable_enqueue_generic_payload_dedup(client, signup, isolated_push):
    signup()
    subscribe(client)
    scheduled(client)
    scheduled(client, "Calendar")
    scheduled(client, done=True)
    notifications.enqueue_due(DUE - timedelta(seconds=1))
    assert jobs() == []
    notifications.enqueue_due(DUE)
    notifications.enqueue_due(DUE)
    assert len(jobs()) == 2 and all(job["status"] == "pending" for job in jobs())
    # A separate worker invocation consumes persisted jobs; no in-memory timer required.
    notifications.deliver_batch(DUE.timestamp())
    assert len(isolated_push) == 2
    for call in isolated_push:
        assert "PRIVATE" not in call["data"] and "medical" not in call["data"]
        assert call["timeout"] == 10 and call["headers"]["Urgency"] == "high"
        assert json.loads(call["data"])["workspace"] in ("Calendar", "Reminders")
    notifications.deliver_batch(DUE.timestamp() + 1)
    assert len(isolated_push) == 2 and all(job["status"] == "sent" for job in jobs())


@pytest.mark.parametrize(
    "action", ["reschedule", "complete", "delete", "unsubscribe", "revoke"]
)
def test_cancel_unsent_push(client, signup, isolated_push, action):
    user = signup()
    sub = subscribe(client)
    item = scheduled(client)
    notifications.enqueue_due(DUE)
    url = P + f"/entries/Reminders/{item['id']}"
    if action == "delete":
        assert client.delete(url).status_code == 200
    elif action == "unsubscribe":
        client.post(
            P + "/notifications/unsubscribe", json={"endpoint": sub["endpoint"]}
        )
    elif action == "revoke":
        preferences = user["preferences"]
        preferences["calendar"] = False
        assert client.put(P + "/settings", json=preferences).status_code == 200
    else:
        assert (
            client.put(
                url,
                json={
                    "title": item["title"],
                    "detail": (DUE + timedelta(days=1)).isoformat()
                    if action == "reschedule"
                    else item["detail"],
                    "done": action == "complete",
                },
            ).status_code
            == 200
        )
    notifications.deliver_batch(DUE.timestamp() + 1)
    assert isolated_push == []
    assert all(job["status"] == "cancelled" for job in jobs())
    if action == "reschedule":
        notifications.enqueue_due(DUE + timedelta(days=1))
        notifications.deliver_batch((DUE + timedelta(days=1)).timestamp())
        assert len(isolated_push) == 1


def test_retry_expired_subscription_and_abandoned_send(client, signup, monkeypatch):
    signup()
    subscribe(client)
    scheduled(client)
    notifications.enqueue_due(DUE)
    calls = []

    def failing(**kwargs):
        calls.append(kwargs)
        raise WebPushException("Transient provider failure")

    monkeypatch.setattr(notifications, "webpush", failing)
    notifications.deliver_batch(DUE.timestamp())
    assert jobs()[0]["attempts"] == 1 and jobs()[0]["status"] == "pending"
    notifications.deliver_batch(DUE.timestamp() + 1)
    assert len(calls) == 1
    with transaction() as conn:
        conn.execute(
            update(push_outbox).values(
                status="sending", next_attempt=DUE.timestamp() + 15
            )
        )
    notifications.deliver_batch(DUE.timestamp() + 15)
    assert len(calls) == 2
    response = requests.Response()
    response.status_code = 410

    def gone(**kwargs):
        raise WebPushException("Gone", response=response)

    monkeypatch.setattr(notifications, "webpush", gone)
    notifications.deliver_batch(DUE.timestamp() + 100)
    assert jobs() == []
    with engine.connect() as conn:
        assert not conn.execute(select(push_subscriptions.c.id)).first()


def test_expiry_prevents_obsolete_delivery_and_redirects_disabled(
    client, signup, isolated_push, monkeypatch
):
    signup()
    subscribe(client)
    scheduled(client)
    notifications.enqueue_due(DUE)
    notifications.deliver_batch(DUE.timestamp() + 3601)
    assert isolated_push == [] and jobs()[0]["status"] == "cancelled"
    seen = {}

    def request(self, *args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(requests.Session, "request", request)
    notifications.NoRedirectSession().post(
        "https://fcm.googleapis.com/fcm/send/x", allow_redirects=True
    )
    assert seen["allow_redirects"] is False
