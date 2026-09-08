import json
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from sqlalchemy import select
from app.database import engine, transaction, users, examples, ledger, rounds
from app.preferences import expenditure, affordable
from fl import pipeline
from fl.privacy import (
    clip,
    sigma,
    privatize,
    quantize,
    decode_sum,
)
from fl.protocol import mask, aggregate

P = "/api/v1"


def test_clip_and_gaussian_calibration():
    x = np.array([3.0, 4.0])
    assert np.linalg.norm(clip(x, norm=2)) == pytest.approx(2)
    assert sigma(0.5, 1e-6, 0.2) > 2.1
    for epsilon in [0, -1, 1.1]:
        with pytest.raises(ValueError):
            sigma(epsilon, 1e-6, 0.2)
    a = privatize(np.zeros(300))
    b = privatize(np.zeros(300))
    assert not np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_real_pairwise_mask_cancellation_and_dropout_abort():
    private = [X25519PrivateKey.generate() for _ in range(3)]
    peers = [p.public_key().public_bytes_raw().hex() for p in private]
    nonce = "12" * 32
    vectors = [
        np.array([0.1, -0.2, 0.3]),
        np.array([0.4, 0.1, -0.5]),
        np.array([0.2, 0.3, 0.1]),
    ]
    masked = [
        mask(quantize(v, 3), private[i], peers, i, nonce) for i, v in enumerate(vectors)
    ]
    for v, y in zip(vectors, masked):
        assert not np.array_equal(quantize(v, 3), y)
    mean = decode_sum(aggregate(masked, 3), 3)
    np.testing.assert_allclose(mean, np.mean(vectors, axis=0), atol=1e-5)
    with pytest.raises(ValueError):
        aggregate(masked[:2], 3)


def test_examples_consent_and_encryption(client, signup):
    user = signup()
    assert (
        client.post(
            P + "/learning/examples",
            json={"text": "remind me secret test", "label": "reminder"},
        ).status_code
        == 403
    )
    p = user["preferences"]
    p["training"] = True
    client.put(P + "/settings", json=p)
    assert (
        client.post(
            P + "/learning/examples",
            json={"text": "remind me secret test", "label": "reminder"},
        ).status_code
        == 201
    )
    with engine.connect() as conn:
        row = (
            conn.execute(select(examples).where(examples.c.user_id == user["id"]))
            .mappings()
            .one()
        )
    assert "secret test" not in row["ciphertext"]
    p["training"] = False
    client.put(P + "/settings", json=p)
    assert client.get(P + "/learning/status").json()["queued"] == 0


def test_insufficient_participants_never_fabricated(client, signup):
    user = signup()
    p = user["preferences"]
    p["training"] = True
    client.put(P + "/settings", json=p)
    pipeline.stop.clear()
    pipeline.run_once()
    assert client.get(P + "/learning/status").json()["history"] == []


def test_end_to_end_three_real_worker_processes_and_budget(client, signup):
    ids = []
    for _ in range(3):
        user = signup()
        ids.append(user["id"])
        p = user["preferences"]
        p["training"] = True
        p["epsilon"] = 1
        assert client.put(P + "/settings", json=p).status_code == 200
        for text in [
            "remind me to call mom",
            "remind me at five",
            "create a reminder tomorrow",
        ]:
            assert (
                client.post(
                    P + "/learning/examples", json={"text": text, "label": "reminder"}
                ).status_code
                == 201
            )
    pipeline.stop.clear()
    pipeline.run_once()
    status = client.get(P + "/learning/status").json()
    assert status["epsilon_spent"] == 0.5
    assert status["delta_spent"] == 1e-6
    assert status["queued"] == 0
    assert status["history"][0]["status"] in ("published", "rejected"), status
    with engine.connect() as conn:
        for uid in ids:
            assert expenditure(conn, uid) == (0.5, 1e-6)
        row = (
            conn.execute(
                select(rounds).where(rounds.c.id == status["history"][0]["id"])
            )
            .mappings()
            .one()
        )
        assert json.loads(row["participants"]) == ids
    # Persistent ledger is append-only; model rejection does not refund privacy.
    with pytest.raises(Exception):
        with transaction() as conn:
            conn.execute(ledger.delete())
    pipeline.initialize_model()
    assert client.get(P + "/learning/status").json()["epsilon_spent"] == 0.5
    # A target below the next round's requirement blocks training.
    with transaction() as conn:
        uid = ids[-1]
        rid = conn.execute(
            rounds.insert().values(
                status="aborted",
                participants=json.dumps([uid]),
                created_at=0,
                detail="test reservation",
            )
        ).inserted_primary_key[0]
        conn.execute(
            ledger.insert().values(user_id=uid, round_id=rid, epsilon=0.5, delta=1e-6)
        )
        user = conn.execute(select(users).where(users.c.id == uid)).mappings().one()
        assert not affordable(conn, user)
    assert client.get(P + "/learning/status").json()["state"] == "Budget exhausted"


def test_revoked_training_not_admitted(client, signup):
    user = signup()
    p = user["preferences"]
    p["training"] = True
    client.put(P + "/settings", json=p)
    for _ in range(3):
        client.post(
            P + "/learning/examples", json={"text": "take a note", "label": "note"}
        )
    p["training"] = False
    client.put(P + "/settings", json=p)
    pipeline.run_once()
    assert client.get(P + "/learning/status").json()["epsilon_spent"] == 0


def test_dropout_retains_budget_and_never_releases_model(client, signup, monkeypatch):
    for _ in range(3):
        user = signup()
        p = user["preferences"]
        p["training"] = True
        client.put(P + "/settings", json=p)
        for _ in range(3):
            client.post(
                P + "/learning/examples",
                json={"text": "write a new note", "label": "note"},
            )
    before = client.get(P + "/learning/status").json()["model_version"]

    def disconnect(processes, timeout=45):
        processes[-1].kill()
        raise RuntimeError("Test dropout")

    monkeypatch.setattr(pipeline, "read_messages", disconnect)
    pipeline.stop.clear()
    pipeline.run_once()
    status = client.get(P + "/learning/status").json()
    assert status["history"][0]["status"] == "aborted"
    assert status["epsilon_spent"] == 0.5
    assert status["model_version"] == before
    assert status["queued"] == 0


def test_local_category_revocation_purges_queued_examples(client, signup):
    user = signup()
    p = user["preferences"]
    p["training"] = True
    client.put(P + "/settings", json=p)
    client.post(
        P + "/learning/examples",
        json={"text": "create calendar event", "label": "calendar"},
    )
    assert client.get(P + "/learning/status").json()["queued"] == 1
    p["calendar"] = False
    client.put(P + "/settings", json=p)
    assert client.get(P + "/learning/status").json()["queued"] == 0
