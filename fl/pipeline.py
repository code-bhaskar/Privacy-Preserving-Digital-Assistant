"""Single product pipeline. Real opted-in user queues only, no fabricated clients.

This release demonstrates client OS processes on one trusted host, NOT physical
user-device isolation. The supervisor sees encrypted examples and public keys,
then only masked updates. The host operator can access process memory/keys.

Two stages share this machinery, selected by `LEARNING_STAGE`:

  * ``softmax`` (default, `run_once` below) federates the full shared intent
    matrix. Every client releases a delta over all ``(F+1) * L`` weights.
  * ``lora`` (`fl/lora/pipeline.py`) freezes that matrix and federates a rank-r
    adapter, so a client releases ``r * L`` numbers instead.

Both stages use the same cohort rules, the same consent re-checks, the same
lifetime privacy ledger, the same client-local Gaussian mechanism, the same
pairwise masking, the same public-seed publication gate, and both publish into
`model_versions` — so serving and the browser cannot tell them apart. The shared
parts live here (`begin_round`, `exchange`, `abort_round`, `finish`, `purge`) and
are exercised by `tests/test_learning.py` and `tests/test_lora_fl.py`.
"""

import base64
import json
import os
import selectors
import subprocess
import sys
import threading
import time
import numpy as np
from sqlalchemy import select, update, delete
from app.config import ROOT, settings
from app.database import (
    transaction,
    users,
    examples,
    rounds,
    ledger,
    models,
    record,
)
from app.preferences import prefs, affordable
from app.security import user_key
from app.local_model import SHAPE, seed_weights, sanity_score
from .privacy import EPSILON_PER_ROUND, DELTA_PER_ROUND, decode_sum
from .protocol import aggregate

stop = threading.Event()
trigger = threading.Event()
_thread = None

MINIMUM_COHORT = 3
EXAMPLES_PER_CLIENT = 20
MINIMUM_EXAMPLES = 3
GATE_FLOOR = 0.75
GATE_REGRESSION = 0.025


def initialize_model():
    with transaction() as conn:
        if conn.execute(select(models.c.id)).first() is None:
            weights = seed_weights()
            conn.execute(
                models.insert().values(
                    weights=json.dumps(weights.tolist()),
                    created_at=time.time(),
                    score=sanity_score(weights),
                    active=True,
                )
            )
        # A single backend process is supported. Reservations remain spent after interruption.
        conn.execute(
            update(rounds)
            .where(rounds.c.status == "training")
            .values(
                status="aborted",
                detail="Interrupted by restart; reserved budget retained.",
            )
        )
        conn.execute(delete(examples).where(examples.c.used == True))


def active_model(conn):
    row = (
        conn.execute(
            select(models).where(models.c.active == True).order_by(models.c.id.desc())
        )
        .mappings()
        .first()
    )
    return row["id"], np.array(json.loads(row["weights"])).reshape(SHAPE)


def read_messages(processes, timeout=45):
    # Bounded wait; no blocking readline on a client that never completes.
    outputs = [None] * len(processes)
    buffers = [bytearray() for _ in processes]
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as selector:
        for i, process in enumerate(processes):
            selector.register(process.stdout, selectors.EVENT_READ, i)
        while selector.get_map():
            if stop.is_set() or time.monotonic() > deadline:
                raise RuntimeError("Client timeout/interrupted")
            for key, _ in selector.select(0.2):
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    raise RuntimeError("Client disconnected")
                buffer = buffers[key.data]
                buffer.extend(chunk)
                if len(buffer) > 100_000:
                    raise RuntimeError("Oversized client message")
                if b"\n" in buffer:
                    line, trailing = bytes(buffer).split(b"\n", 1)
                    if trailing:
                        raise RuntimeError("Unexpected out-of-phase message")
                    outputs[key.data] = json.loads(line)
                    selector.unregister(key.fileobj)
    return outputs


def begin_round(conn, stage):
    """Admit a real cohort or return None. Shared by both stages.

    Admission is deliberately boring: training consent, an affordable lifetime
    budget, and at least three queued examples per client. Fewer than three
    eligible clients means no round at all — a cohort is never padded, simulated
    or reused, because the masking protocol and the noise calibration both
    assume three independent contributors.

    The budget is reserved *before* any client starts and is never refunded: an
    interrupted round may already have released noised state, so retrying for
    free would under-count the account's spend.
    """
    if conn.execute(
        select(rounds.c.id).where(rounds.c.status == "training")
    ).first():
        return None
    candidates = []
    for user in conn.execute(select(users)).mappings():
        if not prefs(user)["training"] or not affordable(conn, user):
            continue
        data = (
            conn.execute(
                select(examples)
                .where(examples.c.user_id == user["id"], examples.c.used == False)
                .limit(EXAMPLES_PER_CLIENT)
            )
            .mappings()
            .all()
        )
        if len(data) >= MINIMUM_EXAMPLES:
            candidates.append((user, data))
    if len(candidates) < MINIMUM_COHORT:
        return None
    candidates = candidates[:MINIMUM_COHORT]
    version, weights = active_model(conn)
    ids = [u["id"] for u, _ in candidates]
    round_id = conn.execute(
        rounds.insert().values(
            status="training",
            stage=stage,
            participants=json.dumps(ids),
            created_at=time.time(),
            detail="Real client workers training; budget reserved.",
        )
    ).inserted_primary_key[0]
    for user, data in candidates:
        conn.execute(
            ledger.insert().values(
                user_id=user["id"],
                round_id=round_id,
                epsilon=EPSILON_PER_ROUND,
                delta=DELTA_PER_ROUND,
            )
        )
        conn.execute(
            update(examples)
            .where(examples.c.id.in_([r["id"] for r in data]))
            .values(used=True)
        )
        record(conn, user["id"], "FL_BUDGET_RESERVED")
    consent_versions = {
        u["id"]: json.loads(u["preferences"]).get("_consent_version", 0)
        for u, _ in candidates
    }
    return {
        "round_id": round_id,
        "candidates": candidates,
        "ids": ids,
        "consent_versions": consent_versions,
        "model_version": version,
        "weights": weights,
    }


def spawn(module, config):
    """Start one client worker with a minimal environment.

    No secrets, no `.env`, no provider keys: the worker receives its per-user
    key and its encrypted rows over a private pipe and derives nothing else.
    """
    process = subprocess.Popen(
        [sys.executable, "-m", module],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        start_new_session=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONUNBUFFERED": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    process.stdin.write(json.dumps(config) + "\n")
    process.stdin.flush()
    return process


def exchange(processes, expected_bytes):
    """Two-phase masked exchange: advertise public keys, then collect updates.

    Every peer key must be 32 bytes and distinct, or the round aborts: a
    duplicated key would let two contributions cancel a mask, and a short key is
    not a valid X25519 point.
    """
    advertisements = read_messages(processes)
    peers = [message["public_key"] for message in advertisements]
    if len(set(peers)) != len(peers) or any(
        len(bytes.fromhex(key)) != 32 for key in peers
    ):
        raise RuntimeError("Invalid peer keys")
    for process in processes:
        process.stdin.write(json.dumps({"peers": peers}) + "\n")
        process.stdin.flush()
    messages = read_messages(processes)
    masked = []
    for message in messages:
        raw = bytes.fromhex(message["masked"])
        if len(raw) != expected_bytes:
            raise RuntimeError("Invalid update dimensions")
        masked.append(np.frombuffer(raw, dtype="<u4"))
    return masked, peers


def recheck_consent(conn, ids, consent_versions):
    """Consent is verified again immediately before the aggregate is computed.

    Revoking training consent mid-round aborts it; the reserved budget stays
    spent, because the clients already performed their noised local release.
    """
    for uid in ids:
        user = (
            conn.execute(select(users).where(users.c.id == uid)).mappings().one()
        )
        if (
            not prefs(user)["training"]
            or json.loads(user["preferences"]).get("_consent_version", 0)
            != consent_versions[uid]
        ):
            raise RuntimeError("Consent revoked")


def accept(score, baseline):
    """Public-seed regression gate. NOT a held-out accuracy benchmark."""
    return score >= max(GATE_FLOOR, baseline - GATE_REGRESSION)


def publish(conn, weights, score, round_id):
    conn.execute(update(models).values(active=False))
    return conn.execute(
        models.insert().values(
            weights=json.dumps(np.asarray(weights).tolist()),
            created_at=time.time(),
            score=score,
            active=True,
            round_id=round_id,
        )
    ).inserted_primary_key[0]


def close_round(conn, round_id, ids, status, detail, action):
    conn.execute(
        update(rounds).where(rounds.c.id == round_id).values(status=status, detail=detail)
    )
    for uid in ids:
        record(conn, uid, action)


def abort_round(round_id, ids):
    with transaction() as conn:
        conn.execute(
            update(rounds)
            .where(rounds.c.id == round_id)
            .values(
                status="aborted",
                detail="Client failure, revocation or protocol validation failure. Reserved budget retained.",
            )
        )
        for uid in ids:
            record(conn, uid, "FL_ROUND_ABORTED")


def finish(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdin.close()
        process.stdout.close()


def purge(candidates):
    """No raw-example retention after an attempt, published or not."""
    with transaction() as conn:
        conn.execute(
            delete(examples).where(
                examples.c.id.in_([r["id"] for _, data in candidates for r in data])
            )
        )


def run_once():
    """Default stage: federate the full shared matrix."""
    processes = []
    with transaction() as conn:
        opened = begin_round(conn, "softmax")
    if not opened:
        return
    round_id, candidates = opened["round_id"], opened["candidates"]
    ids, weights = opened["ids"], opened["weights"]
    try:
        nonce = os.urandom(32).hex()
        for index, (user, data) in enumerate(candidates):
            processes.append(
                spawn(
                    "fl.client",
                    {
                        "uid": user["id"],
                        "index": index,
                        "key": base64.b64encode(user_key(user["id"])).decode(),
                        "examples": [r["ciphertext"] for r in data],
                        "weights": weights.tolist(),
                        "nonce": nonce,
                        "epsilon": EPSILON_PER_ROUND,
                        "delta": DELTA_PER_ROUND,
                    },
                )
            )
        masked, _ = exchange(processes, int(np.prod(SHAPE)) * 4)
        with transaction() as conn:
            recheck_consent(conn, ids, opened["consent_versions"])
            averaged = decode_sum(aggregate(masked, len(ids)), len(ids)).reshape(SHAPE)
            candidate = weights + averaged
            if not np.isfinite(candidate).all():
                raise RuntimeError("Nonfinite candidate")
            score = sanity_score(candidate)
            accepted = accept(score, sanity_score(weights))
            if accepted:
                publish(conn, candidate, score, round_id)
            close_round(
                conn,
                round_id,
                ids,
                "published" if accepted else "rejected",
                f"Public seed regression score {score:.3f}; "
                f"{'activated' if accepted else 'previous model retained'}. Budget consumed.",
                "FL_MODEL_PUBLISHED" if accepted else "FL_CANDIDATE_REJECTED",
            )
    except Exception:
        abort_round(round_id, ids)
    finally:
        finish(processes)
        purge(candidates)


def run_active_stage():
    """Dispatch to the configured stage. Both publish into `model_versions`."""
    if settings.learning_stage == "lora":
        from .lora import pipeline as lora_stage

        return lora_stage.run_once()
    return run_once()


def start():
    global _thread
    stop.clear()

    def loop():
        while not stop.is_set():
            trigger.wait(10)
            trigger.clear()
            if stop.is_set():
                break
            try:
                run_active_stage()
            except Exception:
                # No sensitive exception payloads in logs; next wake retries.
                import logging

                logging.getLogger("ppda").error(
                    "Pipeline transaction failed; retry deferred"
                )

    _thread = threading.Thread(target=loop, name="ppda-learning", daemon=True)
    _thread.start()


def shutdown():
    stop.set()
    trigger.set()
    if _thread:
        _thread.join(timeout=15)
