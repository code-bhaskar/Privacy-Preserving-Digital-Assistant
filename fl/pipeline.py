"""Single product pipeline. Real opted-in user queues only, no fabricated clients.

This release demonstrates client OS processes on one trusted host, NOT physical
user-device isolation. The supervisor sees encrypted examples and public keys,
then only masked updates. The host operator can access process memory/keys.
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
from app.config import ROOT
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


def run_once():
    processes = []
    with transaction() as conn:
        if conn.execute(
            select(rounds.c.id).where(rounds.c.status == "training")
        ).first():
            return
        candidates = []
        for user in conn.execute(select(users)).mappings():
            if not prefs(user)["training"] or not affordable(conn, user):
                continue
            data = (
                conn.execute(
                    select(examples)
                    .where(examples.c.user_id == user["id"], examples.c.used == False)
                    .limit(20)
                )
                .mappings()
                .all()
            )
            if len(data) >= 3:
                candidates.append((user, data))
        if len(candidates) < 3:
            return
        candidates = candidates[:3]
        version, weights = active_model(conn)
        ids = [u["id"] for u, _ in candidates]
        round_id = conn.execute(
            rounds.insert().values(
                status="training",
                participants=json.dumps(ids),
                created_at=time.time(),
                detail="Real client workers training; budget reserved.",
            )
        ).inserted_primary_key[0]
        # Conservative pre-reservation. No refunds, even if rejected/aborted; avoids retry leaks.
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
    versions = {
        u["id"]: json.loads(u["preferences"]).get("_consent_version", 0)
        for u, _ in candidates
    }
    try:
        nonce = os.urandom(32).hex()
        for index, (user, data) in enumerate(candidates):
            process = subprocess.Popen(
                [sys.executable, "-m", "fl.client"],
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
            processes.append(process)
            config = {
                "uid": user["id"],
                "index": index,
                "key": base64.b64encode(user_key(user["id"])).decode(),
                "examples": [r["ciphertext"] for r in data],
                "weights": weights.tolist(),
                "nonce": nonce,
                "epsilon": EPSILON_PER_ROUND,
                "delta": DELTA_PER_ROUND,
            }
            process.stdin.write(json.dumps(config) + "\n")
            process.stdin.flush()
        advertisements = read_messages(processes)
        peers = [msg["public_key"] for msg in advertisements]
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
            if len(raw) != np.prod(SHAPE) * 4:
                raise RuntimeError("Invalid update dimensions")
            masked.append(np.frombuffer(raw, dtype="<u4"))
        # Recheck consent before aggregate computation/release; abort retains reserved budget.
        with transaction() as conn:
            for uid in ids:
                user = (
                    conn.execute(select(users).where(users.c.id == uid))
                    .mappings()
                    .one()
                )
                if (
                    not prefs(user)["training"]
                    or json.loads(user["preferences"]).get("_consent_version", 0)
                    != versions[uid]
                ):
                    raise RuntimeError("Consent revoked")
            averaged = decode_sum(aggregate(masked, len(ids)), len(ids)).reshape(SHAPE)
            candidate = weights + averaged
            if not np.isfinite(candidate).all():
                raise RuntimeError("Nonfinite candidate")
            score = sanity_score(candidate)
            accepted = score >= max(0.75, sanity_score(weights) - 0.025)
            if accepted:
                conn.execute(update(models).values(active=False))
                conn.execute(
                    models.insert().values(
                        weights=json.dumps(candidate.tolist()),
                        created_at=time.time(),
                        score=score,
                        active=True,
                        round_id=round_id,
                    )
                )
            conn.execute(
                update(rounds)
                .where(rounds.c.id == round_id)
                .values(
                    status="published" if accepted else "rejected",
                    detail=f"Public seed regression score {score:.3f}; {'activated' if accepted else 'previous model retained'}. Budget consumed.",
                )
            )
            for uid in ids:
                record(
                    conn,
                    uid,
                    "FL_MODEL_PUBLISHED" if accepted else "FL_CANDIDATE_REJECTED",
                )
    except Exception:
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
    finally:
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
        with transaction() as conn:
            # No raw-example retention after this attempt; ledger and aggregate metadata remain.
            conn.execute(
                delete(examples).where(
                    examples.c.id.in_([r["id"] for _, data in candidates for r in data])
                )
            )


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
                run_once()
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
