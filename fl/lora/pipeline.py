"""Server side of the `lora` stage: federate a low-rank adapter, merge, gate, publish.

The base matrix is frozen for the whole round. Clients return a noised, masked
rank-r adapter; the server averages the masks off, merges the result into the
base, and only publishes it if the merged model still passes the public-seed
regression gate. A rejected candidate is recorded in `lora_adapters` with
``accepted = false`` — the budget stays spent and the artefact stays auditable.

Publication writes to the same `model_versions` table the default stage uses, so
the ONNX serving path, `/assistant/command` and the browser are untouched.
"""

import base64
import json
import math
import os
import time

import numpy as np
from sqlalchemy import select, update

from app.config import settings
from app.database import transaction, adapters
from app.security import user_key
from app.local_model import SHAPE, LABELS, sanity_score
from .. import pipeline as base
from ..privacy import EPSILON_PER_ROUND, DELTA_PER_ROUND, decode_sum, sigma
from ..protocol import aggregate
from .backend import adapter_shape, merge, vector_length

FULL_MATRIX_LENGTH = int(np.prod(SHAPE))


def released_length(rank):
    """Coordinates a client releases in this stage, versus the full matrix."""
    return vector_length(rank)


def run_once():
    processes = []
    with transaction() as conn:
        opened = base.begin_round(conn, "lora")
    if not opened:
        return
    round_id = opened["round_id"]
    candidates = opened["candidates"]
    ids = opened["ids"]
    weights = opened["weights"]
    rank = settings.lora_rank
    clip = settings.lora_clip_norm
    try:
        nonce = os.urandom(32).hex()
        for index, (user, data) in enumerate(candidates):
            processes.append(
                base.spawn(
                    "fl.lora.client",
                    {
                        "uid": user["id"],
                        "index": index,
                        "key": base64.b64encode(user_key(user["id"])).decode(),
                        "examples": [r["ciphertext"] for r in data],
                        # Read-only base: the client cannot alter the published model.
                        "base": weights.tolist(),
                        "rank": rank,
                        "steps": settings.lora_local_steps,
                        "lr": settings.lora_learning_rate,
                        "nonce": nonce,
                        "epsilon": EPSILON_PER_ROUND,
                        "delta": DELTA_PER_ROUND,
                        "clip_norm": clip,
                    },
                )
            )
        masked, _ = base.exchange(processes, released_length(rank) * 4)
        with transaction() as conn:
            base.recheck_consent(conn, ids, opened["consent_versions"])
            averaged = decode_sum(aggregate(masked, len(ids)), len(ids)).reshape(
                adapter_shape(rank)
            )
            if not np.isfinite(averaged).all():
                raise RuntimeError("Nonfinite adapter")
            merged = merge(weights, averaged, rank)
            baseline = sanity_score(weights)
            published = sanity_score(merged)
            accepted = base.accept(published, baseline)
            adapter_id = conn.execute(
                adapters.insert().values(
                    round_id=round_id,
                    rank=rank,
                    base_model_id=opened["model_version"],
                    weights=json.dumps(averaged.tolist()),
                    merged_model_id=None,
                    score=published,
                    accepted=accepted,
                    created_at=time.time(),
                )
            ).inserted_primary_key[0]
            merged_model_id = None
            if accepted:
                merged_model_id = base.publish(conn, merged, published, round_id)
                conn.execute(
                    update(adapters)
                    .where(adapters.c.id == adapter_id)
                    .values(merged_model_id=merged_model_id)
                )
            base.close_round(
                conn,
                round_id,
                ids,
                "published" if accepted else "rejected",
                detail(rank, published, accepted, clip, merged_model_id),
                "FL_LORA_PUBLISHED" if accepted else "FL_LORA_REJECTED",
            )
    except Exception:
        base.abort_round(round_id, ids)
    finally:
        base.finish(processes)
        base.purge(candidates)


def detail(rank, score, accepted, clip, merged_model_id):
    """Round detail is shown in the UI, so it states the trade-off plainly.

    The noise scale is public information: it is computed from committed
    constants (epsilon, delta, clip) and reveals nothing about any client.
    """
    noise = sigma(EPSILON_PER_ROUND, DELTA_PER_ROUND, 2 * clip)
    return (
        f"Rank-{rank} adapter over a frozen base: {released_length(rank)} released "
        f"coordinates instead of {FULL_MATRIX_LENGTH}. Public seed regression score "
        f"{score:.3f}; "
        + (
            f"merged into model v{merged_model_id}."
            if accepted
            else "previous model retained."
        )
        + f" Gaussian sigma {noise:.2f} per coordinate at eps={EPSILON_PER_ROUND}, "
        f"clip={clip}. Budget consumed."
    )


def history(limit=5):
    """Adapter lineage for the status endpoint. Metadata only, no example text."""
    with transaction() as conn:
        rows = (
            conn.execute(select(adapters).order_by(adapters.c.id.desc()).limit(limit))
            .mappings()
            .all()
        )
    return [
        {
            "id": row["id"],
            "round_id": row["round_id"],
            "rank": row["rank"],
            "base_model_id": row["base_model_id"],
            "merged_model_id": row["merged_model_id"],
            "score": row["score"],
            "accepted": bool(row["accepted"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def status():
    """What an operator needs to see to judge whether this stage is doing anything.

    `noise_per_coordinate` versus `clip_norm` is the honest utility summary: at
    the default epsilon the mechanism injects several times the clipped signal
    norm into every released coordinate, so a single round with three clients is
    noise-dominated by construction. Averaging over the cohort divides the noise
    by sqrt(cohort), which is the only lever that improves it without spending
    more epsilon.
    """
    rank = settings.lora_rank
    clip = settings.lora_clip_norm
    noise = sigma(EPSILON_PER_ROUND, DELTA_PER_ROUND, 2 * clip)
    return {
        "stage": "lora",
        "parameterisation": "W = W0 + A @ B; W0 frozen, B rank-%d, zero-initialised" % rank,
        "rank": rank,
        "released_coordinates": released_length(rank),
        "full_matrix_coordinates": FULL_MATRIX_LENGTH,
        "dimension_reduction": round(
            FULL_MATRIX_LENGTH / max(1, released_length(rank)), 2
        ),
        # The coordinate ratio above is not the noise ratio: noise energy scales
        # with the square root of the released dimension. Quoting 32x instead of
        # 5.7x would overstate the benefit, so both are reported.
        "predicted_noise_energy_ratio": round(
            math.sqrt(released_length(rank) / FULL_MATRIX_LENGTH), 4
        ),
        "clip_norm": clip,
        "epsilon_per_round": EPSILON_PER_ROUND,
        "delta_per_round": DELTA_PER_ROUND,
        "noise_per_coordinate": noise,
        "local_steps": settings.lora_local_steps,
        "local_learning_rate": settings.lora_learning_rate,
        "labels": list(LABELS),
        "initialised": bool(history(1)),
        "adapters": history(),
        "note": (
            "Noise is calibrated to the clip, so sigma scales with it: raising the "
            "clip raises the signal and the noise together and does not improve the "
            "signal-to-noise ratio. Cohort size does, by sqrt(n). At these "
            "parameters a published adapter is still noise-dominated, and the gate "
            "asks whether the served model broke, not whether it improved. Measure "
            "both stages with scripts/measure_dp_stages.py."
        ),
    }
