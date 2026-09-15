"""Conservative client-local Gaussian mechanism + basic sequential composition.

Replace-one adjacency on a client's whole local dataset. Clipping each delta at
C gives sensitivity <= 2C. Classical Gaussian sufficient bound for 0<eps<=1:
std > sensitivity * sqrt(2 ln(1.25/delta)) / eps (Dwork & Roth, Theorem A.1).
No subsampling amplification or unimplemented RDP claims. Finite-precision Python
sampling is a research implementation, not an audited discrete Gaussian sampler.
"""

import math
import secrets
import numpy as np

EPSILON_PER_ROUND = 0.5
DELTA_PER_ROUND = 1e-6
DELTA_CAP = 1e-5
CLIP_NORM = 0.1
SCALE = 100_000

# Escalating an out-of-capability prompt to the global model is a separate
# release from a training round. k-RR (fl/text_dp.py) is pure epsilon-DP, so
# delta is 0 rather than an assumed 1e-6: charging a nonzero delta here would be
# conservative bookkeeping for a mechanism that does not need it. The epsilon
# charge is a *policy budget on the number of escalated releases* per account.
# It is deliberately NOT the composed text-DP bound (n_tokens * epsilon_token),
# which is computed per release and shown to the user instead. See
# docs/SECURITY_AND_DP.md, "Prompt escalation accounting".
EPSILON_ESCALATION = 0.25
DELTA_ESCALATION = 0.0


def clip(delta, norm=CLIP_NORM):
    if not np.isfinite(delta).all() or norm <= 0:
        raise ValueError("Invalid delta or norm")
    return delta * min(1.0, norm / max(float(np.linalg.norm(delta)), 1e-30))


def sigma(epsilon, delta, sensitivity):
    if not 0 < epsilon <= 1 or not 0 < delta < 1 or sensitivity <= 0:
        raise ValueError("Unsupported privacy parameters")
    return 1.01 * sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def privatize(
    delta,
    epsilon=EPSILON_PER_ROUND,
    privacy_delta=DELTA_PER_ROUND,
    clip_norm=CLIP_NORM,
):
    """Clip, then add Gaussian noise calibrated to the clipped sensitivity.

    `clip_norm` is a parameter because the low-rank stage (`fl/lora`) releases a
    much shorter vector and may choose a different clip. The guarantee is
    unchanged provided the noise is calibrated to the same constant used for
    clipping, which is what `sigma(..., 2 * clip_norm)` enforces here.
    """
    if not 0 < float(clip_norm):
        raise ValueError("Clip norm must be positive")
    bounded = clip(delta, clip_norm)
    std = sigma(epsilon, privacy_delta, 2 * clip_norm)
    rng = secrets.SystemRandom()
    noise = np.array(
        [rng.normalvariate(0.0, std) for _ in range(bounded.size)]
    ).reshape(bounded.shape)
    return bounded + noise


def quantize(delta, participants):
    if participants < 3 or not np.isfinite(delta).all():
        raise ValueError("Invalid cohort/vector")
    # Saturation is public deterministic post-processing AFTER the local DP mechanism.
    bound = (2**30 - 1) // participants
    return (
        np.clip(np.rint(delta.ravel() * SCALE), -bound, bound)
        .astype(np.int64)
        .astype(np.uint32)
    )


def decode_sum(vector, participants):
    return vector.view(np.int32).astype(float) / SCALE / participants
