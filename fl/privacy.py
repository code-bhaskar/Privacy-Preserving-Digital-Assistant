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


def clip(delta, norm=CLIP_NORM):
    if not np.isfinite(delta).all() or norm <= 0:
        raise ValueError("Invalid delta or norm")
    return delta * min(1.0, norm / max(float(np.linalg.norm(delta)), 1e-30))


def sigma(epsilon, delta, sensitivity):
    if not 0 < epsilon <= 1 or not 0 < delta < 1 or sensitivity <= 0:
        raise ValueError("Unsupported privacy parameters")
    return 1.01 * sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def privatize(delta, epsilon=EPSILON_PER_ROUND, privacy_delta=DELTA_PER_ROUND):
    bounded = clip(delta)
    std = sigma(epsilon, privacy_delta, 2 * CLIP_NORM)
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
