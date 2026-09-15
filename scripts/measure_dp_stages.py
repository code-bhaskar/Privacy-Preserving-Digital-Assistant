#!/usr/bin/env python3
"""Measure what the two federated stages actually cost and deliver.

Run it before writing numbers into a report:

    .venv/bin/python scripts/measure_dp_stages.py --trials 200

What it does
------------
Monte-Carlo simulation of the *mechanism*, not of user data. It draws the same
Gaussian noise the clients draw (`fl/privacy.sigma`, calibrated to epsilon, delta
and the clip norm), averages it over a cohort the way secure aggregation does,
merges it into the public seed model, and reports:

  * the noise energy that reaches the served model, per stage;
  * how often the merged candidate would pass the public-seed publication gate;
  * how the ratio tracks sqrt(released_coordinates / full_matrix_coordinates).

No private data is involved: the base model is the public seed model in
`app/local_model.py` and the gate sentences are public in the same file. Nothing
is written to the database and no network call is made.

What the numbers mean
---------------------
At the shipped defaults (epsilon=0.5, delta=1e-6, clip=0.1, three clients) the
full-matrix stage injects noise several times the norm of the model itself and
its candidates are rejected essentially always; the rank-4 adapter stage injects
sqrt(20/645) of that energy and its candidates pass. That is the utility
argument for federating an adapter rather than a matrix, at identical privacy
cost — the epsilon charged to the account is the same either way.

It is NOT an accuracy claim. The gate asks "did this break the served model",
not "did this improve it", and at these parameters a published round is still
noise-dominated. Say so wherever these numbers are quoted.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.local_model import LABELS, SHAPE, sanity_score, seed_weights  # noqa: E402
from fl.lora import backend  # noqa: E402
from fl.privacy import DELTA_PER_ROUND, EPSILON_PER_ROUND  # noqa: E402
from fl.privacy import sigma as mechanism_sigma  # noqa: E402

FULL_MATRIX = int(np.prod(SHAPE))


def gate_for(baseline):
    """The publication rule from fl/pipeline.accept, restated for measurement."""
    return max(0.75, baseline - 0.025)


def sigma(epsilon, delta, clip):
    """Noise scale the clients use: sensitivity is 2 * clip (replace-one)."""
    return mechanism_sigma(epsilon, delta, 2 * clip)


def simulate_matrix(base, baseline, std, clients, trials, rng):
    gate = gate_for(baseline)
    norms, accepted = [], 0
    for _ in range(trials):
        draws = [rng.normal(0, std, SHAPE) for _ in range(clients)]
        averaged = np.mean(draws, axis=0)
        norms.append(float(np.linalg.norm(averaged)))
        candidate = base + averaged
        if np.isfinite(candidate).all() and sanity_score(candidate) >= gate:
            accepted += 1
    return accepted / trials, float(np.mean(norms))


def simulate_adapter(base, baseline, std, rank, clients, trials, rng):
    gate = gate_for(baseline)
    projection = backend.projection(rank)
    norms, accepted = [], 0
    for _ in range(trials):
        draws = [rng.normal(0, std, (rank, len(LABELS))) for _ in range(clients)]
        averaged = np.mean(draws, axis=0)
        merged = projection @ averaged
        norms.append(float(np.linalg.norm(merged)))
        if sanity_score(backend.merge(base, averaged, rank)) >= gate:
            accepted += 1
    return accepted / trials, float(np.mean(norms))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--ranks", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    parser.add_argument("--clips", type=float, nargs="*", default=[0.1])
    parser.add_argument("--epsilon", type=float, default=EPSILON_PER_ROUND)
    parser.add_argument("--delta", type=float, default=DELTA_PER_ROUND)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args(argv)
    if args.clients < 3:
        parser.error("the protocol requires at least three clients")

    base = seed_weights()
    baseline = sanity_score(base)
    rng = np.random.default_rng(args.seed)
    model_norm = float(np.linalg.norm(base))

    print(__doc__.strip().splitlines()[0])
    print()
    print(f"public seed model      : {SHAPE[0]}x{SHAPE[1]} = {FULL_MATRIX} weights, norm {model_norm:.2f}")
    print(f"public seed gate score : {baseline:.3f} (publish needs >= {gate_for(baseline):.3f})")
    print(f"privacy per round      : epsilon={args.epsilon}, delta={args.delta}")
    print(f"cohort                 : {args.clients} clients, {args.trials} trials each")
    print()
    header = f"{'stage':<18}{'released':>9}{'clip':>7}{'sigma':>8}{'noise norm':>12}{'vs model':>10}{'published':>11}"
    print(header)
    print("-" * len(header))

    for clip in args.clips:
        std = sigma(args.epsilon, args.delta, clip)
        rate, norm = simulate_matrix(
            base, baseline, std, args.clients, args.trials, rng
        )
        print(
            f"{'full matrix':<18}{FULL_MATRIX:>9}{clip:>7.3f}{std:>8.2f}"
            f"{norm:>12.2f}{norm / model_norm:>9.1f}x{rate:>10.0%}"
        )
        for rank in args.ranks:
            released = rank * len(LABELS)
            rate, norm = simulate_adapter(
                base, baseline, std, rank, args.clients, args.trials, rng
            )
            predicted = np.sqrt(released / FULL_MATRIX)
            print(
                f"{'adapter rank ' + str(rank):<18}{released:>9}{clip:>7.3f}{std:>8.2f}"
                f"{norm:>12.2f}{norm / model_norm:>9.1f}x{rate:>10.0%}"
                f"   (predicted noise ratio vs full matrix: {predicted:.3f})"
            )
        print()

    print("Reading this table")
    print("  * 'noise norm' is the Frobenius norm of what the merge adds to the model.")
    print("  * 'vs model' compares it with the model's own norm: above ~1.0 the served")
    print("    model is mostly noise, which is why the gate rejects it.")
    print("  * 'published' is the share of simulated rounds that would pass the gate.")
    print("  * The epsilon charged to a user's lifetime ledger is identical in every")
    print("    row. Only the released dimension changes.")
    print()
    print("This is a simulation of the mechanism, not a measurement of real training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
