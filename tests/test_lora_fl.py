"""The optional low-rank federated stage: mechanism, protocol and end-to-end round.

The interesting claim this stage makes is a *measured* one, not a theoretical
one: for the same epsilon, delta, clip norm, cohort size, masking protocol and
ledger charge, releasing a rank-r adapter injects far less noise into the served
model than releasing the full matrix, because the Gaussian mechanism noises every
released coordinate and the adapter has r*L of them instead of (F+1)*L.
`test_low_rank_release_carries_less_noise_energy_for_the_same_budget` measures
that ratio by Monte Carlo instead of quoting the algebra.

What these tests do NOT claim: that a published round improves accuracy. At
epsilon=0.5 with three clients the released adapter is noise-dominated, and the
publication gate is a "did this break the served model" check on public seed
sentences, not a held-out benchmark. `test_status_states_the_noise_dominance`
pins that admission so it cannot quietly disappear from the API.
"""

import base64
import json

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from sqlalchemy import select

from app.config import settings
from app.database import adapters, engine, examples, rounds, transaction, users
from app.local_model import LABELS, SHAPE, sanity_score, seed_weights
from app.preferences import DEFAULTS, expenditure, prefs
from fl import pipeline
from fl.lora import backend
from fl.lora import pipeline as lora_stage
from fl.privacy import CLIP_NORM, DELTA_PER_ROUND, EPSILON_PER_ROUND, clip, quantize, sigma
from fl.protocol import mask

P = "/api/v1"
FULL_MATRIX = int(np.prod(SHAPE))


def enable_lora(monkeypatch, **overrides):
    monkeypatch.setattr("app.config.settings.learning_stage", "lora")
    for key, value in overrides.items():
        monkeypatch.setattr("app.config.settings." + key, value)
    return settings


def cohort(client, signup, count=3, label="reminder", texts=None):
    """Real accounts, real consent, real encrypted examples."""
    texts = texts or [
        "remind me to call mom",
        "remind me at five",
        "create a reminder tomorrow",
    ]
    ids = []
    for _ in range(count):
        user = signup()
        ids.append(user["id"])
        preferences = user["preferences"]
        preferences.update(training=True, epsilon=3.0)
        assert client.put(P + "/settings", json=preferences).status_code == 200
        for text in texts:
            assert (
                client.post(
                    P + "/learning/examples", json={"text": text, "label": label}
                ).status_code
                == 201
            )
    return ids


@pytest.fixture(autouse=True)
def clean_queue(client):
    """The test database is session-scoped.

    `begin_round` admits the first three eligible accounts it finds, so unused
    examples left behind by an earlier test would join this test's cohort and the
    assertions about who participated would be meaningless.
    """
    with transaction() as conn:
        conn.execute(examples.delete())
    yield


def round_mark():
    """Highest round id so far, so each test only inspects its own round."""
    with engine.connect() as conn:
        rows = conn.execute(select(rounds.c.id)).all()
    return max((row[0] for row in rows), default=0)


def rounds_since(mark):
    with engine.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                select(rounds).where(rounds.c.id > mark).order_by(rounds.c.id)
            ).mappings()
        ]


def adapters_since(mark):
    with engine.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                select(adapters).where(adapters.c.round_id > mark)
            ).mappings()
        ]


# --- the parameterisation ------------------------------------------------------


def test_projection_is_public_deterministic_and_well_conditioned():
    a = backend.projection(4)
    assert a.shape == (SHAPE[0], 4)
    # Derived from a committed seed by hashing, so every process agrees without
    # it being transmitted, and it cannot drift between NumPy versions.
    assert np.array_equal(a, backend.projection(4))
    assert np.allclose(np.linalg.norm(a, axis=0), 1.0)
    assert np.isfinite(a).all()
    # Raising the rank extends the projection instead of reshuffling it, so an
    # adapter published at rank r stays meaningful if the operator moves to r+1.
    assert np.allclose(backend.projection(5)[:, :4], backend.projection(4))
    assert not np.array_equal(backend.projection(5), np.pad(backend.projection(4), ((0, 0), (0, 1))))
    for rank in (0, -1, 33, 1000):
        with pytest.raises(ValueError):
            backend.projection(rank)


def test_a_zero_adapter_reproduces_the_base_model_exactly():
    base = seed_weights()
    adapter = backend.initialize(4)
    assert adapter.shape == (4, len(LABELS))
    assert np.array_equal(backend.merge(base, adapter, 4), base)
    assert backend.score(base, adapter, 4) == sanity_score(base)


def test_local_training_moves_only_the_adapter():
    base = seed_weights()
    frozen = base.copy()
    examples = [
        ("ping me about the dentist visit", "reminder"),
        ("nudge me to pay the electricity bill", "reminder"),
        ("do not forget the groceries", "reminder"),
        ("hello there", "chat"),
    ]
    adapter = backend.train(base, backend.initialize(4), examples, 4, steps=60, lr=0.5)
    assert np.array_equal(base, frozen), "the base matrix must be read-only"
    assert adapter.shape == (4, len(LABELS))
    assert np.isfinite(adapter).all()
    assert np.linalg.norm(adapter) > 0
    # A paraphrase the frozen base reads as "chat": local training moves
    # probability mass towards the client's own label. Four examples do not flip
    # the argmax, and claiming they do would overstate what one round achieves.
    paraphrase = examples[0][0]
    before = backend.probabilities_with(base, backend.initialize(4), 4, paraphrase)
    after = backend.probabilities_with(base, adapter, 4, paraphrase)
    assert backend.label_of(base, backend.initialize(4), 4, paraphrase) == "chat"
    reminder = LABELS.index("reminder")
    assert after[reminder] > before[reminder]
    assert after[LABELS.index("chat")] < before[LABELS.index("chat")]
    # Clipping is not only the sensitivity bound: it is also what keeps an
    # over-fitted local adapter from damaging the shared model.
    assert backend.score(base, clip(adapter, CLIP_NORM), 4) == sanity_score(base)
    with pytest.raises(ValueError):
        backend.train(base, backend.initialize(4), [], 4)
    with pytest.raises(ValueError):
        backend.train(base, backend.initialize(4), examples, 4, steps=0)


def test_the_released_vector_is_smaller_than_the_full_matrix():
    assert backend.vector_length(4) == 4 * len(LABELS) == 20
    assert FULL_MATRIX == SHAPE[0] * SHAPE[1] == 645
    assert lora_stage.released_length(4) < FULL_MATRIX
    assert lora_stage.released_length(32) < FULL_MATRIX


def test_low_rank_release_carries_less_noise_energy_for_the_same_budget():
    """Same epsilon, delta, clip and cohort: only the released dimension differs."""
    base = seed_weights()
    rng = np.random.default_rng(20260916)
    rank, clients, trials = 4, 3, 200
    std = sigma(EPSILON_PER_ROUND, DELTA_PER_ROUND, 2 * CLIP_NORM)

    projection = backend.projection(rank)
    adapter_noise = []
    matrix_noise = []
    for _ in range(trials):
        adapter_draws = [
            rng.normal(0, std, (rank, len(LABELS))) for _ in range(clients)
        ]
        merged = projection @ np.mean(adapter_draws, axis=0)
        adapter_noise.append(float(np.linalg.norm(merged)))
        matrix_draws = [rng.normal(0, std, SHAPE) for _ in range(clients)]
        matrix_noise.append(float(np.linalg.norm(np.mean(matrix_draws, axis=0))))
    adapter_rms = float(np.mean(adapter_noise))
    matrix_rms = float(np.mean(matrix_noise))
    predicted = np.sqrt((rank * len(LABELS)) / FULL_MATRIX)
    assert adapter_rms < 0.35 * matrix_rms
    assert adapter_rms / matrix_rms == pytest.approx(predicted, rel=0.15)


# --- stage selection -----------------------------------------------------------


def test_the_learning_thread_dispatches_on_the_configured_stage(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "run_once", lambda: calls.append("softmax"))
    monkeypatch.setattr(lora_stage, "run_once", lambda: calls.append("lora"))
    monkeypatch.setattr("app.config.settings.learning_stage", "softmax")
    pipeline.run_active_stage()
    monkeypatch.setattr("app.config.settings.learning_stage", "lora")
    pipeline.run_active_stage()
    assert calls == ["softmax", "lora"]


def test_status_reports_the_stage_and_its_real_numbers(client, signup, monkeypatch):
    signup()
    enable_lora(monkeypatch, lora_rank=4)
    status = client.get(P + "/learning/status").json()
    assert status["learning_stage"] == "lora"
    detail = status["stage_detail"]
    assert detail["released_coordinates"] == 4 * len(LABELS)
    assert detail["full_matrix_coordinates"] == FULL_MATRIX
    assert detail["dimension_reduction"] == pytest.approx(
        FULL_MATRIX / (4 * len(LABELS)), rel=1e-6
    )
    # The noise ratio is the square root of the coordinate ratio; reporting the
    # coordinate ratio alone would overstate the benefit by sqrt(32) times.
    assert detail["predicted_noise_energy_ratio"] == pytest.approx(
        np.sqrt((4 * len(LABELS)) / FULL_MATRIX), rel=1e-3
    )
    assert detail["predicted_noise_energy_ratio"] < 0.2
    assert detail["epsilon_per_round"] == EPSILON_PER_ROUND
    assert detail["delta_per_round"] == DELTA_PER_ROUND
    assert detail["noise_per_coordinate"] == pytest.approx(
        sigma(EPSILON_PER_ROUND, DELTA_PER_ROUND, 2 * detail["clip_norm"])
    )


def test_status_states_the_noise_dominance(monkeypatch):
    """The honest utility summary must stay in the API, not only in the docs."""
    enable_lora(monkeypatch)
    status = lora_stage.status()
    assert status["noise_per_coordinate"] > status["clip_norm"]
    assert "signal-to-noise" in status["note"]
    assert status["stage"] == "lora"
    assert "frozen" in status["parameterisation"]


# --- a real round, with three real worker processes ----------------------------


def test_end_to_end_lora_round_publishes_and_charges_the_ledger(
    client, signup, monkeypatch
):
    ids = cohort(client, signup)
    enable_lora(monkeypatch)
    before = client.get(P + "/learning/status").json()["model_version"]
    mark = round_mark()
    pipeline.stop.clear()
    lora_stage.run_once()

    status = client.get(P + "/learning/status").json()
    assert status["history"], "a real cohort must produce a round"
    assert status["history"][0]["stage"] == "lora"
    assert status["history"][0]["status"] in ("published", "rejected"), status
    assert status["epsilon_spent"] == EPSILON_PER_ROUND
    assert status["delta_spent"] == DELTA_PER_ROUND
    assert status["queued"] == 0
    with engine.connect() as conn:
        for uid in ids:
            assert expenditure(conn, uid) == (EPSILON_PER_ROUND, DELTA_PER_ROUND)
        row = (
            conn.execute(select(rounds).where(rounds.c.id == status["history"][0]["id"]))
            .mappings()
            .one()
        )
        assert row["stage"] == "lora"
        assert json.loads(row["participants"]) == ids
        assert "Rank-4 adapter" in row["detail"]
        assert str(FULL_MATRIX) in row["detail"]
    produced = adapters_since(mark)
    assert len(produced) == 1
    adapter_row = produced[0]
    assert adapter_row["rank"] == 4
    assert adapter_row["base_model_id"] == before
    assert adapter_row["round_id"] == row["id"]
    # The aggregated adapter is stored already noised; no example text is stored.
    aggregate_adapter = np.array(json.loads(adapter_row["weights"]))
    assert aggregate_adapter.shape == (4, len(LABELS))
    assert adapter_row["accepted"] == (status["history"][0]["status"] == "published")
    if adapter_row["accepted"]:
        assert adapter_row["merged_model_id"] == status["model_version"]
        assert status["model_version"] != before
        with engine.connect() as conn:
            published = pipeline.active_model(conn)[1]
        assert published.shape == SHAPE and np.isfinite(published).all()
        # Publishing through the shared table means the ONNX path serves it as-is.
        from app.onnx_model import probabilities

        assert probabilities(published, "remind me to call mom").shape == (len(LABELS),)
    else:
        assert adapter_row["merged_model_id"] is None
        assert status["model_version"] == before


def test_an_insufficient_cohort_never_fabricates_a_lora_round(client, signup, monkeypatch):
    cohort(client, signup, count=2)
    enable_lora(monkeypatch)
    mark = round_mark()
    pipeline.stop.clear()
    lora_stage.run_once()
    assert rounds_since(mark) == []
    assert adapters_since(mark) == []
    assert client.get(P + "/learning/status").json()["epsilon_spent"] == 0


def test_a_wrong_dimension_update_aborts_the_round(client, signup, monkeypatch):
    """A full-matrix worker cannot contribute to a low-rank round."""
    cohort(client, signup)
    enable_lora(monkeypatch)
    before = client.get(P + "/learning/status").json()["model_version"]
    # Same protocol, wrong released length: the supervisor validates dimensions
    # before it aggregates anything.
    real_spawn = pipeline.spawn
    monkeypatch.setattr(
        lora_stage.base,
        "spawn",
        lambda module, config: real_spawn(
            "fl.client", dict(config, weights=config["base"])
        ),
    )
    mark = round_mark()
    pipeline.stop.clear()
    lora_stage.run_once()
    produced = rounds_since(mark)
    assert len(produced) == 1
    assert produced[0]["status"] == "aborted"
    assert produced[0]["stage"] == "lora"
    assert client.get(P + "/learning/status").json()["model_version"] == before
    # The budget stays spent: the clients already performed a noised release.
    assert client.get(P + "/learning/status").json()["epsilon_spent"] == EPSILON_PER_ROUND
    assert adapters_since(mark) == []


def test_dropout_aborts_the_round_and_retains_the_budget(client, signup, monkeypatch):
    ids = cohort(client, signup)
    enable_lora(monkeypatch)
    before = client.get(P + "/learning/status").json()["model_version"]

    def disconnect(processes, timeout=45):
        processes[-1].kill()
        raise RuntimeError("Test dropout")

    monkeypatch.setattr(pipeline, "read_messages", disconnect)
    mark = round_mark()
    pipeline.stop.clear()
    lora_stage.run_once()
    status = client.get(P + "/learning/status").json()
    assert status["history"][0]["status"] == "aborted"
    assert status["history"][0]["stage"] == "lora"
    assert status["model_version"] == before
    assert status["epsilon_spent"] == EPSILON_PER_ROUND
    assert adapters_since(mark) == []
    with engine.connect() as conn:
        for uid in ids:
            assert expenditure(conn, uid) == (EPSILON_PER_ROUND, DELTA_PER_ROUND)


def test_consent_revoked_mid_round_aborts_before_aggregation(
    client, signup, monkeypatch
):
    ids = cohort(client, signup)
    enable_lora(monkeypatch)
    before = client.get(P + "/learning/status").json()["model_version"]

    def masked_zeros(processes, expected_bytes):
        """A protocol-valid contribution set, so the abort must come from the
        consent re-check and not from a transport failure."""
        assert expected_bytes == backend.vector_length(settings.lora_rank) * 4
        private = [X25519PrivateKey.generate() for _ in range(len(processes))]
        peers = [key.public_key().public_bytes_raw().hex() for key in private]
        nonce = "11" * 32
        zero = np.zeros((settings.lora_rank, len(LABELS)))
        vectors = [
            mask(quantize(zero, len(peers)), private[i], peers, i, nonce)
            for i in range(len(peers))
        ]
        # One participant withdraws after the clients have already trained.
        with engine.connect() as conn:
            row = (
                conn.execute(select(users).where(users.c.id == ids[-1])).mappings().one()
            )
        preferences = prefs(row)
        assert set(preferences) == set(DEFAULTS)
        preferences["training"] = False
        assert client.put(P + "/settings", json=preferences).status_code == 200
        return vectors, peers

    monkeypatch.setattr(lora_stage.base, "exchange", masked_zeros)
    mark = round_mark()
    pipeline.stop.clear()
    lora_stage.run_once()
    produced = rounds_since(mark)
    assert len(produced) == 1
    assert produced[0]["status"] == "aborted"
    assert client.get(P + "/learning/status").json()["model_version"] == before
    assert adapters_since(mark) == []
    with engine.connect() as conn:
        # Withdrawal does not refund a release that already happened locally.
        assert expenditure(conn, ids[-1]) == (EPSILON_PER_ROUND, DELTA_PER_ROUND)


def test_a_candidate_that_fails_the_gate_is_recorded_but_not_served(
    client, signup, monkeypatch
):
    ids = cohort(client, signup)
    enable_lora(monkeypatch)
    before = client.get(P + "/learning/status").json()["model_version"]
    # Force the public-seed gate to fail, as real noise sometimes does.
    monkeypatch.setattr(lora_stage, "sanity_score", lambda weights: 0.0)
    mark = round_mark()
    pipeline.stop.clear()
    lora_stage.run_once()
    produced = rounds_since(mark)
    assert len(produced) == 1
    assert produced[0]["status"] == "rejected"
    assert "previous model retained" in produced[0]["detail"]
    assert client.get(P + "/learning/status").json()["model_version"] == before
    rejected = adapters_since(mark)
    assert len(rejected) == 1
    adapter_row = rejected[0]
    assert adapter_row["accepted"] is False
    assert adapter_row["merged_model_id"] is None
    assert adapter_row["score"] == 0.0
    # A rejected candidate still cost the account its budget.
    with engine.connect() as conn:
        assert expenditure(conn, ids[0]) == (EPSILON_PER_ROUND, DELTA_PER_ROUND)


def test_adapter_parameters_are_validated_at_startup_not_mid_round():
    from app.config import Settings

    secret = base64.b64encode(b"t" * 32).decode()

    def build(**overrides):
        return Settings(
            jwt_secret="x" * 48, aes_master_key=secret, **overrides
        ).keys()

    for rank in (0, -1, 33):
        with pytest.raises(RuntimeError):
            build(lora_rank=rank)
    for clip_norm in (0, -1.0, 100.0):
        with pytest.raises(RuntimeError):
            build(lora_clip_norm=clip_norm)
    with pytest.raises(RuntimeError):
        build(lora_local_steps=0)
    with pytest.raises(RuntimeError):
        build(lora_learning_rate=0)
    assert build(lora_rank=4, lora_clip_norm=0.1) is not None
