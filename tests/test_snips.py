import importlib.util
from pathlib import Path

from app import snips
from app.config import settings

ROOT = Path(__file__).resolve().parent.parent


def test_reproducible_training_export_and_separate_namespace(
    tmp_path, monkeypatch, client, signup
):
    # Tiny offline fixture tests plumbing, not benchmark quality. Real metrics are documented separately.
    spec = importlib.util.spec_from_file_location(
        "train_snips", ROOT / "scripts/train_snips.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    words = [
        "playlist",
        "restaurant",
        "weather",
        "music",
        "bookrating",
        "creative",
        "screening",
    ]

    def fixture(cache, label, split):
        word = words[snips.LABELS.index(label)]
        values = (
            [f"{word} sample {i}" for i in range(6)]
            if split == "train"
            else [f"{word} held out"]
        )
        return values, {
            "path": "synthetic-test-fixture",
            "count": len(values),
            "sha256": "fixture-only",
        }

    monkeypatch.setattr(module, "download", fixture)
    metrics = module.train(tmp_path)
    assert metrics["train_samples"] == 42 and metrics["validation_samples"] == 7
    assert metrics["onnx_prediction_parity"]
    monkeypatch.setattr(settings, "snips_dir", str(tmp_path))
    assert not snips.status(tmp_path / "absent")["available"]
    assert client.get("/api/v1/models/snips/status").status_code == 401
    signup()
    status = client.get("/api/v1/models/snips/status").json()
    assert status["available"] and status["labels"] == snips.LABELS
    result = client.post(
        "/api/v1/models/snips/predict", json={"text": "weather forecast"}
    ).json()
    assert result["intent"] == "GetWeather" and "no action" in result["purpose"]
    workspace = client.post(
        "/api/v1/assistant/command",
        json={"text": "remind me to call Rahul tomorrow", "mode": "Privacy"},
    ).json()
    assert (
        workspace["intent"] == "reminder"
        and workspace["proposal"]["kind"] == "Reminders"
    )
    assert client.get("/api/v1/entries/Reminders").json() == []
