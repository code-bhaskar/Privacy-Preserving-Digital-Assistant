"""Separate seven-label SNIPS benchmark. Never dispatch these labels as CRUD intents."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.feature_extraction.text import HashingVectorizer

LABELS = [
    "AddToPlaylist",
    "BookRestaurant",
    "GetWeather",
    "PlayMusic",
    "RateBook",
    "SearchCreativeWork",
    "SearchScreeningEvent",
]
FEATURES = 8192
ARTIFACT = "intent_model_snips.onnx"


def vectorizer():
    return HashingVectorizer(
        n_features=FEATURES,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=True,
        dtype=np.float32,
    )


@lru_cache(maxsize=2)
def load_session(path, modified):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    if (
        meta.get("purpose") != "snips-benchmark-only"
        or json.loads(meta.get("labels", "[]")) != LABELS
        or session.get_inputs()[0].shape[-1] != FEATURES
        or session.get_outputs()[0].shape[-1] != len(LABELS)
    ):
        raise ValueError("Incompatible SNIPS artifact; not an assistant task model")
    return session


def status(directory):
    path = Path(directory)
    report = path / "metrics.json"
    if not report.is_file() or not (path / ARTIFACT).is_file():
        return {
            "available": False,
            "detail": "Train the public benchmark with scripts/train_snips.py.",
        }
    try:
        values = json.loads(report.read_text())
        if (
            values.get("onnx_sha256")
            != hashlib.sha256((path / ARTIFACT).read_bytes()).hexdigest()
        ):
            return {
                "available": False,
                "detail": "SNIPS artifact/report integrity mismatch; retrain the benchmark.",
            }
    except (OSError, ValueError):
        return {
            "available": False,
            "detail": "SNIPS report is unavailable or malformed.",
        }
    return {
        "available": True,
        "labels": LABELS,
        "metrics": values,
        "detail": "Public SNIPS benchmark only. Not the assistant CRUD model or private-user FL.",
    }


def predict(directory, text):
    path = Path(directory) / ARTIFACT
    session = load_session(str(path), path.stat().st_mtime_ns)
    x = vectorizer().transform([text]).toarray()
    p = session.run(None, {"features": x})[0][0]
    return {
        "intent": LABELS[int(p.argmax())],
        "confidence": float(p.max()),
        "processing_location": "local backend",
        "purpose": "SNIPS benchmark only; no action executed",
    }
