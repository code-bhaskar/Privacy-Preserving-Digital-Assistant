"""Download pinned public SNIPS data, train a classifier, export isolated ONNX, measure.

No user examples, credentials or application model files are read/written.
Canonical validation split is never used to fit/tune parameters. Public-data
central training is not labelled as federated or differentially private.
"""

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.snips import ARTIFACT, FEATURES, LABELS, load_session, vectorizer

REVISION = "b86ac7f1577868c42158d0dec77db50956046696"
REPO = "snipsco/nlu-benchmark"


def download(cache, label, split):
    filename = (
        f"train_{label}_full.json" if split == "train" else f"validate_{label}.json"
    )
    relative = f"2017-06-custom-intent-engines/{label}/{filename}"
    destination = cache / filename
    if not destination.exists():
        url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{relative}"
        try:
            r = httpx.get(url, timeout=30, follow_redirects=True)
            r.raise_for_status()
            raw = r.content
        except httpx.HTTPError:
            if not shutil.which("gh"):
                raise RuntimeError(
                    "SNIPS download failed. Retry with network access or install/authenticate GitHub CLI for the public-API fallback."
                )
            raw = base64.b64decode(
                subprocess.check_output(
                    [
                        "gh",
                        "api",
                        f"repos/{REPO}/contents/{relative}?ref={REVISION}",
                        "--jq",
                        ".content",
                    ],
                    timeout=45,
                )
            )
        if len(raw) > 20_000_000:
            raise ValueError("Unexpected dataset size")
        # Validate before caching. Some upstream files use Latin-1 encoding.
        decode_json(raw)
        destination.write_bytes(raw)
    raw = destination.read_bytes()
    data = decode_json(raw)
    utterances = [
        "".join(segment["text"] for segment in row["data"]).strip()
        for row in data[label]
    ]
    return utterances, {
        "path": relative,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "count": len(utterances),
    }


def decode_json(raw):
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("latin-1"))


def train(output):
    cache = output / "data" / REVISION
    cache.mkdir(parents=True, exist_ok=True)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    sources = []
    for label in LABELS:
        for split, x, y in [("train", train_x, train_y), ("validate", test_x, test_y)]:
            values, source = download(cache, label, split)
            x.extend(values)
            y.extend([label] * len(values))
            sources.append(source)
        print("Loaded " + label, flush=True)
    # Remove exact normalised train/validation overlap rather than overstate accuracy.
    norm = lambda value: " ".join(value.casefold().split())
    held_out = {norm(t) for t in test_x}
    filtered = [(t, y) for t, y in zip(train_x, train_y) if norm(t) not in held_out]
    removed = len(train_x) - len(filtered)
    train_x, train_y = map(list, zip(*filtered))
    start = time.perf_counter()
    transform = vectorizer()
    x = transform.transform(train_x)
    test = transform.transform(test_x)
    model = LogisticRegression(C=8.0, max_iter=300, solver="lbfgs", random_state=42)
    with threadpool_limits(limits=1):
        model.fit(x, train_y)
        predictions = model.predict(test)
    assert list(model.classes_) == LABELS
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["features", "weights"], ["linear"]),
            helper.make_node("Add", ["linear", "bias"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
        ],
        "snips-intent-benchmark",
        [
            helper.make_tensor_value_info(
                "features", TensorProto.FLOAT, [None, FEATURES]
            )
        ],
        [
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, len(LABELS)]
            )
        ],
        [
            numpy_helper.from_array(model.coef_.T.astype(np.float32), name="weights"),
            numpy_helper.from_array(model.intercept_.astype(np.float32), name="bias"),
        ],
    )
    artifact = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    artifact.metadata_props.add(key="purpose", value="snips-benchmark-only")
    artifact.metadata_props.add(key="labels", value=json.dumps(LABELS))
    onnx.checker.check_model(artifact)
    temporary = output / (ARTIFACT + ".tmp")
    temporary.write_bytes(artifact.SerializeToString())
    session = load_session(str(temporary), temporary.stat().st_mtime_ns)
    onnx_predictions = np.array(LABELS)[
        session.run(None, {"features": test.toarray()})[0].argmax(axis=1)
    ]
    if not np.array_equal(onnx_predictions, predictions):
        raise RuntimeError("ONNX export parity check failed")
    temporary.replace(output / ARTIFACT)
    metrics = {
        "dataset": "SNIPS 2017 custom intents",
        "source_repository": REPO,
        "source_revision": REVISION,
        "citation": "Coucke A. et al. (2018), Snips Voice Platform: an embedded Spoken Language Understanding system for private-by-design voice interfaces. https://arxiv.org/abs/1805.10190",
        "training": "Central training on public data; no DP/FL claim",
        "train_samples": len(train_x),
        "validation_samples": len(test_x),
        "exact_train_validation_overlaps_removed": removed,
        "features": FEATURES,
        "labels": LABELS,
        "model": "Hashed unigram/bigram multinomial logistic regression; C=8; seed=42",
        "validation_accuracy": float(accuracy_score(test_y, predictions)),
        "validation_macro_f1": float(f1_score(test_y, predictions, average="macro")),
        "onnx_prediction_parity": True,
        "onnx_bytes": (output / ARTIFACT).stat().st_size,
        "onnx_sha256": hashlib.sha256((output / ARTIFACT).read_bytes()).hexdigest(),
        "train_and_export_seconds": round(time.perf_counter() - start, 3),
        "classification_report": classification_report(
            test_y, predictions, output_dict=True
        ),
        "source_files": sources,
    }
    (output / "metrics.json.tmp").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "metrics.json.tmp").replace(output / "metrics.json")
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in [
                    "train_samples",
                    "validation_samples",
                    "validation_accuracy",
                    "validation_macro_f1",
                    "onnx_bytes",
                ]
            }
        ),
        flush=True,
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/snips")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train(args.output)
