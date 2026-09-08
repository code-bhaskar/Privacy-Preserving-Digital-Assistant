"""In-memory ONNX export/serving of the active, schema-checked shared model."""

from functools import lru_cache
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
from .local_model import SHAPE, LABELS, features


@lru_cache(maxsize=2)
def session_for(weights_bytes):
    weights = np.frombuffer(weights_bytes, dtype=np.float32).reshape(SHAPE)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["features", "weights"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
        ],
        "ppda-intent-v1",
        [
            helper.make_tensor_value_info(
                "features", TensorProto.FLOAT, [None, SHAPE[0]]
            )
        ],
        [
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, len(LABELS)]
            )
        ],
        [numpy_helper.from_array(weights, name="weights")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    model.metadata_props.add(key="labels", value=",".join(LABELS))
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    if session.get_outputs()[0].shape[-1] != len(LABELS):
        raise ValueError("Model label-space mismatch")
    return session


def probabilities(weights, text):
    if weights.shape != SHAPE or not np.isfinite(weights).all():
        raise ValueError("Invalid model weights")
    session = session_for(np.asarray(weights, dtype=np.float32).tobytes())
    return session.run(None, {"features": features(text)[None, :].astype(np.float32)})[
        0
    ][0]
