"""Optional second federated stage: low-rank adapter learning.

Selected with `LEARNING_STAGE=lora`. The default stage (`softmax`, implemented in
`fl/pipeline.py`) federates a full copy of the shared intent matrix; this stage
freezes that matrix and federates a rank-`r` adapter instead, using the same
consent gates, the same lifetime privacy ledger, the same client-local Gaussian
mechanism, the same pairwise masking and the same public-seed publication gate.

Both stages publish into `model_versions`, so serving, the ONNX graph and the
browser are unchanged whichever stage is active.
"""
