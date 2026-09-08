# Public SNIPS intent benchmark

This small bundled ONNX classifier was actually trained with `scripts/train_snips.py`.
Its report is `metrics.json`, including source-file and ONNX SHA-256 digests.
Raw public data is downloaded to the ignored `data/<pinned-revision>/` directory.

- 13,773 training utterances after removing 11 exact normalised validation overlaps.
- 700 upstream validation utterances: accuracy **97.8571%**, macro-F1 **0.978472**.
- 8,192 hashed unigram/bigram features; multinomial logistic regression, C=8, seed=42.
- ONNX export: 229,815 bytes; predictions match the training library on every validation example.
- Seven original SNIPS labels. This model does **not** dispatch assistant workspace actions.
- Central training on **public** data; not a federated-learning or differential-privacy result.
- No slot extraction, conversational LLM or summarisation model is represented here.

Source: [SNIPS/sonos nlu-benchmark](https://github.com/snipsco/nlu-benchmark),
revision `b86ac7f1577868c42158d0dec77db50956046696`,
`2017-06-custom-intent-engines/*/train_*_full.json` and `validate_*.json`.
The upstream repository dedicates its dataset under CC0-1.0 and requests this citation:

Coucke, A., et al. (2018). **Snips Voice Platform: an embedded Spoken Language Understanding
system for private-by-design voice interfaces.** https://arxiv.org/abs/1805.10190

The original upstream benchmark evaluated slot filling per intent. These are our
seven-class intent results, **not** those original slot-filling scores. Hyperparameters
were fixed before evaluating the validation split. It is a validation benchmark, not an
independent production-task test, and does not measure calendar/reminder accuracy.
