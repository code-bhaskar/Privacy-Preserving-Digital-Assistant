# Local neural summarisation

The UI is **Angular 20**, not React. The API is **FastAPI**. Local means the backend
host (not browser-only execution). This update makes **FLAN-T5-small** the default
local summariser, with T5-small and DistilBART-CNN alternatives.

| LOCAL_SUMMARY_MODEL | Pretrained checkpoint | Input limit |
| --- | --- | --- |
| `flan-t5-small` (default) | `google/flan-t5-small` | 512 tokens including prefix |
| `t5-small` | `google-t5/t5-small` | 512 tokens including prefix |
| `distilbart-cnn` | `sshleifer/distilbart-cnn-12-6` | 1024 tokens |
| `extractive` | Explicit legacy sentence ranking, no neural model | Existing request bound |

FLAN-T5-small is a lightweight instruction-tuned starting point, not a model fine-tuned
here on a summarisation corpus. T5-small supports a `summarize:` prefix. DistilBART-CNN
is summarisation-fine-tuned but substantially larger and slower on CPU. SNIPS training
remains a **separate intent classifier**, not training of any summariser.

## Install on a network-enabled operator host

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-local-summary.txt
.venv/bin/python scripts/download_summary_model.py --model flan-t5-small
```

Set `LOCAL_SUMMARY_MODEL=flan-t5-small` in the backend environment and restart it.
The default model root is `.runtime/summary-models`; `LOCAL_SUMMARY_DIR` overrides it.
For alternatives, download the corresponding `--model` and change the environment.
The downloader resolves and records an immutable upstream revision in `ppda-model.json`;
use `--revision <recorded-sha>` to reproduce it. It validates local loading before writing
that manifest and refuses to overwrite an existing installation. Stage replacements
under another `--directory` and change the configured root after verifying them.

Download weights once on a connected host and securely copy the model directory to an
offline backend if needed. The optional CPU dependencies and weights can be large;
weights/cache files are excluded from Git. Review upstream model cards and licenses.
Never put user documents into the download command or model repository.

## Inference and failure behaviour

- Runtime loading always uses `local_files_only=True`, `trust_remote_code=False`, CPU
  evaluation and inference mode. Summarisation requests never download assets or call a
  remote provider. Existing summary processing consent remains required.
- One generation at a time bounds concurrent CPU/memory use. Other generation requests
  receive a retriable 503 instead of an unbounded queue. Work runs outside the async loop.
- Beam search is deterministic (`do_sample=False`) with at most 128 generated tokens.
  Oversized inputs receive 422 with instructions to split them; input is not silently
  truncated. Check source text against neural summaries: hallucination is possible.
- Missing dependencies/weights or generation failure returns an explicit 503. There is
  **no automatic extractive or cloud fallback**. The old extractive implementation is
  available only when the operator explicitly selects `extractive`, including offline CI.
- Settings and API runtime/response metadata show the actual configured engine and setup
  status. Global still calls the configured OpenAI model (default `gpt-4o-mini`) with cloud
  consent. Ollama remains a separate local general-chat adapter.

## Validation limitations in this session

Offline unit tests exercise all three model adapters using stand-in tensors/tokenizers,
including CPU/offline load flags, generation parameters, missing-model failure, input bounds
and no fallback. Existing end-to-end UI tests deliberately select the extractive fixture;
they do not pretend to exercise downloaded neural weights. The sandbox could not reach
Hugging Face or the PyTorch CPU index (TLS connection failures), so real neural-model
inference and summary quality were **not validated here**. Complete the installation above
on a connected host and run a real summary before considering the neural feature ready.
