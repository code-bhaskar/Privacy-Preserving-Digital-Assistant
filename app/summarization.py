"""Local seq2seq summaries: offline-only inference, bounded input and no cloud fallback."""

import importlib.util
import json
from functools import lru_cache
from pathlib import Path
import threading

from .config import settings
from .local_model import summary_body, summarize as extractive_summary

MODELS = {
    "flan-t5-small": ("google/flan-t5-small", "FLAN-T5-small", 512),
    "t5-small": ("google-t5/t5-small", "T5-small", 512),
    "distilbart-cnn": ("sshleifer/distilbart-cnn-12-6", "DistilBART-CNN-12-6", 1024),
}
LOCK = threading.Lock()


class SummaryUnavailable(RuntimeError):
    pass


class SummaryInputError(ValueError):
    pass


def model_directory():
    return Path(settings.local_summary_dir) / settings.local_summary_model


def engine_label():
    if settings.local_summary_model == "extractive":
        return "Local extractive summariser (no LLM)"
    return (
        "Local " + MODELS[settings.local_summary_model][1] + " (CPU, offline inference)"
    )


def status():
    if settings.local_summary_model == "extractive":
        return {
            "ready": True,
            "engine": engine_label(),
            "detail": "Explicit legacy extractive mode; no neural model.",
        }
    try:
        manifest = json.loads((model_directory() / "ppda-model.json").read_text())
        ready = manifest["repository"] == MODELS[settings.local_summary_model][
            0
        ] and all(
            importlib.util.find_spec(m) is not None
            for m in ("torch", "transformers", "sentencepiece")
        )
    except (OSError, ValueError, KeyError):
        ready = False
    return {
        "ready": ready,
        "engine": engine_label(),
        "detail": "Local model staged; first request loads it into CPU memory."
        if ready
        else "Install requirements-local-summary.txt and run scripts/download_summary_model.py on a network-enabled host. No fallback or request-time download.",
    }


@lru_cache(maxsize=1)
def load_model(name, directory):
    # The only downloadable step lives in an explicit operator script, never here.
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        .to("cpu")
        .eval()
    )
    return tokenizer, model


def summarize(text):
    body = summary_body(text)
    if not body:
        return "Please include the text you want summarised."
    if settings.local_summary_model == "extractive":
        return extractive_summary(text)
    if not status()["ready"]:
        raise SummaryUnavailable(status()["detail"])
    if not LOCK.acquire(blocking=False):
        raise SummaryUnavailable("The local summariser is busy. Please retry shortly.")
    try:
        import torch

        name = settings.local_summary_model
        tokenizer, model = load_model(name, str(model_directory()))
        # T5 expects a task prefix; DistilBART is already summarisation-fine-tuned.
        prompt = ("summarize: " if name in ("t5-small", "flan-t5-small") else "") + body
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
        limit = MODELS[name][2]
        if encoded["input_ids"].shape[-1] > limit:
            raise SummaryInputError(
                f"Text exceeds this model’s {limit}-token input limit. Split it into smaller sections; nothing was silently truncated."
            )
        with torch.inference_mode():
            output = model.generate(
                **encoded, max_new_tokens=128, num_beams=4, do_sample=False
            )
        result = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if not result:
            raise SummaryUnavailable(
                "The local model returned an empty summary. Please rephrase or shorten the input."
            )
        return result
    except (SummaryInputError, SummaryUnavailable):
        raise
    except Exception as exc:
        # Provider/library errors can contain paths or text; expose only a safe message.
        raise SummaryUnavailable(
            "Local summarisation failed. Verify the staged model and optional dependencies. Nothing was sent to the cloud."
        ) from exc
    finally:
        LOCK.release()
