"""Operator-only model staging; never run by a summarisation request.

Requires requirements-local-summary.txt. Downloads public pretrained model assets,
not private text. Records an immutable upstream revision. Copy the staged directory
onto an offline backend if that backend cannot access Hugging Face.
"""

import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

MODELS = {
    "flan-t5-small": "google/flan-t5-small",
    "t5-small": "google-t5/t5-small",
    "distilbart-cnn": "sshleifer/distilbart-cnn-12-6",
}
ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="flan-t5-small")
    parser.add_argument(
        "--directory", type=Path, default=ROOT / ".runtime/summary-models"
    )
    parser.add_argument(
        "--revision",
        help="Optional immutable Hugging Face commit to reproduce a prior installation",
    )
    args = parser.parse_args()
    repo = MODELS[args.model]
    info = HfApi().model_info(repo, revision=args.revision or "main")
    directory = args.directory / args.model
    # Never overwrite an active installation in place.
    if directory.exists():
        raise SystemExit(
            "Destination already exists. Stage in another directory, verify it, then change LOCAL_SUMMARY_DIR."
        )
    files = {f.rfilename for f in info.siblings}
    weights = (
        ["*.safetensors", "*.safetensors.index.json"]
        if "model.safetensors" in files or "model.safetensors.index.json" in files
        else ["pytorch_model*.bin", "pytorch_model.bin.index.json"]
    )
    snapshot_download(
        repo_id=repo,
        revision=info.sha,
        local_dir=directory,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer*",
            "special_tokens_map.json",
            "spiece.model",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "README.md",
            "LICENSE*",
            *weights,
        ],
    )
    # Verify local loading before declaring the installation ready. Remote code prohibited.
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    AutoModelForSeq2SeqLM.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    (directory / "ppda-model.json").write_text(
        json.dumps(
            {"repository": repo, "revision": info.sha, "model": args.model}, indent=2
        )
        + "\n"
    )
    print(
        f"Staged {repo} at revision {info.sha}. Set LOCAL_SUMMARY_MODEL={args.model} and restart the backend."
    )


if __name__ == "__main__":
    main()
