from contextlib import nullcontext
import sys
from types import SimpleNamespace
import pytest
from app import summarization as summary
from app.config import settings


@pytest.mark.parametrize("name", ["flan-t5-small", "t5-small", "distilbart-cnn"])
def test_neural_generation_is_cpu_bounded_and_offline(name, monkeypatch):
    monkeypatch.setattr(settings, "local_summary_model", name)
    seen = {}

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return cls()

        def __call__(self, prompt, **kwargs):
            seen["prompt"] = prompt
            assert kwargs == {"return_tensors": "pt", "truncation": False}
            return {"input_ids": SimpleNamespace(shape=(1, 10))}

        def decode(self, value, **kwargs):
            return "The project launched successfully."

    class Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return cls()

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            seen.update(kwargs)
            return [[1, 2, 3]]

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda n: None, inference_mode=nullcontext),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=Tokenizer, AutoModelForSeq2SeqLM=Model),
    )
    monkeypatch.setattr(summary, "status", lambda: {"ready": True})
    summary.load_model.cache_clear()
    try:
        result = summary.summarize("Summarize: The project launched. Users signed up.")
        assert result == "The project launched successfully."
        assert seen["do_sample"] is False and seen["max_new_tokens"] == 128
        assert seen["prompt"].startswith("summarize: ") == (name != "distilbart-cnn")
    finally:
        summary.load_model.cache_clear()


def test_long_input_rejected_not_truncated(monkeypatch):
    monkeypatch.setattr(settings, "local_summary_model", "flan-t5-small")
    monkeypatch.setattr(summary, "status", lambda: {"ready": True})
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext)
    )
    monkeypatch.setattr(
        summary,
        "load_model",
        lambda *a: (
            lambda *a, **kw: {"input_ids": SimpleNamespace(shape=(1, 513))},
            None,
        ),
    )
    with pytest.raises(
        summary.SummaryInputError, match="nothing was silently truncated"
    ):
        summary.summarize("Some source text")
    assert not summary.LOCK.locked()


@pytest.mark.parametrize("mode", ["Default", "Privacy"])
def test_unavailable_neural_model_has_no_cloud_or_extractive_fallback(
    client, signup, monkeypatch, tmp_path, mode
):
    signup()
    monkeypatch.setattr(settings, "local_summary_model", "flan-t5-small")
    monkeypatch.setattr(settings, "local_summary_dir", str(tmp_path))

    async def forbidden(*a):
        raise AssertionError("Cloud fallback")

    monkeypatch.setattr("app.main.cloud_answer", forbidden)
    monkeypatch.setattr(
        summary, "extractive_summary", lambda *a: pytest.fail("Silent fallback")
    )
    r = client.post(
        "/api/v1/assistant/command",
        json={"text": "Summarize: A project launched today.", "mode": mode},
    )
    assert r.status_code == 503 and "Install" in r.json()["detail"]
    runtime = client.get("/api/v1/runtime").json()
    assert (
        not runtime["local_summary"]["ready"]
        and "FLAN-T5-small" in runtime["summarization_engine"]
    )


def test_busy_and_empty_input(monkeypatch):
    monkeypatch.setattr(settings, "local_summary_model", "flan-t5-small")
    monkeypatch.setattr(summary, "status", lambda: {"ready": True})
    assert (
        summary.summarize("Summarize:")
        == "Please include the text you want summarised."
    )
    summary.LOCK.acquire()
    try:
        with pytest.raises(summary.SummaryUnavailable, match="busy"):
            summary.summarize("Source text")
    finally:
        summary.LOCK.release()
