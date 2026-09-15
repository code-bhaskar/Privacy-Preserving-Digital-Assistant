"""The escalation DP mechanism: empirical privacy behaviour, redaction and honesty
of the reported numbers. Statistical, so tolerances are explicit."""

import math

import pytest

from fl import text_dp


def test_vocabulary_is_public_and_usable():
    words = text_dp.vocabulary()
    assert len(words) > 1000
    assert len(set(words)) == len(words)
    # The output space is public: nothing user-derived, and no PII-looking token.
    assert "reminder" in words and "calendar" in words
    assert not any("@" in word or word.isdigit() for word in words)


def test_pii_is_redacted_before_any_probabilistic_step():
    release = text_dp.perturb(
        "Email rahul.kumar@example.com or call 98765 43210, see https://x.io/a "
        "and password: hunter2 please",
        epsilon_token=20,
    )
    for secret in ("rahul.kumar@example.com", "98765", "hunter2", "https://x.io/a"):
        assert secret not in release["text"]
    kinds = {entry["type"] for entry in release["redactions"]}
    assert {"email", "phone-or-id", "url", "credential"} <= kinds


def test_words_outside_the_public_vocabulary_are_redacted():
    release = text_dp.perturb("Tell Zigglethorpe about the meeting", epsilon_token=20)
    assert "Zigglethorpe" not in release["text"]
    assert text_dp.REDACTED in release["text"]
    assert any(change["from"] == "Zigglethorpe" for change in release["perturbed"])


def test_function_words_are_kept_and_reported_as_unprotected():
    release = text_dp.perturb("what is the meaning of this", epsilon_token=20)
    assert release["text"].startswith("what is the")
    assert release["protected_tokens"] <= 2


def test_empirical_krr_rate_matches_the_claimed_epsilon():
    """P(keep)/P(any one other word) must be about e^epsilon for a real k-RR."""
    epsilon = 6.0
    size = len(text_dp.vocabulary())
    trials = 4000
    word = "weather"
    assert word in text_dp.vocabulary()
    kept = 0
    for _ in range(trials):
        kept += text_dp.perturb(word, epsilon_token=epsilon)["text"] == word
    empirical_keep = kept / trials
    expected_keep = text_dp.retention_probability(epsilon, size)
    assert abs(empirical_keep - expected_keep) < 0.03, (
        f"keep rate {empirical_keep:.3f} vs expected {expected_keep:.3f}"
    )
    ratio = empirical_keep / ((1 - empirical_keep) / (size - 1))
    assert 0.7 * math.exp(epsilon) < ratio < 1.4 * math.exp(epsilon)


def test_release_is_randomised_not_deterministic():
    releases = {text_dp.perturb("weather", epsilon_token=0.5)["text"] for _ in range(30)}
    assert len(releases) > 5


def test_reported_numbers_are_the_ones_the_ui_shows():
    release = text_dp.perturb(
        "explain the recipe for a good summary", epsilon_token=8.0
    )
    assert release["vocabulary_size"] == len(text_dp.vocabulary())
    assert release["protected_tokens"] >= 3
    assert release["composed_epsilon"] == pytest.approx(
        8.0 * release["protected_tokens"]
    )
    assert release["retention_probability"] == pytest.approx(
        text_dp.retention_probability(8.0, release["vocabulary_size"])
    )
    # The label must name the mechanism and its strength, not overstate it.
    assert "k-RR" in release["mechanism"]
    assert "8" in release["mechanism"] and "LDP" in release["mechanism"]


def test_a_prompt_with_a_rare_word_is_never_returned_verbatim():
    # Structural, not statistical: a token outside the public output space cannot
    # survive, so any prompt containing one is always altered.
    original = "Explain Zigglethorpe to me"
    assert "zigglethorpe" not in text_dp.vocabulary()
    for _ in range(5):
        assert text_dp.perturb(original)["text"] != original


def test_the_output_space_is_public_not_user_derived():
    # Words are preservable only if they are already public. Nothing from a user
    # record can enter the vocabulary at runtime.
    words = text_dp.vocabulary()
    assert isinstance(words, tuple) and len(words) > 1000
    assert list(words) == sorted(words)
    assert all(word.isalpha() and word.islower() for word in words)


def test_epsilon_bounds_are_enforced():
    for bad in (0, -1, 21, 1000):
        with pytest.raises(ValueError):
            text_dp.perturb("anything", epsilon_token=bad)
