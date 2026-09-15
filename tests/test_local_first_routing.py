"""Local-first routing, capability escalation, DP release accounting and
tolerant input reading (spoken numbers and misspellings)."""

from datetime import datetime

import pytest

P = "/api/v1"


def command(client, text, mode="Default"):
    response = client.post(
        P + "/assistant/command", json={"text": text, "mode": mode}
    )
    assert response.status_code == 200, response.text
    return response.json()


def enable_cloud(user, client, monkeypatch):
    """Consent + a configured (mocked) provider, so escalation is permitted."""
    preferences = user["preferences"]
    preferences["cloud"] = True
    assert client.put(P + "/settings", json=preferences).status_code == 200
    monkeypatch.setattr("app.main.settings.openai_api_key", "not-a-real-key")


def mock_provider(monkeypatch, seen=None, epsilon=None):
    """Deterministic release + provider, so the wire contract is testable."""
    seen = [] if seen is None else seen

    async def provider(text):
        seen.append(text)
        return "Global model answer"

    release = {
        "text": "PERTURBED",
        "mechanism": "Token-level 10-LDP k-RR over 100 public words + redaction",
        "epsilon_token": epsilon or 10.0,
        "retention_probability": 0.9,
        "vocabulary_size": 100,
        "protected_tokens": 3,
        "composed_epsilon": 30.0,
        "redactions": [{"type": "email", "count": 1}],
        "perturbed": [{"from": "selected", "to": "[redacted]"}],
    }
    monkeypatch.setattr("app.main.cloud_answer", provider)
    monkeypatch.setattr("app.main.text_dp.perturb", lambda text, epsilon: release)
    return seen


def forbid_cloud(monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("A request the local stack can handle reached the cloud")

    monkeypatch.setattr("app.main.cloud_answer", forbidden)


# --- local-first -------------------------------------------------------------


def test_supported_tasks_never_escalate(client, signup, monkeypatch):
    user = signup()
    enable_cloud(user, client, monkeypatch)
    forbid_cloud(monkeypatch)
    for text in [
        "remind me to call Rahul tomorrow at 5 pm",
        "add an event to my calendar next monday at 4 pm",
        "note: buy milk and eggs",
        "list my reminders",
        "Summarize: Apples are fruit. Apples contain fibre. Pears are fruit too.",
    ]:
        result = command(client, text)
        assert result["route"] == "local", text
        assert result["location"] == "local backend", text
        assert result["router"]["capable"] is True, text
        assert result["privacy"] is None, text


def test_out_of_scope_request_escalates_with_dp_release(client, signup, monkeypatch):
    user = signup()
    enable_cloud(user, client, monkeypatch)
    seen = mock_provider(monkeypatch)
    result = command(client, "what is the capital of France")
    assert result["route"] == "global" and "OpenAI" in result["location"]
    assert result["router"]["capable"] is False
    assert result["router"]["outcome"] == "escalated"
    # Only the perturbed release is transmitted; the raw prompt never is.
    assert seen == ["PERTURBED"]
    assert result["privacy"]["applied"] is True
    assert result["privacy"]["sent_prompt"] == "PERTURBED"
    assert result["privacy"]["composition_bound"] == 30.0
    # The release is charged to the account's lifetime budget.
    assert result["privacy"]["epsilon_charged"] == 0.25
    assert client.get(P + "/learning/status").json()["epsilon_spent"] == pytest.approx(
        0.25
    )


def test_privacy_mode_refuses_escalation(client, signup, monkeypatch):
    user = signup()
    enable_cloud(user, client, monkeypatch)
    forbid_cloud(monkeypatch)
    result = command(client, "explain how a compiler works", mode="Privacy")
    assert result["route"] == "local" and result["router"]["outcome"] == "blocked"
    assert "Privacy mode" in result["router"]["policy"]
    assert client.get(P + "/learning/status").json()["epsilon_spent"] == 0


def test_missing_consent_keeps_the_request_local(client, signup, monkeypatch):
    signup()
    monkeypatch.setattr("app.main.settings.openai_api_key", "not-a-real-key")
    forbid_cloud(monkeypatch)
    result = command(client, "translate this paragraph into spanish")
    assert result["route"] == "local" and result["router"]["outcome"] == "blocked"
    assert "consent" in result["router"]["policy"]
    # Explicit Global mode must fail loudly instead of pretending to be local.
    assert (
        client.post(
            P + "/assistant/command",
            json={"text": "translate this paragraph into spanish", "mode": "Global"},
        ).status_code
        == 403
    )


def test_sensitive_prompt_is_not_escalated_automatically(client, signup, monkeypatch):
    user = signup()
    enable_cloud(user, client, monkeypatch)
    forbid_cloud(monkeypatch)
    result = command(client, "my medical diagnosis is private, what should I do")
    assert result["route"] == "local" and result["router"]["outcome"] == "blocked"
    assert "sensitive" in result["router"]["policy"].lower()


def test_exhausted_budget_blocks_further_escalation(client, signup, monkeypatch):
    user = signup()
    preferences = user["preferences"]
    preferences["cloud"] = True
    preferences["epsilon"] = 1  # four 0.25 releases
    assert client.put(P + "/settings", json=preferences).status_code == 200
    monkeypatch.setattr("app.main.settings.openai_api_key", "not-a-real-key")
    mock_provider(monkeypatch)
    for attempt in range(4):
        assert command(client, "what is the capital of France")["route"] == "global"
    assert client.get(P + "/learning/status").json()["epsilon_spent"] == pytest.approx(
        1.0
    )
    blocked = command(client, "what is the capital of France")
    assert blocked["route"] == "local" and blocked["router"]["outcome"] == "blocked"
    assert "budget" in blocked["router"]["policy"]


def test_escalation_never_writes_workspace_records(client, signup, monkeypatch):
    user = signup()
    enable_cloud(user, client, monkeypatch)
    mock_provider(monkeypatch)
    command(client, "who wrote the novel pride and prejudice")
    assert client.get(P + "/entries/Reminders").json() == []
    assert client.get(P + "/entries/Calendar").json() == []


# --- tolerant reading --------------------------------------------------------


def test_spoken_numbers_become_real_times(client, signup):
    signup()
    draft = command(client, "remind me to call Rahul tomorrow at seven thirty pm")[
        "proposal"
    ]
    when = datetime.fromisoformat(draft["detail"])
    assert (when.hour, when.minute) == (19, 30)
    assert draft["title"].strip().lower().endswith("rahul")
    assert (
        datetime.fromisoformat(
            command(client, "remind me to take medicine at half past eight")[
                "proposal"
            ]["detail"]
        ).minute
        == 30
    )
    assert (
        datetime.fromisoformat(
            command(client, "remind me to stretch in twenty minutes")["proposal"][
                "detail"
            ]
        )
        > datetime.now()
    )


def test_misspellings_are_understood_and_reported(client, signup):
    signup()
    result = command(client, "set a remindar for tomorow at five")
    assert result["proposal"]["kind"] == "Reminders"
    assert result["proposal"]["detail"]
    corrections = {c["from"]: c["to"] for c in result["normalized"]["corrections"]}
    assert corrections["remindar"] == "reminder"
    assert corrections["tomorow"] == "tomorrow"
    assert any("remindar" in note for note in result["normalized"]["notes"])
    assert (
        command(client, "add an event to my calender next monday at four")[
            "proposal"
        ]["kind"]
        == "Calendar"
    )


def test_proper_nouns_are_not_corrected(client, signup):
    signup()
    result = command(client, "remind me to call Rahul at seven")
    assert "Rahul" in result["proposal"]["title"]


# --- reviewable change list --------------------------------------------------


def test_draft_reports_the_changes_and_a_correctness_check(client, signup):
    signup()
    result = command(client, "remind me to call Rahul tomorrow at seven thirty pm")
    fields = {change["field"]: change["after"] for change in result["changes"]}
    assert fields["Action"] == "Create reminders"
    assert "Rahul" in fields["Title"]
    assert "19:30" in fields["When"]
    assert result["check"]["ok"] is True
    assert any("seven thirty pm" in note for note in result["check"]["notes"])
    # Nothing is written until the review is confirmed.
    assert client.get(P + "/entries/Reminders").json() == []


def test_check_flags_a_draft_without_a_time(client, signup):
    signup()
    result = command(client, "remind me to call Rahul")
    assert result["proposal"]["detail"] == ""
    assert result["check"]["ok"] is False
    assert any("No time" in warning for warning in result["check"]["warnings"])


def test_update_draft_shows_before_and_after(client, signup):
    signup()
    created = client.post(
        P + "/entries/Reminders",
        json={"title": "Call Rahul", "detail": "2099-01-02T10:00"},
    ).json()
    result = command(client, f'Rename #{created["id"]} to "Call Dad"')
    rows = {
        change["field"]: (change.get("before"), change.get("after"))
        for change in result["changes"]
    }
    assert rows["Title"] == ("Call Rahul", "Call Dad")
    assert result["check"]["ok"] is True
    assert client.get(P + "/entries/Reminders").json()[0]["title"] == "Call Rahul"


def test_instruction_words_do_not_become_the_title(client, signup):
    signup()
    result = command(client, "remind me to call Rahul tomorrow at 5 pm")
    assert result["proposal"]["title"] == "call Rahul"
    result = command(client, "add an event to my calender next monday at four")
    assert result["proposal"]["kind"] == "Calendar"
    when = datetime.fromisoformat(result["proposal"]["detail"])
    assert (when.weekday(), when.hour) == (0, 4)  # the next Monday, 04:00
    assert result["proposal"]["title"] == "Calendar event"


def test_relative_durations_keep_their_time_and_ask_for_a_subject(client, signup):
    signup()
    result = command(client, "remind me in twenty minutes to stretch")
    when = datetime.fromisoformat(result["proposal"]["detail"])
    assert when > datetime.now()
    # No subject in the request, so the draft says so instead of titling it "to".
    assert result["proposal"]["title"] == "Reminder"
    assert result["check"]["ok"] is False
    assert any("placeholder" in warning for warning in result["check"]["warnings"])


def test_quoted_titles_win_over_the_instruction_prefix(client, signup):
    signup()
    draft = command(client, 'Create a reminder "Design sync" tomorrow at 10 am')[
        "proposal"
    ]
    assert draft["title"] == "Design sync"
    assert datetime.fromisoformat(draft["detail"]).hour == 10


def test_real_dp_mechanism_reaches_the_provider_and_the_ui(client, signup, monkeypatch):
    """No mock on the mechanism: the release built by fl/text_dp is what is sent,
    and privacy_report must agree with its keys or this raises at runtime."""
    user = signup()
    enable_cloud(user, client, monkeypatch)
    sent = []

    async def provider(text):
        sent.append(text)
        return "Global answer"

    monkeypatch.setattr("app.main.cloud_answer", provider)
    # Out of scope, and carrying a phone number but no guard keyword, so the
    # automatic escalation path is allowed and redaction is exercised.
    original = "explain how photosynthesis works and call 98765 43210"
    result = command(client, original)
    assert result["route"] == "global"
    privacy = result["privacy"]
    assert len(sent) == 1 and sent[0] == privacy["sent_prompt"]
    assert sent[0] != original
    assert "98765" not in sent[0] and "43210" not in sent[0]
    # Every field the UI renders is really present and numeric where claimed.
    assert privacy["applied"] is True
    assert privacy["vocabulary_size"] > 1000
    assert 0 < privacy["retention_probability"] < 1
    assert privacy["protected_tokens"] >= 1
    assert privacy["composition_bound"] == pytest.approx(
        privacy["epsilon_token"] * privacy["protected_tokens"]
    )
    assert any(entry["type"] == "phone-or-id" for entry in privacy["redactions"])
    assert privacy["epsilon_charged"] == 0.25
    assert "per token" in privacy["notice"]
    # The ledger charge is the real mechanism's, not a mocked constant.
    assert client.get(P + "/learning/status").json()["epsilon_spent"] == pytest.approx(
        0.25
    )


def test_an_email_in_the_prompt_blocks_automatic_escalation(client, signup, monkeypatch):
    user = signup()
    enable_cloud(user, client, monkeypatch)
    seen = mock_provider(monkeypatch)
    result = command(client, "what is the capital of France, ask a@b.com")
    assert result["route"] == "local" and seen == []
    assert "sensitive" in result["router"]["policy"].lower()
    assert client.get(P + "/learning/status").json()["epsilon_spent"] == 0
