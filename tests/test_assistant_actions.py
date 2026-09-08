from datetime import datetime

P = "/api/v1"


def command(client, text):
    result = client.post(
        P + "/assistant/command", json={"text": text, "mode": "Privacy"}
    )
    assert result.status_code == 200, result.text
    return result.json()


def create(client, kind="Reminders", title="Call Rahul"):
    r = client.post(
        P + "/entries/" + kind, json={"title": title, "detail": "2099-01-02T10:00"}
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_assistant_create_read_update_delete_requires_confirmation(client, signup):
    signup()
    for kind, noun in [("Calendar", "event"), ("Reminders", "reminder")]:
        draft = command(client, f'Create a {noun} "Design sync" tomorrow at 10 am')[
            "proposal"
        ]
        assert draft["operation"] == "create" and draft["title"] == "Design sync"
        assert datetime.fromisoformat(draft["detail"]).hour == 10
        assert client.get(P + "/entries/" + kind).json() == []
        item = create(client, kind, "Design sync")
        listing = command(client, f"List my {noun}s")
        assert f"#{item['id']}" in listing["text"] and listing["proposal"] is None
        planned = command(
            client, f"Reschedule {noun} #{item['id']} to 2099-01-04T11:30"
        )["proposal"]
        assert planned["operation"] == "update" and planned["id"] == item["id"]
        assert planned["detail"] == "2099-01-04T11:30"
        assert client.get(P + "/entries/" + kind).json()[0]["detail"] == item["detail"]
        updated = client.put(
            P + f"/entries/{kind}/{item['id']}",
            json={
                "title": planned["title"],
                "detail": planned["detail"],
                "expected_version": planned["version"],
            },
        )
        assert updated.status_code == 200, updated.text
        deletion = command(client, f'Please delete {noun} "Design sync"')["proposal"]
        assert deletion["operation"] == "delete"
        assert len(client.get(P + "/entries/" + kind).json()) == 1
        assert (
            client.delete(
                P + f"/entries/{kind}/{item['id']}",
                headers={"If-Match": deletion["version"]},
            ).status_code
            == 200
        )
        assert client.get(P + "/entries/" + kind).json() == []


def test_ambiguous_targets_and_owner_isolation(client, signup):
    signup()
    first = create(client)
    create(client)
    result = command(client, 'delete reminder "Call Rahul"')
    assert result["proposal"] is None and "More than one" in result["text"]
    assert command(client, "delete all reminders")["proposal"] is None
    assert len(client.get(P + "/entries/Reminders").json()) == 2
    signup()
    assert command(client, f"delete reminder #{first['id']}")["proposal"] is None
    assert command(client, f"delete #{first['id']}")["proposal"] is None
    assert client.delete(P + f"/entries/Reminders/{first['id']}").status_code == 404


def test_stale_update_delete_and_timezone_change(client, signup):
    user = signup()
    item = create(client)
    draft = command(client, f'update reminder #{item["id"]} title to "Call Dad"')[
        "proposal"
    ]
    assert draft["title"] == "Call Dad"
    url = P + f"/entries/Reminders/{item['id']}"
    assert (
        client.put(
            url, json={"title": "New title", "detail": item["detail"]}
        ).status_code
        == 200
    )
    assert (
        client.put(
            url,
            json={
                "title": draft["title"],
                "detail": draft["detail"],
                "expected_version": draft["version"],
            },
        ).status_code
        == 409
    )
    assert client.delete(url, headers={"If-Match": draft["version"]}).status_code == 409
    current = client.get(P + "/entries/Reminders").json()[0]
    preferences = user["preferences"]
    preferences["timezone"] = "UTC"
    assert client.put(P + "/settings", json=preferences).status_code == 200
    viewed = client.get(P + "/entries/Reminders").json()[0]
    assert viewed["detail"] == "2099-01-02T04:30"
    assert (
        client.delete(url, headers={"If-Match": current["version"]}).status_code == 409
    )


def test_mutation_titles_and_completion_are_not_summary_instructions(client, signup):
    signup()
    item = create(client, title="Summary review")
    result = command(client, f"Complete reminder #{item['id']}")
    assert result["proposal"]["done"] is True
    assert client.get(P + "/entries/Reminders").json()[0]["done"] is False
    result = command(client, 'Could you please delete reminder "Summary review"')
    assert (
        result["intent"] == "reminder" and result["proposal"]["operation"] == "delete"
    )
    result = command(client, f'Rename #{item["id"]} to "Renamed"')
    assert result["proposal"]["title"] == "Renamed"


def test_summarizer_identity_and_no_cloud(client, signup, monkeypatch):
    signup()

    async def forbidden(*args):
        raise AssertionError("Local summary contacted cloud")

    monkeypatch.setattr("app.main.cloud_answer", forbidden)
    result = command(
        client,
        "Summarize: The sky is blue. Rain is expected tomorrow. Bring an umbrella.",
    )
    assert result["summarization_engine"] == "Local extractive summariser (no LLM)"
    assert result["location"] == "local backend"
    runtime = client.get(P + "/runtime").json()
    assert runtime["cloud_model"] == "gpt-4o-mini"
