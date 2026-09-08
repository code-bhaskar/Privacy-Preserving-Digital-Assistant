import json
import pytest
from cryptography.exceptions import InvalidTag
from sqlalchemy import select, text
from app.database import engine, transaction, users, entries
from app.security import encrypt, decrypt, verify_password

P = "/api/v1"


def test_login_logout_cookie_and_revocation(client, signup):
    user = signup()
    cookie = client.cookies.get("ppda_session")
    assert client.get(P + "/auth/session").status_code == 200
    assert client.post(P + "/auth/logout").status_code == 200
    client.cookies.set("ppda_session", cookie)
    assert client.get(P + "/auth/session").status_code == 401
    response = client.post(
        P + "/auth/login",
        json={"email": user["email"], "password": "test-password-very-long"},
    )
    assert response.status_code == 200
    header = response.headers["set-cookie"].lower()
    assert "httponly" in header and "secure" in header and "samesite=lax" in header
    assert "password" not in response.json()


def test_unauthenticated_and_csrf(client, signup):
    assert client.get(P + "/entries/Notes").status_code == 401
    signup()
    client.headers["x-csrf-token"] = "wrong"
    assert (
        client.post(P + "/entries/Notes", json={"title": "secret"}).status_code == 403
    )
    assert client.get(P + "/entries/Notes").status_code == 200


def test_cross_origin_login_rejected(client):
    r = client.post(
        P + "/auth/login",
        json={"email": "other@example.com", "password": "bad"},
        headers={"Origin": "https://evil.example"},
    )
    assert r.status_code == 403


def test_password_hash_and_length(client, signup):
    user = signup()
    with engine.connect() as conn:
        stored = (
            conn.execute(select(users).where(users.c.id == user["id"])).mappings().one()
        )
    assert stored["password_hash"].startswith("$2b$")
    assert verify_password("test-password-very-long", stored["password_hash"])
    for password in ["short", "🙂" * 30]:
        r = client.post(
            P + "/auth/register",
            json={
                "email": "length@example.com",
                "name": "Length",
                "password": password,
            },
        )
        assert r.status_code == 422


def test_login_rate_limit_persistent(client):
    for _ in range(5):
        assert (
            client.post(
                P + "/auth/login",
                json={"email": "missing@example.com", "password": "wrong"},
            ).status_code
            == 401
        )
    assert (
        client.post(
            P + "/auth/login",
            json={"email": "missing@example.com", "password": "wrong"},
        ).status_code
        == 429
    )


def test_consent_denial_audited(client, signup):
    signup(consent=False)
    assert (
        client.post(P + "/assistant/command", json={"text": "hello"}).status_code == 403
    )
    assert (
        client.post(P + "/entries/Notes", json={"title": "private"}).status_code == 403
    )
    assert any(
        "CONSENT_BLOCKED" in x["action"] for x in client.get(P + "/audit").json()
    )


@pytest.mark.parametrize(
    "kind,detail",
    [
        ("Notes", "private note body"),
        ("Calendar", "2026-09-10T15:00"),
        ("Reminders", "2026-09-10T17:00"),
    ],
)
def test_encrypted_crud_and_owner_isolation(client, signup, kind, detail):
    user = signup()
    r = client.post(
        P + "/entries/" + kind, json={"title": "top secret title", "detail": detail}
    )
    assert r.status_code == 201, r.text
    item = r.json()
    with engine.connect() as conn:
        row = (
            conn.execute(select(entries).where(entries.c.id == item["id"]))
            .mappings()
            .one()
        )
    assert "top secret" not in row["ciphertext"] and detail not in row["ciphertext"]
    assert (
        decrypt(user["id"], "entry:" + kind, row["ciphertext"])["title"]
        == "top secret title"
    )
    assert (
        client.put(
            P + f"/entries/{kind}/{item['id']}",
            json={"title": "updated secret", "detail": detail},
        ).status_code
        == 200
    )
    assert client.get(P + "/entries/" + kind).json()[0]["title"] == "updated secret"
    signup()
    assert client.get(P + "/entries/" + kind).json() == []
    assert (
        client.put(
            P + f"/entries/{kind}/{item['id']}",
            json={"title": "stolen", "detail": detail},
        ).status_code
        == 404
    )
    assert client.delete(P + f"/entries/{kind}/{item['id']}").status_code == 404


def test_aad_tampering_and_nonce():
    one = encrypt(100, "entry:Notes", {"title": "test"})
    two = encrypt(100, "entry:Notes", {"title": "test"})
    assert one != two
    with pytest.raises(InvalidTag):
        decrypt(101, "entry:Notes", one)
    with pytest.raises(InvalidTag):
        decrypt(100, "entry:Calendar", one)


def test_audit_append_only_and_verification(client, signup):
    signup()
    client.post(P + "/entries/Notes", json={"title": "no plaintext in audit"})
    assert client.get(P + "/audit/verify").json()["valid"]
    assert "no plaintext" not in json.dumps(client.get(P + "/audit").json())
    for action in ["UPDATE audit_logs SET action='changed'", "DELETE FROM audit_logs"]:
        with pytest.raises(Exception):
            with transaction() as conn:
                conn.execute(text(action))


def test_password_change_revokes_all_sessions(client, signup):
    user = signup()
    cookie = client.cookies.get("ppda_session")
    assert (
        client.post(
            P + "/auth/password",
            json={
                "current_password": "test-password-very-long",
                "new_password": "new-very-long-password",
            },
        ).status_code
        == 200
    )
    client.cookies.set("ppda_session", cookie)
    assert client.get(P + "/auth/session").status_code == 401
    assert (
        client.post(
            P + "/auth/login",
            json={"email": user["email"], "password": "test-password-very-long"},
        ).status_code
        == 401
    )
    assert (
        client.post(
            P + "/auth/login",
            json={"email": user["email"], "password": "new-very-long-password"},
        ).status_code
        == 200
    )


def test_local_commands_never_call_cloud(client, signup, monkeypatch):
    signup()

    async def forbidden(*args, **kwargs):
        pytest.fail("Privacy mode attempted a cloud call")

    monkeypatch.setattr("app.main.cloud_answer", forbidden)
    for mode in ["Privacy", "Default"]:
        r = client.post(
            P + "/assistant/command",
            json={"text": "remind me to call Rahul tomorrow at 5 pm", "mode": mode},
        )
        assert r.status_code == 200, r.text
        assert r.json()["proposal"]["kind"] == "Reminders"
        assert r.json()["location"] == "local backend"
        assert r.json()["explanation"]
        assert (
            client.get(P + "/entries/Reminders").json() == []
        )  # confirmation is required
    r = client.post(
        P + "/assistant/command",
        json={
            "text": "Summarize: Apples are fruit. Apples contain fibre. Pears are fruit too. A train arrived.",
            "mode": "Privacy",
        },
    )
    assert r.status_code == 200 and r.json()["text"].startswith("• ")


def test_cloud_consent_configuration_and_no_history(client, signup, monkeypatch):
    user = signup()
    assert (
        client.post(
            P + "/assistant/command", json={"text": "hello", "mode": "Global"}
        ).status_code
        == 403
    )
    p = user["preferences"]
    p["cloud"] = True
    assert client.put(P + "/settings", json=p).status_code == 200
    assert (
        client.post(
            P + "/assistant/command", json={"text": "hello", "mode": "Global"}
        ).status_code
        == 503
    )
    seen = []

    async def provider(text):
        seen.append(text)
        return "Actual adapter response in mocked provider test"

    monkeypatch.setattr("app.main.cloud_answer", provider)
    r = client.post(
        P + "/assistant/command",
        json={"text": "only selected prompt", "mode": "Global"},
    )
    assert r.status_code == 200 and seen == ["only selected prompt"]
    assert "OpenAI" in r.json()["location"]


def test_settings_validation_and_profile(client, signup):
    user = signup()
    p = user["preferences"]
    p["epsilon"] = 0
    assert client.put(P + "/settings", json=p).status_code == 422
    p["epsilon"] = 2
    p["timezone"] = "Bad/Zone"
    assert client.put(P + "/settings", json=p).status_code == 422
    p["timezone"] = "Asia/Kolkata"
    assert client.put(P + "/settings", json=p).status_code == 200
    assert client.patch(P + "/profile", json={"name": "New name"}).status_code == 200
    assert client.get(P + "/auth/session").json()["name"] == "New name"


def test_summary_takes_priority_over_content_keywords(client, signup):
    signup()
    r = client.post(
        P + "/assistant/command",
        json={
            "text": "Summarize: The meeting is tomorrow. Remember the calendar. A reminder was sent.",
            "mode": "Privacy",
        },
    )
    assert r.json()["intent"] == "summary"
    assert r.json()["proposal"] is None


def test_large_body_rejected(client):
    r = client.post(
        P + "/auth/login",
        content=b"x" * 100001,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 413


def test_onnx_predictions_and_shape_guard():
    import numpy as np
    from app.local_model import seed_weights, probabilities
    from app.onnx_model import probabilities as onnx_probabilities

    weights = seed_weights()
    np.testing.assert_allclose(
        onnx_probabilities(weights, "schedule a meeting"),
        probabilities(weights, "schedule a meeting"),
        atol=1e-6,
    )
    with pytest.raises(ValueError):
        onnx_probabilities(np.zeros((2, 2)), "hello")


def test_default_keeps_sensitive_prompt_local(client, signup, monkeypatch):
    user = signup()
    p = user["preferences"]
    p["cloud"] = True
    client.put(P + "/settings", json=p)
    monkeypatch.setattr("app.main.settings.openai_api_key", "not-a-real-key")

    async def forbidden(*args, **kwargs):
        pytest.fail("Sensitive Default prompt attempted cloud transmission")

    monkeypatch.setattr("app.main.cloud_answer", forbidden)
    assert (
        client.post(
            P + "/assistant/command",
            json={"text": "my medical diagnosis is private", "mode": "Default"},
        ).json()["location"]
        == "local backend"
    )


def test_embedded_preview_cookie_policy(client, signup, monkeypatch):
    monkeypatch.setattr("app.main.settings.cookie_samesite", "none")
    monkeypatch.setattr("app.main.settings.cookie_partitioned", True)
    user = signup()
    r = client.post(
        P + "/auth/login",
        json={"email": user["email"], "password": "test-password-very-long"},
    )
    header = r.headers["set-cookie"].lower()
    assert "secure" in header and "httponly" in header
    assert "samesite=none" in header and "partitioned" in header
    client.headers["x-csrf-token"] = r.json()["csrf"]
    assert client.get(P + "/auth/session").status_code == 200
    r = client.post(P + "/auth/logout")
    assert "partitioned" in r.headers["set-cookie"].lower()
