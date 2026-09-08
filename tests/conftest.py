import base64
import os
from pathlib import Path
import tempfile

# Never exercise the user's actual workspace or secrets.
TEMP = tempfile.TemporaryDirectory(prefix="ppda-tests-")
os.environ.update(
    DATABASE_URL="sqlite:///" + str(Path(TEMP.name) / "tests.db"),
    JWT_SECRET="test-only-" + "x" * 48,
    AES_MASTER_KEY=base64.b64encode(b"t" * 32).decode(),
    PIPELINE_ENABLED="false",
    COOKIE_SECURE="true",
    COOKIE_SAMESITE="lax",
    COOKIE_PARTITIONED="false",
    OPENAI_API_KEY="",
    LOCAL_SUMMARY_MODEL="extractive",
)
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from app.main import app
from app.database import transaction, attempts


@pytest.fixture(scope="session")
def client():
    command.upgrade(Config("alembic.ini"), "head")
    with TestClient(app, base_url="https://testserver") as c:
        yield c


@pytest.fixture(autouse=True)
def reset_auth_attempts(client):
    client.cookies.clear()
    client.headers.pop("x-csrf-token", None)
    with transaction() as conn:
        conn.execute(attempts.delete())


@pytest.fixture
def signup(client):
    import uuid

    def create(consent=True, email=None):
        client.cookies.clear()
        client.headers.pop("x-csrf-token", None)
        response = client.post(
            "/api/v1/auth/register",
            json={
                "name": "Test User",
                "email": email or f"{uuid.uuid4().hex}@example.com",
                "password": "test-password-very-long",
                "local_consent": consent,
            },
        )
        assert response.status_code == 201, response.text
        data = response.json()
        client.headers["x-csrf-token"] = data["csrf"]
        return data

    return create
