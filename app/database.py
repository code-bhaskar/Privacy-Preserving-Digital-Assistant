"""Small, transactionally serialized SQLite store with versioned Alembic schema."""

from contextlib import contextmanager
import hashlib
import json
import time
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Text,
    Boolean,
    Float,
    ForeignKey,
    select,
    text,
    event,
)
from .config import settings
from .security import keyed

metadata = MetaData()
users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String, unique=True, nullable=False),
    Column("name", String, nullable=False),
    Column("password_hash", String, nullable=False),
    Column("preferences", Text, nullable=False),
    Column("created_at", Float, nullable=False),
)
sessions = Table(
    "sessions",
    metadata,
    Column("sid", String, primary_key=True),
    Column("user_id", ForeignKey("users.id"), nullable=False),
    Column("expires", Float, nullable=False),
    Column("revoked", Boolean, nullable=False, default=False),
)
attempts = Table(
    "login_attempts",
    metadata,
    Column("key", String, primary_key=True),
    Column("times", Text, nullable=False),
)
entries = Table(
    "entries",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", ForeignKey("users.id"), nullable=False, index=True),
    Column("kind", String, nullable=False),
    Column("ciphertext", Text, nullable=False),
    Column("due_at", String),
    Column("done", Boolean, nullable=False, default=False),
    Column("fired", Boolean, nullable=False, default=False),
)
audits = Table(
    "audit_logs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, nullable=False, index=True),
    Column("action", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("prev_hash", String, nullable=False),
    Column("integrity_hash", String, nullable=False),
    Column("signature", String, nullable=False),
)
examples = Table(
    "training_examples",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", ForeignKey("users.id"), nullable=False, index=True),
    Column("ciphertext", Text, nullable=False),
    Column("used", Boolean, nullable=False, default=False),
)
rounds = Table(
    "federated_rounds",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("status", String, nullable=False),
    Column("participants", Text, nullable=False),
    Column("created_at", Float, nullable=False),
    Column("detail", Text, nullable=False),
)
ledger = Table(
    "privacy_ledger",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", ForeignKey("users.id"), nullable=False, index=True),
    Column("round_id", ForeignKey("federated_rounds.id"), nullable=False),
    Column("epsilon", Float, nullable=False),
    Column("delta", Float, nullable=False),
)
models = Table(
    "model_versions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("weights", Text, nullable=False),
    Column("created_at", Float, nullable=False),
    Column("score", Float, nullable=False),
    Column("active", Boolean, nullable=False),
    Column("round_id", Integer),
)

if not settings.database_url.startswith("sqlite:"):
    raise RuntimeError(
        "This release supports SQLite only. PostgreSQL requires a tested port of transaction locking and triggers."
    )
engine = create_engine(
    settings.database_url, connect_args={"check_same_thread": False, "timeout": 30}
)


@event.listens_for(engine, "connect")
def configure(dbapi, _):
    dbapi.execute("PRAGMA foreign_keys=ON")
    dbapi.execute("PRAGMA busy_timeout=30000")
    dbapi.execute("PRAGMA journal_mode=WAL")


@contextmanager
def transaction():
    # Across processes, not just a Python mutex. Serialises audit tails, budget reservations and auth writes.
    with engine.connect() as conn:
        conn.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


def check_migrations():
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from .config import ROOT

    config = Config(str(ROOT / "alembic.ini"))
    head = ScriptDirectory.from_config(config).get_current_head()
    try:
        with engine.connect() as conn:
            revision = conn.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar()
    except Exception as exc:
        raise RuntimeError("Run: .venv/bin/alembic upgrade head") from exc
    if revision != head:
        raise RuntimeError(
            "Database migration required: .venv/bin/alembic upgrade head"
        )


def record(conn, uid: int, action: str):
    previous = (
        conn.execute(
            select(audits.c.integrity_hash)
            .where(audits.c.user_id == uid)
            .order_by(audits.c.id.desc())
            .limit(1)
        ).scalar()
        or "GENESIS"
    )
    stamp = str(time.time_ns())
    payload = json.dumps([previous, uid, action, stamp], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    conn.execute(
        audits.insert().values(
            user_id=uid,
            action=action,
            created_at=stamp,
            prev_hash=previous,
            integrity_hash=digest,
            signature=keyed("audit:" + digest),
        )
    )


push_subscriptions = Table(
    "push_subscriptions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", ForeignKey("users.id"), nullable=False, index=True),
    Column("endpoint_hash", String, nullable=False, unique=True),
    Column("ciphertext", Text, nullable=False),
    Column("created_at", Float, nullable=False),
)
push_outbox = Table(
    "push_outbox",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("entry_id", ForeignKey("entries.id", ondelete="CASCADE"), nullable=False),
    Column(
        "subscription_id",
        ForeignKey("push_subscriptions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("due_at", String, nullable=False),
    Column("status", String, nullable=False),
    Column("attempts", Integer, nullable=False, default=0),
    Column("next_attempt", Float, nullable=False),
    Column("expires_at", Float, nullable=False),
)
