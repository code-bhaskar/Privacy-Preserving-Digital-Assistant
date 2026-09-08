"""Initial local-first application schema and append-only protections."""

from alembic import op
from app.database import metadata

revision = "0001"
down_revision = None


def upgrade():
    metadata.create_all(
        op.get_bind(),
        tables=[
            metadata.tables[name]
            for name in (
                "users",
                "sessions",
                "login_attempts",
                "entries",
                "audit_logs",
                "training_examples",
                "federated_rounds",
                "privacy_ledger",
                "model_versions",
            )
        ],
    )
    for table in ["audit_logs", "privacy_ledger"]:
        for action in ["UPDATE", "DELETE"]:
            op.execute(
                f"CREATE TRIGGER {table}_no_{action.lower()} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT, 'append-only'); END"
            )


def downgrade():
    raise RuntimeError(
        "Destructive downgrade disabled: audit and budget history must be preserved."
    )
