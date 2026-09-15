"""Escalation releases need a privacy-ledger entry without a federated round.

An out-of-capability prompt that is released to the global model after
`fl.text_dp` is a separate kind of release from a training round, so it must not
be forced to reference `federated_rounds`. Existing rows keep their round_id.
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"

TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS privacy_ledger_no_{action} BEFORE {action}
ON privacy_ledger BEGIN SELECT RAISE(ABORT, 'append-only'); END
"""


def upgrade():
    # SQLite cannot drop a NOT NULL constraint in place; batch mode rebuilds the
    # table and copies the existing rows.
    with op.batch_alter_table(
        "privacy_ledger",
        recreate="always",
        copy_from=sa.Table(
            "privacy_ledger",
            sa.MetaData(),
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("round_id", sa.Integer(), nullable=True),
            sa.Column("epsilon", sa.Float(), nullable=False),
            sa.Column("delta", sa.Float(), nullable=False),
            # Foreign-key enforcement is enabled for this database, so the
            # rebuilt table has to carry the constraint over.
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.ForeignKeyConstraint(["round_id"], ["federated_rounds.id"]),
        ),
    ) as batch:
        batch.alter_column("round_id", existing_type=sa.Integer(), nullable=True)
    op.execute("CREATE INDEX IF NOT EXISTS ix_privacy_ledger_user_id ON privacy_ledger (user_id)")
    # Rebuilding a table drops the triggers attached to it, so the append-only
    # protection has to be reinstated in the same migration.
    for action in ("UPDATE", "DELETE"):
        op.execute(TRIGGERS.format(action=action.lower()).strip())


def downgrade():
    raise RuntimeError(
        "Destructive downgrade disabled: audit and budget history must be preserved."
    )
