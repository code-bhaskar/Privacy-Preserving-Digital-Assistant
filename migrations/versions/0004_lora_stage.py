"""Low-rank federated stage: round provenance and adapter lineage.

Two additive changes, both idempotent because revision 0001 creates its tables
from the live metadata object (so a fresh database already has the column and
the migration must not try to add it twice):

1. `federated_rounds.stage` records which stage produced a round, "softmax" or
   "lora". Pre-existing rows were all produced by the full-matrix stage, so they
   are backfilled with that value by the column default.
2. `lora_adapters` stores one row per completed low-rank round: the aggregated
   (already noised) adapter, the base model it was trained against, the model
   version it was merged into, and the gate score. Rejected rounds are kept,
   because the privacy budget for them was spent and must stay auditable.

No column is dropped and no row is rewritten, so the append-only audit and
ledger protections are untouched.
"""

from alembic import op
import sqlalchemy as sa
from app.database import metadata

revision = "0004"
down_revision = "0003"


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("federated_rounds")}
    if "stage" not in columns:
        op.add_column(
            "federated_rounds",
            sa.Column(
                "stage",
                sa.String(),
                nullable=False,
                server_default=sa.text("'softmax'"),
            ),
        )
    if "lora_adapters" not in set(inspector.get_table_names()):
        metadata.create_all(bind, tables=[metadata.tables["lora_adapters"]])
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_lora_adapters_round_id ON lora_adapters (round_id)"
    )


def downgrade():
    raise RuntimeError(
        "Destructive downgrade disabled: adapter lineage and round provenance are audit history."
    )
