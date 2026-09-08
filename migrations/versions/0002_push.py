"""Encrypted browser subscriptions and a durable notification outbox."""

from alembic import op
from app.database import push_subscriptions, push_outbox

revision = "0002"
down_revision = "0001"


def upgrade():
    push_subscriptions.create(op.get_bind(), checkfirst=True)
    push_outbox.create(op.get_bind(), checkfirst=True)
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_push_delivery ON push_outbox (entry_id, subscription_id, due_at)"
    )


def downgrade():
    push_outbox.drop(op.get_bind())
    push_subscriptions.drop(op.get_bind())
