from alembic import context
from app.database import engine, metadata

with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=metadata)
    with context.begin_transaction():
        context.run_migrations()
