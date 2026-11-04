"""index cho khoá ngoại + unique (user_id, topic) cho mastery_records

Revision ID: 0002_indexes_constraints
Revises: 0001_baseline
Create Date: 2026-10-30
"""
from alembic import op

revision = "0002_indexes_constraints"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None

_INDEXES = [
    ("ix_study_spaces_user_id", "study_spaces", "user_id"),
    ("ix_documents_owner_id", "documents", "owner_id"),
    ("ix_chat_sessions_user_id", "chat_sessions", "user_id"),
    ("ix_chat_sessions_space_id", "chat_sessions", "space_id"),
    ("ix_chat_turns_session_id", "chat_turns", "session_id"),
    ("ix_quiz_attempts_user_id", "quiz_attempts", "user_id"),
    ("ix_quiz_attempts_space_id", "quiz_attempts", "space_id"),
    ("ix_mastery_records_user_id", "mastery_records", "user_id"),
]


def upgrade() -> None:
    for name, table, column in _INDEXES:
        op.create_index(name, table, [column])
    with op.batch_alter_table("mastery_records") as batch:
        batch.create_unique_constraint("uq_mastery_user_topic", ["user_id", "topic"])


def downgrade() -> None:
    with op.batch_alter_table("mastery_records") as batch:
        batch.drop_constraint("uq_mastery_user_topic", type_="unique")
    for name, table, _ in reversed(_INDEXES):
        op.drop_index(name, table_name=table)
