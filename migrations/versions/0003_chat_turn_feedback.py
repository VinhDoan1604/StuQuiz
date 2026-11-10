"""thêm đánh giá 👍/👎 của người dùng cho từng lượt trả lời

Revision ID: 0003_chat_turn_feedback
Revises: 0002_indexes_constraints
Create Date: 2026-11-10
"""
import sqlalchemy as sa
from alembic import op

revision = "0003_chat_turn_feedback"
down_revision = "0002_indexes_constraints"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_turns", sa.Column("feedback", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("chat_turns") as batch:
        batch.drop_column("feedback")
