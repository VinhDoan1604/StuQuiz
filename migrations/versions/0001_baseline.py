"""baseline schema (tương đương DB tạo bởi init_db.py + migrate_part3.sql cũ)

Revision ID: 0001_baseline
Revises:
Create Date: 2026-10-30

DB đã tồn tại từ trước (tạo bằng create_all): chạy `alembic stamp 0001_baseline`
rồi `alembic upgrade head` thay vì upgrade từ đầu.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=False)


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("email", sa.String(), nullable=False, unique=True),
        sa.Column("full_name", sa.String(), nullable=True),
        sa.Column("password_hash", sa.String(), nullable=True),
        sa.Column("auth_provider", sa.String(), nullable=True),
        sa.Column("google_id", sa.String(), nullable=True),
        sa.Column("is_verified", sa.Boolean(), nullable=True),
        sa.Column("verification_code", sa.String(), nullable=True),
        sa.Column("verification_code_expires_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("verification_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_table(
        "study_spaces",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("user_id", UUID, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("icon", sa.String(), nullable=True),
        sa.Column("document_ids", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "documents",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("file_type", sa.String(), nullable=True),
        sa.Column("subject", sa.String(), nullable=True),
        sa.Column("uploaded_at", sa.DateTime(), nullable=True),
        sa.Column("num_chunks", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_table(
        "chat_sessions",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("user_id", UUID, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("space_id", UUID, sa.ForeignKey("study_spaces.id"), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("rolling_summary", sa.Text(), nullable=True),
        sa.Column("summarized_turn_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_table(
        "chat_turns",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("session_id", UUID, sa.ForeignKey("chat_sessions.id"), nullable=True),
        sa.Column("role", sa.String(), nullable=True),
        sa.Column("intent", sa.String(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("reflection_trace", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "quiz_attempts",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("user_id", UUID, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("space_id", UUID, sa.ForeignKey("study_spaces.id"), nullable=True),
        sa.Column("topic", sa.String(), nullable=True),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("choices", sa.JSON(), nullable=True),
        sa.Column("correct_choice", sa.String(), nullable=True),
        sa.Column("user_choice", sa.String(), nullable=True),
        sa.Column("is_correct", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("difficulty", sa.String(), nullable=True),
        sa.Column("answered_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "mastery_records",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("user_id", UUID, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("topic", sa.String(), nullable=True),
        sa.Column("correct_count", sa.Integer(), nullable=True),
        sa.Column("incorrect_count", sa.Integer(), nullable=True),
        sa.Column("streak", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("mastery_score", sa.Integer(), nullable=True),
        sa.Column("last_reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("next_review_due_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "parent_chunks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("section", sa.String(), nullable=True),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
    )
    op.create_index("ix_parent_chunks_doc_id", "parent_chunks", ["doc_id"])


def downgrade() -> None:
    op.drop_index("ix_parent_chunks_doc_id", table_name="parent_chunks")
    for table in (
        "parent_chunks", "mastery_records", "quiz_attempts", "chat_turns",
        "chat_sessions", "documents", "study_spaces", "users",
    ):
        op.drop_table(table)
