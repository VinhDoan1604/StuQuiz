# STATUS: READY
"""
SQLAlchemy models cho Postgres. Đây là phần dữ liệu quan hệ:
users, documents, chat history, mastery tracking, quiz results.

Vector embedding KHÔNG lưu ở đây — nằm trong Qdrant
(app/indexing/qdrant_store.py), Postgres chỉ giữ metadata + payload liên
kết qua `chunk_id`.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, DateTime, ForeignKey, Text, Integer, Boolean, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.session import Base


def gen_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    email = Column(String, unique=True, nullable=False)
    full_name = Column(String, nullable=True)
    password_hash = Column(String, nullable=True)
    auth_provider = Column(String, default="local")  # "local" | "google"
    google_id = Column(String, nullable=True)
    is_verified = Column(Boolean, default=False)
    verification_code = Column(String, nullable=True)
    verification_code_expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="owner")
    sessions = relationship("ChatSession", back_populates="user")
    spaces = relationship("StudySpace", back_populates="user")


class StudySpace(Base):
    """
    Bộ học tập chuyên biệt (Study Space / Knowledge Notebook).
    Mỗi không gian gắn liền với một tập hợp tài liệu nhất định,
    có lịch sử chat và các bài quiz riêng biệt.
    """
    __tablename__ = "study_spaces"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"))
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    icon = Column(String, default="book")
    document_ids = Column(JSON, default=list)  # list[str] các Document ID thuộc bộ này
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="spaces")
    sessions = relationship("ChatSession", back_populates="space", cascade="all, delete-orphan")
    quiz_attempts = relationship("QuizAttempt", back_populates="space", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    owner_id = Column(UUID(as_uuid=False), ForeignKey("users.id"))
    filename = Column(String, nullable=False)
    file_type = Column(String)          # pdf | docx | pptx | note
    subject = Column(String, nullable=True)   # môn học, dùng để filter metadata
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    num_chunks = Column(Integer, default=0)
    status = Column(String, default="ready")          # "processing" | "ready" | "failed"
    error_message = Column(Text, nullable=True)

    owner = relationship("User", back_populates="documents")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"))
    space_id = Column(UUID(as_uuid=False), ForeignKey("study_spaces.id"), nullable=True)
    title = Column(String, default="Cuộc trò chuyện mới")
    created_at = Column(DateTime, default=datetime.utcnow)
    # Bản tóm tắt lịch sử dài hạn do Memory Manager sinh ra định kỳ
    # (xem app/memory/context_window.py) — thay cho việc nhồi full history.
    rolling_summary = Column(Text, nullable=True)
    # Số turn đã được gộp vào rolling_summary (phục vụ incremental summarization)
    summarized_turn_count = Column(Integer, default=0, nullable=False)

    user = relationship("User", back_populates="sessions")
    space = relationship("StudySpace", back_populates="sessions")
    turns = relationship("ChatTurn", back_populates="session", cascade="all, delete-orphan")


class ChatTurn(Base):
    __tablename__ = "chat_turns"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    session_id = Column(UUID(as_uuid=False), ForeignKey("chat_sessions.id"))
    role = Column(String)                # user | assistant
    intent = Column(String, nullable=True)   # qa | teaching | quiz | summary
    content = Column(Text)
    # Trace của Reflection Agent cho lượt này, hiển thị ra UI (mục 3 tài liệu):
    # {"context_grade": ..., "answer_grade": ..., "rewrites": [...]}
    reflection_trace = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="turns")


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"))
    space_id = Column(UUID(as_uuid=False), ForeignKey("study_spaces.id"), nullable=True)
    topic = Column(String)               # gắn với 1 khái niệm/section cụ thể
    question = Column(Text)
    choices = Column(JSON)               # ["A. ...", "B. ...", ...]
    correct_choice = Column(String)
    user_choice = Column(String, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    space = relationship("StudySpace", back_populates="quiz_attempts")


class MasteryRecord(Base):
    """
    Trạng thái nắm vững theo từng topic cho từng user — đây là bảng lõi
    của tính năng khác biệt với NotebookLM (mastery tracking / spaced
    review). Logic cập nhật điểm mastery nằm ở
    app/memory/mastery_tracker.py (STUB — cần đầu tư kỹ).
    """
    __tablename__ = "mastery_records"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"))
    topic = Column(String)
    correct_count = Column(Integer, default=0)
    incorrect_count = Column(Integer, default=0)
    streak = Column(Integer, default=0, nullable=False)  # Số lần trả lời đúng liên tiếp (phục vụ Spaced Repetition)
    mastery_score = Column(Integer, default=0)   # 0-100, thuật toán ở stub
    last_reviewed_at = Column(DateTime, nullable=True)
    next_review_due_at = Column(DateTime, nullable=True)  # spaced repetition
