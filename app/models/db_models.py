"""
SQLAlchemy models (dữ liệu quan hệ): users, study spaces, documents, chat
history, quiz attempts, mastery tracking, parent chunks.

Vector embedding KHÔNG lưu ở đây — nằm trong Qdrant
(app/indexing/qdrant_store.py); child chunk trong Qdrant trỏ về parent chunk
ở bảng `parent_chunks` qua `parent_id`.

Mọi thay đổi schema phải đi kèm 1 migration Alembic (thư mục migrations/).
"""
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.session import Base
from app.utils.ids import new_id
from app.utils.time import utcnow


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=False), primary_key=True, default=new_id)
    email = Column(String, unique=True, nullable=False)
    full_name = Column(String, nullable=True)
    password_hash = Column(String, nullable=True)
    auth_provider = Column(String, default="local")  # "local" | "google" | "demo"
    google_id = Column(String, nullable=True)
    is_verified = Column(Boolean, default=False)
    verification_code = Column(String, nullable=True)   # HMAC của OTP, không lưu mã gốc
    verification_code_expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    verification_attempts = Column(Integer, default=0, nullable=False)

    documents = relationship("Document", back_populates="owner")
    sessions = relationship("ChatSession", back_populates="user")
    spaces = relationship("StudySpace", back_populates="user")


class StudySpace(Base):
    """
    Không gian học tập (Study Space / Knowledge Notebook): 1 tập tài liệu,
    lịch sử chat và các lượt quiz riêng.
    """
    __tablename__ = "study_spaces"
    id = Column(UUID(as_uuid=False), primary_key=True, default=new_id)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), index=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    icon = Column(String, default="book")
    document_ids = Column(JSON, default=list)  # list[str] các Document ID thuộc không gian này
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="spaces")
    sessions = relationship("ChatSession", back_populates="space", cascade="all, delete-orphan")
    quiz_attempts = relationship("QuizAttempt", back_populates="space", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=False), primary_key=True, default=new_id)
    owner_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), index=True)
    filename = Column(String, nullable=False)
    file_type = Column(String)                 # pdf | docx | pptx | txt | md
    subject = Column(String, nullable=True)    # môn học, dùng để filter metadata
    uploaded_at = Column(DateTime, default=utcnow)
    num_chunks = Column(Integer, default=0)
    status = Column(String, default="processing")   # "processing" | "ready" | "failed"
    error_message = Column(Text, nullable=True)

    owner = relationship("User", back_populates="documents")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(UUID(as_uuid=False), primary_key=True, default=new_id)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), index=True)
    space_id = Column(UUID(as_uuid=False), ForeignKey("study_spaces.id"), nullable=True, index=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    # Bản tóm tắt lịch sử dài hạn (app/memory/context_window.py) — thay cho nhồi full history.
    rolling_summary = Column(Text, nullable=True)
    # Số turn đã được gộp vào rolling_summary (incremental summarization)
    summarized_turn_count = Column(Integer, default=0, nullable=False)

    user = relationship("User", back_populates="sessions")
    space = relationship("StudySpace", back_populates="sessions")
    turns = relationship("ChatTurn", back_populates="session", cascade="all, delete-orphan")


class ChatTurn(Base):
    __tablename__ = "chat_turns"
    id = Column(UUID(as_uuid=False), primary_key=True, default=new_id)
    session_id = Column(UUID(as_uuid=False), ForeignKey("chat_sessions.id"), index=True)
    role = Column(String)                    # user | assistant
    intent = Column(String, nullable=True)   # qa | teaching | quiz | summary
    content = Column(Text)
    # Reflection trace + citations của lượt trả lời (hiển thị lại khi mở lịch sử)
    reflection_trace = Column(JSON, nullable=True)
    feedback = Column(Integer, nullable=True)   # đánh giá của người dùng: 1 = 👍, -1 = 👎
    created_at = Column(DateTime, default=utcnow)

    session = relationship("ChatSession", back_populates="turns")


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"
    id = Column(UUID(as_uuid=False), primary_key=True, default=new_id)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), index=True)
    space_id = Column(UUID(as_uuid=False), ForeignKey("study_spaces.id"), nullable=True, index=True)
    topic = Column(String)                   # gắn với 1 khái niệm/section cụ thể
    question = Column(Text)
    choices = Column(JSON)                   # ["A. ...", "B. ...", ...]
    correct_choice = Column(String)
    user_choice = Column(String, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    explanation = Column(Text, nullable=True)
    difficulty = Column(String, nullable=True)
    answered_at = Column(DateTime, nullable=True)

    space = relationship("StudySpace", back_populates="quiz_attempts")


class MasteryRecord(Base):
    """
    Mức nắm vững theo từng topic của từng user (mastery tracking / spaced
    review). Logic cập nhật nằm ở app/memory/mastery_tracker.py.
    """
    __tablename__ = "mastery_records"
    __table_args__ = (UniqueConstraint("user_id", "topic", name="uq_mastery_user_topic"),)

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_id)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), index=True)
    topic = Column(String)
    correct_count = Column(Integer, default=0)
    incorrect_count = Column(Integer, default=0)
    streak = Column(Integer, default=0, nullable=False)  # số lần đúng liên tiếp
    mastery_score = Column(Integer, default=0)           # 0-100
    last_reviewed_at = Column(DateTime, nullable=True)
    next_review_due_at = Column(DateTime, nullable=True)


class ParentChunk(Base):
    """Section/parent chunk: không nhúng vector; dùng mở rộng ngữ cảnh sau rerank."""
    __tablename__ = "parent_chunks"
    id = Column(String, primary_key=True)            # = Chunk.id của parent
    doc_id = Column(String, index=True, nullable=False)
    text = Column(Text, nullable=False)
    section = Column(String, nullable=True)
    page = Column(Integer, nullable=True)
    token_count = Column(Integer, default=0)
