"""
app/core/quota.py

Hạn mức theo ngày (UTC), đếm thẳng từ các bảng đã có nên không cần thêm bảng.
Rate limit (app/core/rate_limit.py) chặn dồn dập theo phút và theo IP; quota chặn tổng
chi phí theo user và toàn hệ thống. Giá trị 0 = không giới hạn.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.models.db_models import ChatSession, ChatTurn, Document, QuizAttempt, User
from app.utils.time import utcnow

_TOO_MANY = 429


def _today_start() -> datetime:
    return utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def _deny(detail: str) -> None:
    raise HTTPException(status_code=_TOO_MANY, detail=detail)


def is_demo(user: User) -> bool:
    return user.auth_provider == "demo"


def enforce_chat_quota(db: Session, user: User) -> None:
    today = _today_start()
    questions = db.query(ChatTurn).filter(ChatTurn.role == "user", ChatTurn.created_at >= today)
    if settings.quota_global_chat_per_day and questions.count() >= settings.quota_global_chat_per_day:
        _deny("Hệ thống đã hết hạn mức hỏi đáp của hôm nay. Vui lòng quay lại vào ngày mai.")

    limit = settings.quota_demo_chat_per_day if is_demo(user) else settings.quota_chat_per_day
    if limit:
        used = questions.join(ChatSession, ChatTurn.session_id == ChatSession.id).filter(ChatSession.user_id == user.id).count()
        if used >= limit:
            _deny(f"Bạn đã dùng hết {limit} câu hỏi của hôm nay. Vui lòng quay lại vào ngày mai.")


def enforce_upload_quota(db: Session, user: User) -> None:
    if is_demo(user) and not settings.demo_allow_upload:
        raise HTTPException(status_code=403, detail="Tài khoản dùng thử không tải được tài liệu lên.")
    owned = db.query(Document).filter(Document.owner_id == user.id)
    if settings.quota_documents_per_user and owned.count() >= settings.quota_documents_per_user:
        _deny(f"Bạn đã đạt giới hạn {settings.quota_documents_per_user} tài liệu. Hãy xoá bớt tài liệu cũ.")
    if settings.quota_uploads_per_day and owned.filter(Document.uploaded_at >= _today_start()).count() >= settings.quota_uploads_per_day:
        _deny(f"Bạn đã tải lên {settings.quota_uploads_per_day} tài liệu hôm nay. Vui lòng quay lại vào ngày mai.")


def enforce_quiz_quota(db: Session, user: User, num_questions: int) -> None:
    limit = settings.quota_quiz_questions_per_day
    if not limit:
        return
    used = db.query(QuizAttempt).filter(QuizAttempt.user_id == user.id, QuizAttempt.created_at >= _today_start()).count()
    if used + num_questions > limit:
        _deny(f"Bạn đã dùng hết {limit} câu trắc nghiệm của hôm nay. Vui lòng quay lại vào ngày mai.")
