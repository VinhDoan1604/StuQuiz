"""
app/memory/mastery_tracker.py

Mastery tracking & Spaced Review (mục 2.8, 5.11) — điểm khác biệt cốt lõi
của StudyRAG giúp học sinh/sinh viên củng cố kiến thức theo chu kỳ não bộ.

Thuật toán Spaced Repetition (SuperMemo/Leitner streak schedule):
- streak = 1 -> ôn tập sau 1 ngày.
- streak = 2 -> ôn tập sau 3 ngày.
- streak = 3 -> ôn tập sau 7 ngày.
- streak >= 4 -> ôn tập sau 14 ngày.
- Khi trả lời SAI: streak reset về 0, lên lịch ôn lại sau 1 ngày.

Thuật toán mastery_score (0-100, khởi tạo mặc định 0):
- Đúng: mastery_score += 10 (clamp 100).
- Sai:  mastery_score -= 15 (clamp 0). Giảm mạnh hơn tăng (bất đối xứng)
  để ưu tiên tính chắc chắn của kiến thức.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.models.db_models import MasteryRecord
from app.models.schemas import MasteryTopic

logger = logging.getLogger(__name__)

DEFAULT_INITIAL_SCORE = 0
SCORE_INCREMENT_CORRECT = 10
SCORE_PENALTY_INCORRECT = 15
SCORE_MIN = 0
SCORE_MAX = 100

# Lịch ôn tập theo streak: 1 -> 1 ngày, 2 -> 3 ngày, 3 -> 7 ngày, >=4 -> 14 ngày.
_SPACED_REPETITION_SCHEDULE_DAYS = [1, 3, 7, 14]
WEAK_SCORE_THRESHOLD = 60


def _next_interval_days(streak: int) -> int:
    """Tính số ngày ôn tập tiếp theo dựa vào chuỗi đúng liên tiếp (streak)."""
    index = min(max(streak - 1, 0), len(_SPACED_REPETITION_SCHEDULE_DAYS) - 1)
    return _SPACED_REPETITION_SCHEDULE_DAYS[index]


def _get_or_create_record(db: Session, user_id: str, topic: str) -> MasteryRecord:
    record = (
        db.query(MasteryRecord)
        .filter(MasteryRecord.user_id == user_id, MasteryRecord.topic == topic)
        .first()
    )
    if record is None:
        record = MasteryRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            topic=topic,
            correct_count=0,
            incorrect_count=0,
            streak=0,
            mastery_score=DEFAULT_INITIAL_SCORE,
            last_reviewed_at=None,
            next_review_due_at=None,
        )
        db.add(record)
    return record


def record_quiz_result(db: Session, user_id: str, topic: str, is_correct: bool) -> None:
    """
    Cập nhật MasteryRecord sau mỗi lần sinh viên trả lời 1 câu quiz thuộc `topic`.
    - Tăng điểm khi đúng, phạt nặng hơn khi sai (bất đối xứng).
    - Cập nhật số lần đúng/sai tích lũy.
    - Cập nhật streak và ngày ôn tập tiếp theo theo Spaced Repetition.
    """
    if not topic or not topic.strip():
        logger.warning("record_quiz_result: topic rỗng, bỏ qua (user_id=%s).", user_id)
        return

    record = _get_or_create_record(db, user_id, topic)
    now = datetime.utcnow()

    if is_correct:
        record.correct_count = (record.correct_count or 0) + 1
        record.mastery_score = min(SCORE_MAX, (record.mastery_score or 0) + SCORE_INCREMENT_CORRECT)
        record.streak = (record.streak or 0) + 1
        interval_days = _next_interval_days(record.streak)
    else:
        record.incorrect_count = (record.incorrect_count or 0) + 1
        record.mastery_score = max(SCORE_MIN, (record.mastery_score or 0) - SCORE_PENALTY_INCORRECT)
        record.streak = 0
        interval_days = 1

    record.last_reviewed_at = now
    record.next_review_due_at = now + timedelta(days=interval_days)

    db.add(record)
    db.commit()

    logger.info(
        "record_quiz_result(user=%s, topic='%s', correct=%s): score=%d, streak=%d, next_review=%s",
        user_id, topic, is_correct, record.mastery_score, record.streak, record.next_review_due_at,
    )


def get_weak_topics(db: Session, user_id: str, limit: int = 5) -> list[MasteryTopic]:
    """
    Trả về tối đa `limit` topic của `user_id` cần chú ý nhất — vì điểm
    thấp (< WEAK_SCORE_THRESHOLD) HOẶC đã quá hạn ôn tập
    (next_review_due_at <= now). Ưu tiên topic quá hạn ôn tập lâu nhất
    trước, sau đó mới tới điểm thấp nhất.
    """
    now = datetime.utcnow()

    candidates = (
        db.query(MasteryRecord)
        .filter(
            MasteryRecord.user_id == user_id,
            (MasteryRecord.mastery_score < WEAK_SCORE_THRESHOLD)
            | (MasteryRecord.next_review_due_at <= now),
        )
        .all()
    )

    def sort_key(record: MasteryRecord) -> tuple[int, float, int]:
        is_overdue = bool(record.next_review_due_at and record.next_review_due_at <= now)
        overdue_seconds = (now - record.next_review_due_at).total_seconds() if is_overdue else 0.0
        return (0 if is_overdue else 1, -overdue_seconds, record.mastery_score)

    candidates.sort(key=sort_key)

    return [
        MasteryTopic(
            topic=r.topic,
            mastery_score=r.mastery_score,
            streak=r.streak,
            last_reviewed_at=r.last_reviewed_at,
            next_review_due_at=r.next_review_due_at,
        )
        for r in candidates[:limit]
    ]