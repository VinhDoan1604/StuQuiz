"""
routes_mastery v2: chỉ ĐỌC. Endpoint POST /record (client tự báo is_correct) đã bị gỡ —
mastery chỉ được cập nhật bởi POST /quiz/answer sau khi server tự chấm.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db
from app.memory.mastery_tracker import get_weak_topics
from app.models.db_models import MasteryRecord, User
from app.models.schemas import MasteryTopic

router = APIRouter()


def _to_topic(r: MasteryRecord) -> MasteryTopic:
    return MasteryTopic(
        topic=r.topic,
        mastery_score=r.mastery_score,
        streak=r.streak,
        last_reviewed_at=r.last_reviewed_at,
        next_review_due_at=r.next_review_due_at,
    )


@router.get("/weak-topics", response_model=list[MasteryTopic])
def weak_topics(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return get_weak_topics(db, str(user.id))


@router.get("/all", response_model=list[MasteryTopic])
def all_topics(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return [_to_topic(r) for r in db.query(MasteryRecord).filter(MasteryRecord.user_id == user.id).all()]
