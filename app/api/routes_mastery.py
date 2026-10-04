from pydantic import BaseModel
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.schemas import MasteryTopic
from app.models.db_models import MasteryRecord
from app.memory.mastery_tracker import get_weak_topics, record_quiz_result

router = APIRouter()


class QuizResultSubmit(BaseModel):
    user_id: str = "demo_user"
    topic: str
    is_correct: bool


@router.get("/weak-topics/{user_id}", response_model=list[MasteryTopic])
def weak_topics(user_id: str, db: Session = Depends(get_db)):
    return get_weak_topics(db, user_id)


@router.get("/all/{user_id}", response_model=list[MasteryTopic])
def all_topics(user_id: str, db: Session = Depends(get_db)):
    records = db.query(MasteryRecord).filter(MasteryRecord.user_id == user_id).all()
    return [
        MasteryTopic(
            topic=r.topic,
            mastery_score=r.mastery_score,
            streak=r.streak,
            last_reviewed_at=r.last_reviewed_at,
            next_review_due_at=r.next_review_due_at,
        )
        for r in records
    ]


@router.post("/record")
def record_result(req: QuizResultSubmit, db: Session = Depends(get_db)):
    record_quiz_result(db, user_id=req.user_id, topic=req.topic, is_correct=req.is_correct)
    return {"status": "ok", "user_id": req.user_id, "topic": req.topic, "is_correct": req.is_correct}
