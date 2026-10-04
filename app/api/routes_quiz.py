from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.schemas import QuizGenerateRequest, QuizQuestion
from app.agents.quiz_summary_agent import generate_quiz

router = APIRouter()


@router.post("/generate", response_model=list[QuizQuestion])
async def generate(req: QuizGenerateRequest, db: Session = Depends(get_db)):
    user_id = req.user_id or "demo_user"
    return await generate_quiz(
        db=db,
        user_id=user_id,
        document_ids=req.document_ids,
        num_questions=req.num_questions,
        prioritize_weak_topics=req.prioritize_weak_topics,
        topic_prompt=req.topic_prompt,
        space_id=req.space_id,
    )
