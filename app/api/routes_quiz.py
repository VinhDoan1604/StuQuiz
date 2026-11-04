"""
Quiz: sinh đề và chấm điểm phía server.

- /generate KHÔNG trả đáp án đúng cho client (QuizQuestionOut không có correct_choice).
- /answer: server tự chấm đúng/sai từ bản ghi QuizAttempt, ghi lại lựa chọn của user và
  cập nhật mastery/spaced-repetition trong CÙNG 1 transaction. Idempotent: gửi lại
  cùng attempt trả về kết quả đã lưu, không cộng điểm lần 2.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.agents.quiz_summary_agent import generate_quiz
from app.core.ownership import get_owned_space, resolve_owned_document_ids
from app.core.security import get_current_user
from app.db.session import get_db
from app.memory.mastery_tracker import record_quiz_result
from app.models.db_models import QuizAttempt, User
from app.models.schemas import QuizAnswerRequest, QuizAnswerResult, QuizGenerateRequest, QuizQuestionOut
from app.utils.ids import parse_uuid
from app.utils.time import utcnow

router = APIRouter()


@router.post("/generate", response_model=list[QuizQuestionOut])
async def generate(
    req: QuizGenerateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not req.document_ids:
        raise HTTPException(status_code=400, detail="Cần chọn ít nhất một tài liệu.")

    doc_ids = resolve_owned_document_ids(db, user, req.document_ids, strict=True)
    space_id = str(get_owned_space(db, user, req.space_id).id) if req.space_id else None

    return await generate_quiz(
        db=db,
        user_id=str(user.id),
        document_ids=doc_ids,
        num_questions=req.num_questions,
        prioritize_weak_topics=req.prioritize_weak_topics,
        topic_prompt=req.topic_prompt,
        space_id=space_id,
    )


@router.post("/answer", response_model=QuizAnswerResult)
def answer(
    req: QuizAnswerRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    attempt_id = parse_uuid(req.attempt_id)
    if attempt_id is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi.")

    attempt = (
        db.query(QuizAttempt)
        .filter(QuizAttempt.id == attempt_id, QuizAttempt.user_id == user.id)
        .with_for_update()  # chặn 2 request đồng thời cùng chấm 1 câu (Postgres)
        .first()
    )
    if attempt is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi.")

    if attempt.user_choice is None:
        if req.choice not in (attempt.choices or []):
            raise HTTPException(status_code=400, detail="Lựa chọn không thuộc câu hỏi này.")
        attempt.user_choice = req.choice
        attempt.is_correct = req.choice == attempt.correct_choice
        attempt.answered_at = utcnow()
        db.flush()
        # Cập nhật mastery cùng session; record_quiz_result commit cả attempt + mastery
        record_quiz_result(db, user_id=str(user.id), topic=attempt.topic or "", is_correct=attempt.is_correct)
        db.commit()  # topic rỗng => record_quiz_result bỏ qua, vẫn phải lưu attempt

    return QuizAnswerResult(
        attempt_id=str(attempt.id),
        is_correct=bool(attempt.is_correct),
        correct_choice=attempt.correct_choice,
        explanation=attempt.explanation or "",
    )
