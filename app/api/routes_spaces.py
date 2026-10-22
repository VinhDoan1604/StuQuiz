from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.ownership import get_owned_space, resolve_owned_document_ids
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.db_models import ChatSession, ChatTurn, Document, QuizAttempt, StudySpace, User

router = APIRouter()


class CreateSpaceRequest(BaseModel):
    title: str
    description: Optional[str] = None
    icon: Optional[str] = "book"


class UpdateSpaceRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None


class AttachDocumentRequest(BaseModel):
    document_id: str


class SpaceResponse(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    icon: Optional[str] = "book"
    user_id: Optional[str] = None
    document_ids: List[str] = []
    num_documents: int = 0
    created_at: Optional[str] = None


def _to_response(s: StudySpace) -> SpaceResponse:
    ids = list(s.document_ids or [])
    return SpaceResponse(
        id=str(s.id),
        title=s.title,
        description=s.description,
        icon=s.icon or "book",
        user_id=str(s.user_id) if s.user_id else None,
        document_ids=ids,
        num_documents=len(ids),
        created_at=s.created_at.isoformat() if s.created_at else None,
    )


@router.get("", response_model=List[SpaceResponse])
def list_spaces(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    spaces = (
        db.query(StudySpace)
        .filter(StudySpace.user_id == user.id)
        .order_by(StudySpace.created_at.desc())
        .all()
    )
    return [_to_response(s) for s in spaces]


@router.post("", response_model=SpaceResponse)
def create_space(
    req: CreateSpaceRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Tên bộ học tập không được để trống.")
    if len(title) > 200:
        raise HTTPException(status_code=400, detail="Tên bộ học tập quá dài (tối đa 200 ký tự).")

    space = StudySpace(
        title=title,
        description=(req.description or "").strip()[:1000],
        icon=req.icon or "book",
        user_id=user.id,
        document_ids=[],
    )
    db.add(space)
    db.commit()
    db.refresh(space)
    return _to_response(space)


@router.patch("/{space_id}", response_model=SpaceResponse)
def update_space(
    space_id: str,
    req: UpdateSpaceRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    space = get_owned_space(db, user, space_id)
    if req.title is not None and req.title.strip():
        space.title = req.title.strip()[:200]
    if req.description is not None:
        space.description = req.description.strip()[:1000]
    if req.icon is not None:
        space.icon = req.icon
    db.commit()
    db.refresh(space)
    return _to_response(space)


@router.get("/{space_id}")
def get_space_detail(space_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    space = get_owned_space(db, user, space_id)
    doc_ids = space.document_ids or []
    documents = []
    if doc_ids:
        docs = (
            db.query(Document)
            .filter(Document.id.in_(doc_ids), Document.owner_id == user.id)
            .all()
        )
        documents = [
            {
                "id": str(d.id),
                "filename": d.filename,
                "file_type": d.file_type,
                "num_chunks": d.num_chunks,
                "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
            }
            for d in docs
        ]
    return {
        "id": str(space.id),
        "title": space.title,
        "description": space.description,
        "icon": space.icon,
        "document_ids": doc_ids,
        "documents": documents,
        "created_at": space.created_at.isoformat() if space.created_at else None,
    }


@router.post("/{space_id}/documents")
def attach_document(
    space_id: str,
    req: AttachDocumentRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    space = get_owned_space(db, user, space_id)
    # strict=True: không cho gắn tài liệu của người khác vào bộ của mình
    (doc_id,) = resolve_owned_document_ids(db, user, [req.document_id], strict=True)

    current = list(space.document_ids or [])
    if doc_id not in current:
        current.append(doc_id)
        space.document_ids = current
        db.commit()
    return {"status": "ok", "document_ids": space.document_ids}


@router.delete("/{space_id}/documents/{doc_id}")
def detach_document(
    space_id: str, doc_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    space = get_owned_space(db, user, space_id)
    current = list(space.document_ids or [])
    if doc_id in current:
        current.remove(doc_id)
        space.document_ids = current
        db.commit()
    return {"status": "ok", "document_ids": space.document_ids}


@router.get("/{space_id}/chat-history")
def get_space_chat_history(
    space_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    space = get_owned_space(db, user, space_id)
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.space_id == space.id, ChatSession.user_id == user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    result = []
    for s in sessions:
        turns = (
            db.query(ChatTurn)
            .filter(ChatTurn.session_id == s.id)
            .order_by(ChatTurn.created_at.asc())
            .all()
        )
        if not turns:
            continue
        first_user_turn = next((t.content for t in turns if t.role == "user"), None)
        title = s.title or (first_user_turn[:60] if first_user_turn else "Cuộc hội thoại mới")
        result.append(
            {
                "session_id": str(s.id),
                "title": title,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "turns_count": len(turns),
                "turns": [
                    {
                        "id": str(t.id),
                        "role": t.role,
                        "content": t.content,
                        "intent": t.intent,
                        "reflection_trace": t.reflection_trace,
                        "created_at": t.created_at.isoformat() if t.created_at else None,
                    }
                    for t in turns
                ],
            }
        )
    return result


@router.get("/{space_id}/quiz-history")
def get_space_quiz_history(
    space_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    space = get_owned_space(db, user, space_id)
    attempts = (
        db.query(QuizAttempt)
        .filter(QuizAttempt.space_id == space.id, QuizAttempt.user_id == user.id, QuizAttempt.user_choice.isnot(None))
        .order_by(QuizAttempt.created_at.desc())
        .all()
    )
    return [
        {
            "id": str(a.id),
            "topic": a.topic,
            "question": a.question,
            "choices": a.choices,
            "correct_choice": a.correct_choice,
            "user_choice": a.user_choice,
            "is_correct": a.is_correct,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in attempts
    ]


@router.delete("/{space_id}")
def delete_space(space_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    space = get_owned_space(db, user, space_id)
    db.delete(space)
    db.commit()
    return {"status": "deleted", "space_id": space_id}
