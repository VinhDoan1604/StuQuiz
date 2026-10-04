import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.db_models import StudySpace, Document, ChatSession, ChatTurn, QuizAttempt, User

router = APIRouter()


class CreateSpaceRequest(BaseModel):
    title: str
    description: Optional[str] = None
    icon: Optional[str] = "book"
    user_id: Optional[str] = "demo_user"


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


def _resolve_user(user_id: Optional[str], db: Session) -> Optional[User]:
    if user_id:
        try:
            val = str(uuid.UUID(str(user_id)))
            user = db.query(User).filter(User.id == val).first()
            if user:
                return user
        except (ValueError, AttributeError):
            val = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(user_id)))
            user = db.query(User).filter(User.id == val).first()
            if user:
                return user
    return None


@router.get("", response_model=List[SpaceResponse])
def list_spaces(user_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Trả về danh sách tất cả các Bộ Học Tập của người dùng."""
    user = _resolve_user(user_id, db)
    if not user:
        return []

    spaces = (
        db.query(StudySpace)
        .filter(StudySpace.user_id == user.id)
        .order_by(StudySpace.created_at.desc())
        .all()
    )

    return [
        SpaceResponse(
            id=str(s.id),
            title=s.title,
            description=s.description,
            icon=s.icon or "book",
            user_id=str(s.user_id) if s.user_id else None,
            document_ids=s.document_ids or [],
            num_documents=len(s.document_ids or []),
            created_at=s.created_at.isoformat() if s.created_at else None,
        )
        for s in spaces
    ]


@router.patch("/{space_id}", response_model=SpaceResponse)
def update_space(space_id: str, req: UpdateSpaceRequest, db: Session = Depends(get_db)):
    """Đổi tên hoặc cập nhật thông tin Bộ học tập (Góp ý #1)."""
    space = db.query(StudySpace).filter(StudySpace.id == space_id).first()
    if not space:
        raise HTTPException(status_code=404, detail="Không tìm thấy Bộ học tập.")

    if req.title is not None and req.title.strip():
        space.title = req.title.strip()
    if req.description is not None:
        space.description = req.description.strip()
    if req.icon is not None:
        space.icon = req.icon

    db.commit()
    db.refresh(space)

    return SpaceResponse(
        id=str(space.id),
        title=space.title,
        description=space.description,
        icon=space.icon or "book",
        user_id=str(space.user_id) if space.user_id else None,
        document_ids=space.document_ids or [],
        num_documents=len(space.document_ids or []),
        created_at=space.created_at.isoformat() if space.created_at else None,
    )


@router.post("", response_model=SpaceResponse)
def create_space(req: CreateSpaceRequest, db: Session = Depends(get_db)):
    """Tạo một Bộ Học Tập mới."""
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Tên bộ học tập không được để trống.")

    user = _resolve_user(req.user_id, db)

    space = StudySpace(
        title=req.title.strip(),
        description=req.description.strip() if req.description else "",
        icon=req.icon or "book",
        user_id=user.id if user else None,
        document_ids=[],
    )
    db.add(space)
    db.commit()
    db.refresh(space)

    return SpaceResponse(
        id=str(space.id),
        title=space.title,
        description=space.description,
        icon=space.icon or "book",
        user_id=str(space.user_id) if space.user_id else None,
        document_ids=space.document_ids or [],
        num_documents=0,
        created_at=space.created_at.isoformat() if space.created_at else None,
    )


@router.get("/{space_id}")
def get_space_detail(space_id: str, db: Session = Depends(get_db)):
    """Lấy chi tiết Bộ học tập kèm danh sách các tài liệu thuộc bộ."""
    space = db.query(StudySpace).filter(StudySpace.id == space_id).first()
    if not space:
        raise HTTPException(status_code=404, detail="Không tìm thấy Bộ học tập.")

    doc_ids = space.document_ids or []
    documents = []
    if doc_ids:
        docs = db.query(Document).filter(Document.id.in_(doc_ids)).all()
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
def attach_document(space_id: str, req: AttachDocumentRequest, db: Session = Depends(get_db)):
    """Gắn một tài liệu đã nạp vào Bộ học tập này."""
    space = db.query(StudySpace).filter(StudySpace.id == space_id).first()
    if not space:
        raise HTTPException(status_code=404, detail="Không tìm thấy Bộ học tập.")

    current_ids = list(space.document_ids or [])
    if req.document_id not in current_ids:
        current_ids.append(req.document_id)
        space.document_ids = current_ids
        db.commit()

    return {"status": "ok", "document_ids": space.document_ids}


@router.delete("/{space_id}/documents/{doc_id}")
def detach_document(space_id: str, doc_id: str, db: Session = Depends(get_db)):
    """Gỡ tài liệu ra khỏi Bộ học tập."""
    space = db.query(StudySpace).filter(StudySpace.id == space_id).first()
    if not space:
        raise HTTPException(status_code=404, detail="Không tìm thấy Bộ học tập.")

    current_ids = list(space.document_ids or [])
    if doc_id in current_ids:
        current_ids.remove(doc_id)
        space.document_ids = current_ids
        db.commit()

    return {"status": "ok", "document_ids": space.document_ids}


@router.get("/{space_id}/chat-history")
def get_space_chat_history(space_id: str, db: Session = Depends(get_db)):
    """Lấy danh sách các phiên hội thoại kèm tin nhắn đã diễn ra trong Bộ học tập này."""
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.space_id == space_id)
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
        result.append({
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
        })
    return result


@router.get("/{space_id}/quiz-history")
def get_space_quiz_history(space_id: str, db: Session = Depends(get_db)):
    """Lấy danh sách các câu hỏi trắc nghiệm đã làm trong Bộ học tập này."""
    attempts = (
        db.query(QuizAttempt)
        .filter(QuizAttempt.space_id == space_id)
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
def delete_space(space_id: str, db: Session = Depends(get_db)):
    """Xóa một Bộ học tập."""
    space = db.query(StudySpace).filter(StudySpace.id == space_id).first()
    if not space:
        raise HTTPException(status_code=404, detail="Không tìm thấy Bộ học tập.")
    db.delete(space)
    db.commit()
    return {"status": "deleted", "space_id": space_id}
