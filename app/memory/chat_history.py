# STATUS: READY
"""
CRUD đơn giản cho lịch sử chat (mục 5.11, phần lưu DB) — thuần thao tác
DB, không có logic "thông minh" nào (đó là context_window.py, STUB).
"""
import uuid
from sqlalchemy.orm import Session
from app.models.db_models import User, StudySpace, ChatSession, ChatTurn


def _safe_uuid(val: str | None) -> str:
    if not val:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(str(val)))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(val)))


def get_or_create_session(db: Session, session_id: str, user_id: str, space_id: str | None = None) -> ChatSession:
    clean_session_id = _safe_uuid(session_id)
    clean_user_id = _safe_uuid(user_id)
    clean_space_id = _safe_uuid(space_id) if space_id else None

    # Đảm bảo space_id hợp lệ nếu được truyền vào
    if clean_space_id:
        space_exists = db.query(StudySpace).filter(StudySpace.id == clean_space_id).first()
        if not space_exists:
            clean_space_id = None

    # Đảm bảo User tồn tại để không vi phạm FK
    user = db.get(User, clean_user_id)
    if not user:
        user = User(
            id=clean_user_id,
            email=f"{clean_user_id[:8]}@studyrag.local",
            full_name="Người Dùng StudyRAG",
        )
        db.add(user)
        try:
            db.commit()
        except Exception:
            db.rollback()

    session = db.get(ChatSession, clean_session_id)
    if session is None:
        session = ChatSession(id=clean_session_id, user_id=clean_user_id, space_id=clean_space_id)
        db.add(session)
        db.commit()
        db.refresh(session)
    elif clean_space_id and not session.space_id:
        session.space_id = clean_space_id
        db.commit()
    return session


def add_turn(db: Session, session_id: str, role: str, content: str,
             intent: str | None = None, reflection_trace: dict | None = None) -> ChatTurn:
    clean_session_id = _safe_uuid(session_id)
    turn = ChatTurn(
        session_id=clean_session_id, role=role, content=content,
        intent=intent, reflection_trace=reflection_trace,
    )
    db.add(turn)
    db.commit()
    db.refresh(turn)
    return turn


def get_recent_turns(db: Session, session_id: str, limit: int = 10) -> list[ChatTurn]:
    clean_session_id = _safe_uuid(session_id)
    return (
        db.query(ChatTurn)
        .filter(ChatTurn.session_id == clean_session_id)
        .order_by(ChatTurn.created_at.desc())
        .limit(limit)
        .all()
    )
