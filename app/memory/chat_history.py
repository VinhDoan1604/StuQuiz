"""
CRUD lịch sử chat — thuần thao tác DB. Logic tóm tắt/ngân sách token nằm ở
context_window.py.

Mọi id truyền vào đây ĐÃ được tầng API xác thực (UUID hợp lệ, thuộc user hiện
tại — xem app/core/ownership.py), nên module này không tự chuẩn hoá hay tạo
user ngầm.
"""
from sqlalchemy.orm import Session

from app.models.db_models import ChatSession, ChatTurn

_TITLE_MAX_CHARS = 80


def get_or_create_session(
    db: Session, session_id: str, user_id: str, space_id: str | None = None, first_message: str | None = None
) -> ChatSession:
    session = db.get(ChatSession, session_id)
    if session is None:
        title = (first_message or "").strip().replace("\n", " ")[:_TITLE_MAX_CHARS] or None
        session = ChatSession(id=session_id, user_id=user_id, space_id=space_id, title=title)
        db.add(session)
        db.commit()
        db.refresh(session)
    elif space_id and not session.space_id:
        session.space_id = space_id
        db.commit()
    return session


def add_turn(db: Session, session_id: str, role: str, content: str,
             intent: str | None = None, reflection_trace: dict | None = None) -> ChatTurn:
    turn = ChatTurn(
        session_id=session_id, role=role, content=content,
        intent=intent, reflection_trace=reflection_trace,
    )
    db.add(turn)
    db.commit()
    db.refresh(turn)
    return turn


def get_recent_turns(db: Session, session_id: str, limit: int = 10) -> list[ChatTurn]:
    return (
        db.query(ChatTurn)
        .filter(ChatTurn.session_id == session_id)
        .order_by(ChatTurn.created_at.desc())
        .limit(limit)
        .all()
    )
