"""
app/core/ownership.py

Mọi kiểm tra "tài nguyên này có thuộc về user hiện tại không" nằm MỘT chỗ.
Trả 404 (không phải 403) khi không thuộc về user để không lộ sự tồn tại
của tài nguyên người khác.
"""
from __future__ import annotations

import uuid

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.db_models import ChatSession, Document, StudySpace, User


def _valid_uuid(value: str | None) -> str | None:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return None


def get_owned_space(db: Session, user: User, space_id: str) -> StudySpace:
    sid = _valid_uuid(space_id)
    space = (
        db.query(StudySpace)
        .filter(StudySpace.id == sid, StudySpace.user_id == user.id)
        .first()
        if sid
        else None
    )
    if space is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy Bộ học tập.")
    return space


def get_owned_document(db: Session, user: User, document_id: str) -> Document:
    did = _valid_uuid(document_id)
    doc = (
        db.query(Document)
        .filter(Document.id == did, Document.owner_id == user.id)
        .first()
        if did
        else None
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    return doc


def resolve_owned_document_ids(
    db: Session,
    user: User,
    document_ids: list[str] | None,
    strict: bool = True,
) -> list[str]:
    """
    - document_ids=None  -> toàn bộ tài liệu của user.
    - strict=True        -> có id không thuộc user => 404 (dùng cho thao tác ghi / quiz).
    - strict=False       -> lặng lẽ bỏ id không hợp lệ (dùng cho chat: id cũ đã xoá
                            không nên làm hỏng cả cuộc hội thoại).
    """
    if document_ids is None:
        rows = db.query(Document.id).filter(Document.owner_id == user.id).all()
        return [str(r[0]) for r in rows]

    wanted = {v for v in (_valid_uuid(d) for d in document_ids) if v}
    if not wanted:
        if strict and document_ids:
            raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
        return []

    rows = (
        db.query(Document.id)
        .filter(Document.owner_id == user.id, Document.id.in_(list(wanted)))
        .all()
    )
    owned = {str(r[0]) for r in rows}
    if strict and owned != wanted:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    return sorted(owned)


def scope_session_id(raw_session_id: str, user: User) -> str:
    """
    Client có thể gửi session_id tuỳ ý (vd 'session_abc'). Nếu để nguyên, hai user
    chọn trùng chuỗi sẽ va chạm cùng 1 ChatSession. Id không phải UUID được gắn
    thêm user.id trước khi băm; UUID hợp lệ giữ nguyên (rồi kiểm tra chủ sở hữu).
    """
    valid = _valid_uuid(raw_session_id)
    if valid:
        return valid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user.id}:{raw_session_id}"))


def assert_session_owner(db: Session, session_id: str, user: User) -> None:
    session = db.get(ChatSession, session_id)
    if session is not None and str(session.user_id) != str(user.id):
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên hội thoại.")