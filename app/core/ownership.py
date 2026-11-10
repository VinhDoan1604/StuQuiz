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

from app.config import settings
from app.models.db_models import ChatSession, Document, StudySpace, User
from app.utils.ids import parse_uuid as _valid_uuid


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


def public_demo_document_ids(db: Session) -> list[str]:
    """Tài liệu công khai dùng chung cho tài khoản dùng thử (scripts/seed_public_demo.py)."""
    rows = (
        db.query(Document.id)
        .join(User, Document.owner_id == User.id)
        .filter(User.email == settings.demo_owner_email, Document.status == "ready")
        .order_by(Document.filename)
        .all()
    )
    return [str(r[0]) for r in rows]


def _readable_owner_filter(db: Session, user: User):
    """Quyền ĐỌC: tài liệu của chính user; riêng tài khoản dùng thử được đọc thêm bộ tài liệu demo.
    Quyền ghi/xoá vẫn đi qua get_owned_document nên tài liệu demo luôn chỉ đọc."""
    if user.auth_provider != "demo":
        return Document.owner_id == user.id
    return Document.owner_id.in_(
        [user.id, *[r[0] for r in db.query(User.id).filter(User.email == settings.demo_owner_email).all()]]
    )


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
        rows = db.query(Document.id).filter(_readable_owner_filter(db, user)).all()
        return [str(r[0]) for r in rows]

    wanted = {v for v in (_valid_uuid(d) for d in document_ids) if v}
    if not wanted:
        if strict and document_ids:
            raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
        return []

    rows = (
        db.query(Document.id)
        .filter(_readable_owner_filter(db, user), Document.id.in_(list(wanted)))
        .all()
    )
    owned = {str(r[0]) for r in rows}
    if strict and owned != wanted:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    return sorted(owned)


def scope_session_id(raw_session_id: str, user: User) -> str:
    """
    Client nên gửi UUID (crypto.randomUUID). Chuỗi tuỳ ý khác (vd 'session_abc')
    vẫn được chấp nhận nhưng gắn thêm user.id trước khi băm để hai user chọn trùng
    chuỗi không va chạm cùng 1 ChatSession. UUID hợp lệ giữ nguyên (rồi kiểm tra
    chủ sở hữu bằng assert_session_owner).
    """
    valid = _valid_uuid(raw_session_id)
    if valid:
        return valid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user.id}:{raw_session_id}"))


def assert_session_owner(db: Session, session_id: str, user: User) -> None:
    session = db.get(ChatSession, session_id)
    if session is not None and str(session.user_id) != str(user.id):
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên hội thoại.")
