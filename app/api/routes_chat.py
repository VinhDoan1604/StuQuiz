from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.ownership import (
    assert_session_owner,
    get_owned_space,
    resolve_owned_document_ids,
    scope_session_id,
)
from app.core.security import get_current_user
from app.db.session import get_db
from app.graph.build_graph import run_graph, stream_graph
from app.models.db_models import User
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter()


def _prepare(req: ChatRequest, user: User, db: Session) -> dict:
    """
    Mọi định danh do client gửi (session, space, document) đều được kiểm tra thuộc
    về user HIỆN TẠI trước khi vào graph. Danh sách document_ids đưa xuống tầng
    retrieval LUÔN là tập đã xác thực (không bao giờ None => không có truy vấn toàn cục).
    """
    session_key = scope_session_id(req.session_id, user)
    assert_session_owner(db, session_key, user)

    space = get_owned_space(db, user, req.space_id) if req.space_id else None

    if req.document_ids:
        doc_ids = resolve_owned_document_ids(db, user, req.document_ids, strict=False)
    elif space is not None and space.document_ids:
        doc_ids = resolve_owned_document_ids(db, user, list(space.document_ids), strict=False)
    elif space is not None:
        doc_ids = []
    else:
        doc_ids = resolve_owned_document_ids(db, user, None)

    return {
        "session_id": session_key,
        "user_message": req.message,
        "document_ids": doc_ids,
        "space_id": str(space.id) if space else None,
        "user_id": str(user.id),
    }


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    return await run_graph(**_prepare(req, user, db))


@router.post("/stream")
async def chat_stream(
    req: ChatRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """SSE: bắn tiến trình từng node (Orchestrator -> Retrieval -> Critic -> ...) về UI."""
    params = _prepare(req, user, db)
    return StreamingResponse(
        stream_graph(**params),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
