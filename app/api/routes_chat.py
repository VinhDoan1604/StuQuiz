from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.core.ownership import (
    assert_session_owner,
    get_owned_space,
    resolve_owned_document_ids,
    scope_session_id,
)
from app.core.rate_limit import chat_limiter
from app.core.security import get_current_user
from app.db.session import get_db
from app.graph.build_graph import run_graph, stream_graph
from app.llm.providers import ChatModelOption, chat_model_options, find_model_option
from app.llm.router import reset_model_override, set_model_override
from app.models.db_models import User
from app.models.schemas import ChatModelItem, ChatModelsResponse, ChatRequest, ChatResponse

router = APIRouter(dependencies=[Depends(chat_limiter)])


def _resolve_model(model_id: str | None) -> ChatModelOption | None:
    """Chỉ chấp nhận model nằm trong LLM_CHAT_MODELS (client không tự chọn model tuỳ ý)."""
    if not model_id:
        return None
    option = find_model_option(model_id)
    if option is None:
        raise HTTPException(status_code=400, detail=f"Model không hợp lệ: '{model_id}'.")
    return option


@router.get("/models", response_model=ChatModelsResponse)
async def list_chat_models(user: User = Depends(get_current_user)):
    """Các model user được chọn khi chat (rỗng => UI ẩn bộ chọn, dùng model mặc định)."""
    return ChatModelsResponse(
        models=[ChatModelItem(id=o.id, provider=o.provider, model=o.model, label=o.label) for o in chat_model_options()]
    )


def _prepare(req: ChatRequest, user: User, db: Session) -> dict:
    """
    Mọi định danh do client gửi (session, space, document) đều được kiểm tra thuộc
    về user HIỆN TẠI trước khi vào graph. Danh sách document_ids đưa xuống tầng
    retrieval LUÔN là tập đã xác thực (không bao giờ None => không có truy vấn toàn cục).

    Phạm vi tài liệu:
    - Có document_ids  -> đúng các tài liệu đó (bỏ qua id không thuộc user).
    - Có space_id      -> các tài liệu của space (space rỗng => không có tài liệu).
    - Không có cả hai  -> toàn bộ tài liệu của user.
    """
    if len(req.message) > settings.max_chat_message_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Tin nhắn quá dài (tối đa {settings.max_chat_message_chars} ký tự).",
        )

    session_key = scope_session_id(req.session_id, user)
    assert_session_owner(db, session_key, user)

    space = get_owned_space(db, user, req.space_id) if req.space_id else None

    if req.document_ids:
        doc_ids = resolve_owned_document_ids(db, user, req.document_ids, strict=False)
    elif space is not None:
        doc_ids = resolve_owned_document_ids(db, user, list(space.document_ids or []), strict=False)
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
    option = _resolve_model(req.model)
    params = _prepare(req, user, db)
    token = set_model_override(option.provider if option else None, option.model if option else None)
    try:
        return await run_graph(**params)
    finally:
        reset_model_override(token)


@router.post("/stream")
async def chat_stream(
    req: ChatRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """SSE: bắn tiến trình từng node (Orchestrator -> Retrieval -> Critic -> ...) về UI."""
    option = _resolve_model(req.model)
    params = _prepare(req, user, db)
    return StreamingResponse(
        _stream_with_model(option, params),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_with_model(option: ChatModelOption | None, params: dict) -> AsyncIterator[str]:
    # Đặt override BÊN TRONG generator: StreamingResponse chạy generator ở task khác với endpoint.
    set_model_override(option.provider if option else None, option.model if option else None)
    async for chunk in stream_graph(**params):
        yield chunk
