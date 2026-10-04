from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.graph.build_graph import run_graph, stream_graph

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    result = await run_graph(
        session_id=req.session_id,
        user_message=req.message,
        document_ids=req.document_ids,
        space_id=req.space_id,
        user_id=req.user_id,
    )
    return result


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """
    Endpoint SSE Streaming: Bắn tiến trình các Node (Orchestrator -> Retrieval ->
    Reflection Critic -> Generation -> Critic) theo thời gian thực về UI.
    """
    return StreamingResponse(
        stream_graph(
            session_id=req.session_id,
            user_message=req.message,
            document_ids=req.document_ids,
            space_id=req.space_id,
            user_id=req.user_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
