"""
app/graph/build_graph.py  (v2)

Topology đồ thị KHÔNG đổi so với bản cũ (orchestrator -> retrieval -> CRAG loop ->
generation -> Self-RAG loop -> accept; quiz/summary rẽ nhánh sau retrieval).
Các node và hàm route_* giữ nguyên tên vì eval/pipelines.py import trực tiếp.

Thay đổi:
1. Mọi thao tác DB đồng bộ (SQLAlchemy) chạy trong `asyncio.to_thread` với Session
   riêng cho từng việc -> không còn chặn event loop của FastAPI.
2. `maybe_summarize` (gọi LLM đồng bộ, vài giây) KHÔNG còn chạy trước khi trả
   kết quả: chạy nền sau khi đã phản hồi, có khoá theo session để không gộp trùng.
3. run_graph / stream_graph dùng chung _prepare_state / _build_response /
   _persist (trước đây ~100 dòng bị lặp 2 lần).
4. reflection trace trung thực: có `*_verified` (bộ chấm có chạy được không) và
   `*_forced` (bị ép chấp nhận do hết lượt thử); citations được lưu cùng turn để
   mở lại lịch sử vẫn thấy nguồn.
5. `document_ids is None` mới nạp từ space (list rỗng nghĩa là "không có tài liệu").
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Literal

from langgraph.graph import END, START, StateGraph

from app.agents.generation_agent import run_generation
from app.agents.orchestrator_agent import run_orchestrator
from app.agents.quiz_summary_agent import run_quiz_summary
from app.agents.reflection_agent import grade_answer, grade_context
from app.agents.retrieval_agent import run_retrieval
from app.agents.state import GraphState
from app.config import settings
from app.db.session import SessionLocal
from app.memory.chat_history import add_turn, get_or_create_session
from app.memory.context_window import build_context, maybe_summarize
from app.memory.mastery_tracker import get_weak_topics
from app.models.db_models import StudySpace

logger = logging.getLogger(__name__)


def _fmt(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"


# ---------------------------------------------------------------------------
# 1. Nodes
# ---------------------------------------------------------------------------
async def orchestrator_node(state: GraphState) -> dict[str, Any]:
    logger.info("LangGraph -> [orchestrator] query=%s", state.get("user_query", "")[:60])
    result = await run_orchestrator(state)
    return {
        "intent": result.get("intent", "qa"),
        "suggested_review_topic": result.get("suggested_review_topic"),
    }


async def retrieval_node(state: GraphState) -> dict[str, Any]:
    logger.info("LangGraph -> [retrieval] intent=%s", state.get("intent"))
    result = await run_retrieval(state)
    return {
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "rewritten_queries": result.get("rewritten_queries", []),
    }


async def grade_context_node(state: GraphState) -> dict[str, Any]:
    grade_res = await grade_context(state)
    history = list(state.get("reflection_history") or [])
    history.append(grade_res)
    logger.info(
        "grade_context: grade=%s sufficient=%s graded=%s forced=%s",
        _fmt(grade_res.get("context_grade")), grade_res.get("context_sufficient"),
        grade_res.get("graded", True), grade_res.get("forced", False),
    )
    return {"reflection_history": history}


async def rewrite_query_node(state: GraphState) -> dict[str, Any]:
    loops = state.get("num_reflection_loops", 0) + 1
    logger.info("LangGraph -> [rewrite_query] CRAG loop %d/%d", loops, settings.max_reflection_loops)
    return {"num_reflection_loops": loops}


async def generation_node(state: GraphState) -> dict[str, Any]:
    logger.info("LangGraph -> [generation] mode=%s", state.get("intent", "qa"))
    result = await run_generation(state)
    return {"answer_draft": result.get("answer_draft", ""), "citations": result.get("citations", [])}


async def grade_answer_node(state: GraphState) -> dict[str, Any]:
    grade_res = await grade_answer(state)
    history = list(state.get("reflection_history") or [])
    history.append(grade_res)
    logger.info(
        "grade_answer: grade=%s faithful=%s graded=%s forced=%s",
        _fmt(grade_res.get("answer_grade")), grade_res.get("answer_faithful"),
        grade_res.get("graded", True), grade_res.get("forced", False),
    )
    return {"reflection_history": history}


async def retry_generation_node(state: GraphState) -> dict[str, Any]:
    loops = state.get("num_reflection_loops", 0) + 1
    logger.info("LangGraph -> [retry_generation] Self-RAG loop %d/%d", loops, settings.max_reflection_loops)
    return {"num_reflection_loops": loops}


async def accept_answer_node(state: GraphState) -> dict[str, Any]:
    draft = state.get("answer_draft", "")
    return {"final_answer": draft, "citations": state.get("citations", [])}


async def quiz_summary_node(state: GraphState) -> dict[str, Any]:
    result = await run_quiz_summary(state)
    return {"final_answer": result.get("final_answer", ""), "citations": result.get("citations", [])}


# ---------------------------------------------------------------------------
# 2. Routing
# ---------------------------------------------------------------------------
def route_after_orchestrator(state: GraphState) -> Literal["retrieval"]:
    return "retrieval"


def route_after_retrieval(state: GraphState) -> Literal["quiz_summary", "grade_context"]:
    return "quiz_summary" if state.get("intent", "qa") in ("quiz", "summary") else "grade_context"


def route_after_grade_context(state: GraphState) -> Literal["rewrite_query", "generation"]:
    history = state.get("reflection_history") or []
    if not history:
        return "generation"
    sufficient = history[-1].get("context_sufficient", True)
    if not sufficient and state.get("num_reflection_loops", 0) < settings.max_reflection_loops:
        return "rewrite_query"
    return "generation"


def route_after_grade_answer(state: GraphState) -> Literal["retry_generation", "accept_answer"]:
    history = state.get("reflection_history") or []
    if not history:
        return "accept_answer"
    faithful = history[-1].get("answer_faithful", True)
    if not faithful and state.get("num_reflection_loops", 0) < settings.max_reflection_loops:
        return "retry_generation"
    return "accept_answer"


# ---------------------------------------------------------------------------
# 3. Compile
# ---------------------------------------------------------------------------
def build_study_graph():
    wf = StateGraph(GraphState)
    wf.add_node("orchestrator", orchestrator_node)
    wf.add_node("retrieval", retrieval_node)
    wf.add_node("grade_context", grade_context_node)
    wf.add_node("rewrite_query", rewrite_query_node)
    wf.add_node("generation", generation_node)
    wf.add_node("grade_answer", grade_answer_node)
    wf.add_node("retry_generation", retry_generation_node)
    wf.add_node("accept_answer", accept_answer_node)
    wf.add_node("quiz_summary", quiz_summary_node)

    wf.add_edge(START, "orchestrator")
    wf.add_conditional_edges("orchestrator", route_after_orchestrator, {"retrieval": "retrieval"})
    wf.add_conditional_edges(
        "retrieval", route_after_retrieval,
        {"quiz_summary": "quiz_summary", "grade_context": "grade_context"},
    )
    wf.add_conditional_edges(
        "grade_context", route_after_grade_context,
        {"rewrite_query": "rewrite_query", "generation": "generation"},
    )
    wf.add_edge("rewrite_query", "retrieval")
    wf.add_edge("generation", "grade_answer")
    wf.add_conditional_edges(
        "grade_answer", route_after_grade_answer,
        {"retry_generation": "retry_generation", "accept_answer": "accept_answer"},
    )
    wf.add_edge("retry_generation", "generation")
    wf.add_edge("accept_answer", END)
    wf.add_edge("quiz_summary", END)
    return wf.compile()


_COMPILED_GRAPH = None


def get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_study_graph()
    return _COMPILED_GRAPH


# ---------------------------------------------------------------------------
# 4. Helpers dùng chung cho run_graph / stream_graph
# ---------------------------------------------------------------------------
def _normalize_uuid(val: str | None) -> str:
    if not val:
        return str(uuid.uuid4())
    try:
        uuid.UUID(str(val))
        return str(val)
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(val)))


def _latest(history: list[dict], key: str) -> dict | None:
    return next((e for e in reversed(history) if key in e), None)


def _extract_reflection_trace(final_state: GraphState) -> dict[str, Any]:
    history = final_state.get("reflection_history") or []
    ctx = _latest(history, "context_grade")
    ans = _latest(history, "answer_grade")
    return {
        "context_grade": ctx.get("context_grade") if ctx else None,
        "context_sufficient": ctx.get("context_sufficient") if ctx else None,
        "context_verified": ctx.get("graded", True) if ctx else None,
        "context_forced": bool(ctx.get("forced", False)) if ctx else None,
        "rewritten_queries": final_state.get("rewritten_queries") or [],
        "answer_grade": ans.get("answer_grade") if ans else None,
        "answer_faithful": ans.get("answer_faithful") if ans else None,
        "answer_verified": ans.get("graded", True) if ans else None,
        "answer_forced": bool(ans.get("forced", False)) if ans else None,
        "num_loops": final_state.get("num_reflection_loops", 0),
    }


def _empty_trace() -> dict[str, Any]:
    return _extract_reflection_trace({})


def _greeting_response() -> dict[str, Any]:
    return {
        "intent": "qa",
        "answer": "Xin chào! Bạn có thể đặt câu hỏi về tài liệu học tập, yêu cầu giảng bài hoặc làm bài trắc nghiệm.",
        "citations": [],
        "reflection": _empty_trace(),
    }


def _error_response() -> dict[str, Any]:
    return {
        "intent": "qa",
        "answer": "Xin lỗi, đã xảy ra sự cố khi xử lý câu hỏi. Vui lòng thử lại sau.",
        "citations": [],
        "reflection": _empty_trace(),
    }


def _build_response(final_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent": final_state.get("intent", "qa"),
        "answer": final_state.get("final_answer") or final_state.get("answer_draft") or "",
        "citations": final_state.get("citations") or [],
        "reflection": _extract_reflection_trace(final_state),  # type: ignore[arg-type]
    }


# --- Các hàm ĐỒNG BỘ (chạy trong thread, mỗi hàm tự mở Session) -------------
def _load_memory_sync(
    db_session_id: str, db_user_id: str, space_id: str | None, message: str, document_ids: list[str] | None
) -> tuple[str, dict[str, Any], list[str] | None]:
    summary, mastery = "", {}
    with SessionLocal() as db:
        try:
            chat_session = get_or_create_session(db, session_id=db_session_id, user_id=db_user_id, space_id=space_id)
            add_turn(db, session_id=db_session_id, role="user", content=message)
            summary = build_context(db, session_id=db_session_id, token_budget=1000)
            weak = get_weak_topics(db, user_id=str(chat_session.user_id or db_user_id), limit=5)
            mastery = {
                "weak_topics": [
                    {
                        "topic": t.topic,
                        "mastery_score": t.mastery_score,
                        "streak": t.streak,
                        "last_reviewed_at": t.last_reviewed_at.isoformat() if t.last_reviewed_at else None,
                        "next_review_due_at": t.next_review_due_at.isoformat() if t.next_review_due_at else None,
                    }
                    for t in weak
                ]
            }
        except Exception as exc:
            db.rollback()
            logger.warning("Không nạp được bộ nhớ hội thoại (%s) — dùng context rỗng.", exc)

        if document_ids is None and space_id:
            try:
                sp = db.query(StudySpace).filter(StudySpace.id == space_id).first()
                document_ids = list(sp.document_ids or []) if sp else []
            except Exception as exc:
                logger.warning("Không nạp được document_ids của space '%s': %s", space_id, exc)
    return summary, mastery, document_ids


def _save_assistant_turn_sync(db_session_id: str, response: dict[str, Any]) -> None:
    trace = {**response["reflection"], "citations": response["citations"]}
    with SessionLocal() as db:
        add_turn(
            db, session_id=db_session_id, role="assistant", content=response["answer"],
            intent=response["intent"], reflection_trace=trace,
        )


def _summarize_sync(db_session_id: str) -> None:
    with SessionLocal() as db:
        maybe_summarize(db, session_id=db_session_id)


# --- Tóm tắt nền -----------------------------------------------------------
_bg_tasks: set[asyncio.Task] = set()
_summarizing: set[str] = set()


def _schedule_summarize(db_session_id: str) -> None:
    if db_session_id in _summarizing:
        return
    _summarizing.add(db_session_id)

    async def _job() -> None:
        try:
            await asyncio.to_thread(_summarize_sync, db_session_id)
        except Exception:
            logger.exception("Tóm tắt nền thất bại (session=%s).", db_session_id)
        finally:
            _summarizing.discard(db_session_id)

    task = asyncio.create_task(_job())
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


# --- Async wrappers --------------------------------------------------------
async def _prepare_state(
    session_id: str, message: str, document_ids: list[str] | None, space_id: str | None, user_id: str | None
) -> tuple[str, GraphState]:
    db_session_id = _normalize_uuid(session_id)
    db_user_id = _normalize_uuid(user_id or "anonymous")
    summary, mastery, document_ids = await asyncio.to_thread(
        _load_memory_sync, db_session_id, db_user_id, space_id, message, document_ids
    )
    state: GraphState = {
        "session_id": session_id,
        "user_query": message,
        "chat_history_summary": summary,
        "mastery_context": mastery,
        "num_reflection_loops": 0,
        "reflection_history": [],
        "rewritten_queries": [],
        "retrieved_chunks": [],
        "citations": [],
        "answer_draft": "",
        "final_answer": "",
        "document_ids": document_ids,
    }
    return db_session_id, state


async def _persist(db_session_id: str, response: dict[str, Any]) -> None:
    try:
        await asyncio.to_thread(_save_assistant_turn_sync, db_session_id, response)
    except Exception as exc:
        logger.warning("Không lưu được lượt trả lời của trợ lý (%s).", exc)
        return
    _schedule_summarize(db_session_id)


# ---------------------------------------------------------------------------
# 5. Entrypoints
# ---------------------------------------------------------------------------
async def run_graph(
    session_id: str,
    user_message: str,
    document_ids: list[str] | None = None,
    space_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    message = (user_message or "").strip()
    if not message:
        return _greeting_response()

    db_session_id, state = await _prepare_state(session_id, message, document_ids, space_id, user_id)
    try:
        final_state = await get_graph().ainvoke(state, config={"recursion_limit": 25})
    except Exception:
        logger.exception("run_graph: lỗi thực thi LangGraph.")
        return _error_response()

    response = _build_response(final_state)
    await _persist(db_session_id, response)
    return response


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _step_event(node: str, out: dict[str, Any]) -> dict[str, Any] | None:
    """Ánh xạ output của node -> sự kiện SSE cho UI."""
    base = {"event": "step", "node": node}
    if node == "orchestrator":
        return {**base, "title": "Phân Loại Ý Định", "intent": out.get("intent"),
                "review_topic": out.get("suggested_review_topic")}
    if node == "retrieval":
        return {**base, "title": "Truy Xuất Ngữ Cảnh", "chunks_found": len(out.get("retrieved_chunks") or []),
                "rewritten_queries": out.get("rewritten_queries", [])}
    if node == "grade_context":
        h = (out.get("reflection_history") or [{}])[-1]
        return {**base, "title": "Reflection Critic: Chấm Ngữ Cảnh", "grade": h.get("context_grade"),
                "sufficient": h.get("context_sufficient"), "graded": h.get("graded", True),
                "forced": h.get("forced", False), "feedback": h.get("feedback", "")}
    if node == "rewrite_query":
        return {**base, "title": "CRAG Loop: Viết Lại Truy Vấn", "loop": out.get("num_reflection_loops", 1)}
    if node == "generation":
        return {**base, "title": "Sinh Câu Trả Lời Nháp", "citations": out.get("citations") or []}
    if node == "grade_answer":
        h = (out.get("reflection_history") or [{}])[-1]
        return {**base, "title": "Critic: Kiểm Tra Tính Trung Thực", "grade": h.get("answer_grade"),
                "faithful": h.get("answer_faithful"), "graded": h.get("graded", True),
                "forced": h.get("forced", False), "feedback": h.get("feedback", "")}
    if node == "retry_generation":
        return {**base, "title": "Self-RAG Loop: Sinh Lại Bản Thảo", "loop": out.get("num_reflection_loops", 1)}
    if node in ("accept_answer", "quiz_summary"):
        return {**base, "title": "Hoàn Tất Câu Trả Lời"}
    return None


async def stream_graph(
    session_id: str,
    user_message: str,
    document_ids: list[str] | None = None,
    space_id: str | None = None,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """SSE: phát tiến trình từng node; gửi 'done' NGAY khi có đáp án (tóm tắt nền chạy sau)."""
    message = (user_message or "").strip()
    if not message:
        yield _sse({"event": "done", "response": _greeting_response()})
        return

    try:
        yield _sse({"event": "status", "step": "init", "message": "Đang nạp bộ nhớ hội thoại..."})
        db_session_id, state = await _prepare_state(session_id, message, document_ids, space_id, user_id)

        accumulated: dict[str, Any] = dict(state)
        async for chunk in get_graph().astream(state, stream_mode="updates", config={"recursion_limit": 25}):
            for node_name, node_output in chunk.items():
                accumulated.update(node_output)
                event = _step_event(node_name, node_output)
                if event:
                    yield _sse(event)

        response = _build_response(accumulated)
        await _persist(db_session_id, response)
        yield _sse({"event": "done", "response": response})
    except Exception as exc:
        logger.exception("stream_graph: lỗi trong quá trình stream.")
        yield _sse({"event": "error", "message": str(exc)})
