"""
app/graph/build_graph.py

LangGraph Multi-Agent StateGraph cho StudyRAG:
- Orchestrator Agent (phân loại intent: qa/teaching/quiz/summary)
- Retrieval Agent (CRAG retrieval loop, query rewrite/decompose, hybrid search + rerank)
- Reflection Agent (Context Grading & Answer Faithfulness Critic)
- Generation Agent (QA mode kèm citation, Teaching mode scaffolding)
- Quiz & Summary Agent (chế độ quiz hoặc summary từ hội thoại)

Kiến trúc tuần hoàn (Cyclic StateGraph):
                   [START]
                      │
                      ▼
                orchestrator
                      │
          ┌───────────┴───────────┐
  (qa / teaching)            (quiz / summary)
          │                       │
          ▼                       ▼
      retrieval               retrieval
          │                       │
          ▼                       ▼
    grade_context            quiz_summary
          │                       │
    [sufficient?                  ▼
    loops < max?]               [END]
          │
      ┌───┴──────────┐
  (False)          (True)
      │              │
      ▼              ▼
rewrite_query    generation
      │              │
      └──► retrieval ▼
                grade_answer
                     │
               [faithful?
               loops < max?]
                     │
                 ┌───┴──────────┐
             (False)          (True)
                 │              │
                 ▼              ▼
             generation    accept_answer
             (with feedback)    │
                                ▼
                              [END]

Tích hợp bộ nhớ:
- Trước graph:
  + Nạp rolling summary + sliding window context qua `build_context`
  + Nạp danh sách chủ đề yếu / đến hạn ôn qua `get_weak_topics`
- Sau graph:
  + Ghi turn của assistant vào PostgreSQL
  + Kích hoạt tóm tắt ngầm định kỳ qua `maybe_summarize`
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Literal, AsyncGenerator

from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.db.session import SessionLocal
from app.memory.chat_history import get_or_create_session, add_turn
from app.memory.context_window import build_context, maybe_summarize
from app.memory.mastery_tracker import get_weak_topics
from app.agents.state import GraphState, ReflectionResult
from app.agents.orchestrator_agent import run_orchestrator
from app.agents.retrieval_agent import run_retrieval
from app.agents.reflection_agent import grade_context, grade_answer
from app.agents.generation_agent import run_generation
from app.agents.quiz_summary_agent import run_quiz_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Định nghĩa các Node trong Graph
# ---------------------------------------------------------------------------

async def orchestrator_node(state: GraphState) -> dict[str, Any]:
    """Phân loại intent và tìm chủ đề cần gợi ý ôn tập."""
    logger.info("LangGraph -> [orchestrator_node] processing query: %s", state.get("user_query", "")[:60])
    result = await run_orchestrator(state)
    return {
        "intent": result.get("intent", "qa"),
        "suggested_review_topic": result.get("suggested_review_topic"),
    }


async def retrieval_node(state: GraphState) -> dict[str, Any]:
    """Truy xuất tài liệu theo intent (hoặc theo CRAG retry feedback)."""
    logger.info("LangGraph -> [retrieval_node] intent=%s", state.get("intent"))
    result = await run_retrieval(state)
    return {
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "rewritten_queries": result.get("rewritten_queries", []),
    }


async def grade_context_node(state: GraphState) -> dict[str, Any]:
    """Reflection Agent: Chấm độ liên quan & đầy đủ của context vừa truy xuất."""
    logger.info("LangGraph -> [grade_context_node] evaluating %d chunks", len(state.get("retrieved_chunks", [])))
    grade_res = await grade_context(state)
    history = list(state.get("reflection_history") or [])
    history.append(grade_res)
    logger.info(
        "grade_context_node result: grade=%.2f, sufficient=%s",
        grade_res.get("context_grade", 0.0),
        grade_res.get("context_sufficient", True),
    )
    return {"reflection_history": history}


async def rewrite_query_node(state: GraphState) -> dict[str, Any]:
    """Kích hoạt vòng lặp CRAG retry: tăng biến đếm loop phản tư."""
    current_loops = state.get("num_reflection_loops", 0) + 1
    logger.info("LangGraph -> [rewrite_query_node] CRAG loop count=%d/%d", current_loops, settings.max_reflection_loops)
    return {"num_reflection_loops": current_loops}


async def generation_node(state: GraphState) -> dict[str, Any]:
    """Sinh câu trả lời nháp (QA có trích dẫn hoặc Teaching scaffolding)."""
    logger.info("LangGraph -> [generation_node] mode=%s", state.get("intent", "qa"))
    result = await run_generation(state)
    return {
        "answer_draft": result.get("answer_draft", ""),
        "citations": result.get("citations", []),
    }


async def grade_answer_node(state: GraphState) -> dict[str, Any]:
    """Reflection Critic: Kiểm tra tính trung thực (anti-hallucination) & completeness."""
    logger.info("LangGraph -> [grade_answer_node] evaluating draft")
    grade_res = await grade_answer(state)
    history = list(state.get("reflection_history") or [])
    history.append(grade_res)
    logger.info(
        "grade_answer_node result: grade=%.2f, faithful=%s",
        grade_res.get("answer_grade", 0.0),
        grade_res.get("answer_faithful", True),
    )
    return {"reflection_history": history}


async def retry_generation_node(state: GraphState) -> dict[str, Any]:
    """Kích hoạt vòng lặp Self-RAG regenerate: tăng biến đếm loop phản tư."""
    current_loops = state.get("num_reflection_loops", 0) + 1
    logger.info("LangGraph -> [retry_generation_node] Self-RAG retry count=%d/%d", current_loops, settings.max_reflection_loops)
    return {"num_reflection_loops": current_loops}


async def accept_answer_node(state: GraphState) -> dict[str, Any]:
    """Phê duyệt answer_draft thành final_answer sau khi Critic thông qua."""
    draft = state.get("answer_draft", "")
    citations = state.get("citations", [])
    logger.info("LangGraph -> [accept_answer_node] finalized answer length=%d", len(draft))
    return {
        "final_answer": draft,
        "citations": citations,
    }


async def quiz_summary_node(state: GraphState) -> dict[str, Any]:
    """Xử lý yêu cầu quiz/summary trực tiếp từ hội thoại chat."""
    logger.info("LangGraph -> [quiz_summary_node] intent=%s", state.get("intent"))
    result = await run_quiz_summary(state)
    return {
        "final_answer": result.get("final_answer", ""),
        "citations": result.get("citations", []),
    }


# ---------------------------------------------------------------------------
# 2. Các hàm Conditional Edge Routing
# ---------------------------------------------------------------------------

def route_after_orchestrator(state: GraphState) -> Literal["retrieval"]:
    """Mọi intent đều cần qua retrieval để lấy context/chunk phù hợp."""
    return "retrieval"


def route_after_retrieval(state: GraphState) -> Literal["quiz_summary", "grade_context"]:
    """Nếu intent là quiz hoặc summary -> rẽ sang quiz_summary; nếu là qa/teaching -> grade context."""
    intent = state.get("intent", "qa")
    if intent in ("quiz", "summary"):
        return "quiz_summary"
    return "grade_context"


def route_after_grade_context(state: GraphState) -> Literal["rewrite_query", "generation"]:
    """Kiểm tra kết quả chấm context: nếu thiếu và còn lượt loop -> rewrite_query; ngược lại -> generation."""
    history = state.get("reflection_history") or []
    if not history:
        return "generation"

    latest = history[-1]
    is_sufficient = latest.get("context_sufficient", True)
    loops = state.get("num_reflection_loops", 0)

    if not is_sufficient and loops < settings.max_reflection_loops:
        return "rewrite_query"
    return "generation"


def route_after_grade_answer(state: GraphState) -> Literal["retry_generation", "accept_answer"]:
    """Kiểm tra kết quả chấm answer: nếu chưa trung thực và còn lượt loop -> retry_generation; ngược lại -> accept."""
    history = state.get("reflection_history") or []
    if not history:
        return "accept_answer"

    latest = history[-1]
    is_faithful = latest.get("answer_faithful", True)
    loops = state.get("num_reflection_loops", 0)

    if not is_faithful and loops < settings.max_reflection_loops:
        return "retry_generation"
    return "accept_answer"


# ---------------------------------------------------------------------------
# 3. Lắp ráp & Compile LangGraph StateGraph
# ---------------------------------------------------------------------------

def build_study_graph():
    """Khởi tạo và biên dịch đồ thị LangGraph StateGraph tuần hoàn chuẩn."""
    workflow = StateGraph(GraphState)

    # Đăng ký các Nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("grade_context", grade_context_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("grade_answer", grade_answer_node)
    workflow.add_node("retry_generation", retry_generation_node)
    workflow.add_node("accept_answer", accept_answer_node)
    workflow.add_node("quiz_summary", quiz_summary_node)

    # Thiết lập các Edges
    workflow.add_edge(START, "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"retrieval": "retrieval"},
    )
    workflow.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "quiz_summary": "quiz_summary",
            "grade_context": "grade_context",
        },
    )
    workflow.add_conditional_edges(
        "grade_context",
        route_after_grade_context,
        {
            "rewrite_query": "rewrite_query",
            "generation": "generation",
        },
    )
    workflow.add_edge("rewrite_query", "retrieval")
    workflow.add_edge("generation", "grade_answer")
    workflow.add_conditional_edges(
        "grade_answer",
        route_after_grade_answer,
        {
            "retry_generation": "retry_generation",
            "accept_answer": "accept_answer",
        },
    )
    workflow.add_edge("retry_generation", "generation")
    workflow.add_edge("accept_answer", END)
    workflow.add_edge("quiz_summary", END)

    return workflow.compile()


_COMPILED_GRAPH = None


def get_graph():
    """Singleton getter cho compiled StateGraph."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_study_graph()
    return _COMPILED_GRAPH


# ---------------------------------------------------------------------------
# 4. Entrypoint run_graph (kết nối Bộ nhớ + DB + Graph execution)
# ---------------------------------------------------------------------------

def _extract_reflection_trace(final_state: GraphState) -> dict[str, Any]:
    """Trích xuất ReflectionTrace hiển thị ra UI và lưu trữ nhật ký."""
    history = final_state.get("reflection_history") or []
    ctx_grade = None
    ctx_sufficient = None
    for entry in reversed(history):
        if "context_grade" in entry:
            ctx_grade = entry.get("context_grade")
            ctx_sufficient = entry.get("context_sufficient")
            break

    ans_grade = None
    ans_faithful = None
    for entry in reversed(history):
        if "answer_grade" in entry:
            ans_grade = entry.get("answer_grade")
            ans_faithful = entry.get("answer_faithful")
            break

    return {
        "context_grade": ctx_grade,
        "context_sufficient": ctx_sufficient,
        "rewritten_queries": final_state.get("rewritten_queries") or [],
        "answer_grade": ans_grade,
        "answer_faithful": ans_faithful,
        "num_loops": final_state.get("num_reflection_loops", 0),
    }


def _normalize_uuid(val: str) -> str:
    """Chuyển đổi chuỗi bất kỳ thành UUID hợp lệ và tất định (deterministic) cho cột UUID DB."""
    if not val:
        return str(uuid.uuid4())
    try:
        uuid.UUID(str(val))
        return str(val)
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(val)))


async def run_graph(
    session_id: str,
    user_message: str,
    document_ids: list[str] | None = None,
    space_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Entrypoint chính thực thi toàn bộ luồng chat cho StudyRAG.
    
    Quy trình:
    1. Tiền xử lý bộ nhớ: Đọc chat history + mastery weak topics từ PostgreSQL.
    2. Chạy đồ thị LangGraph với recursive limit an toàn.
    3. Hậu xử lý: Ghi lại lượt chat của Assistant vào DB, kích hoạt tóm tắt ngầm.
    4. Trả về payload khớp hoàn toàn schema `ChatResponse`.
    """
    clean_message = (user_message or "").strip()
    if not clean_message:
        return {
            "intent": "qa",
            "answer": "Xin chào! Bạn có thể đặt câu hỏi về tài liệu học tập, yêu cầu giảng bài hoặc làm bài trắc nghiệm.",
            "citations": [],
            "reflection": {
                "context_grade": None,
                "context_sufficient": None,
                "rewritten_queries": [],
                "answer_grade": None,
                "answer_faithful": None,
                "num_loops": 0,
            },
        }

    db_session_id = _normalize_uuid(session_id)
    db_user_id = _normalize_uuid(user_id or "demo_user")
    db = SessionLocal()
    chat_history_summary = ""
    mastery_context: dict[str, Any] = {}

    try:
        # 1. Tải bộ nhớ trước Graph
        try:
            chat_session = get_or_create_session(db, session_id=db_session_id, user_id=db_user_id, space_id=space_id)
            user_id = chat_session.user_id or db_user_id
            add_turn(db, session_id=db_session_id, role="user", content=clean_message)
            chat_history_summary = build_context(db, session_id=db_session_id, token_budget=1000)
            weak_topics = get_weak_topics(db, user_id=user_id, limit=5)
            mastery_context = {
                "weak_topics": [
                    {
                        "topic": t.topic,
                        "mastery_score": t.mastery_score,
                        "streak": t.streak,
                        "last_reviewed_at": t.last_reviewed_at.isoformat() if t.last_reviewed_at else None,
                        "next_review_due_at": t.next_review_due_at.isoformat() if t.next_review_due_at else None,
                    }
                    for t in weak_topics
                ]
            }
        except Exception as db_exc:
            logger.warning("run_graph: Không thể truy xuất bộ nhớ từ database (%s) — dùng context rỗng.", db_exc)

        if not document_ids and space_id:
            try:
                from app.models.db_models import StudySpace
                sp = db.query(StudySpace).filter(StudySpace.id == space_id).first()
                if sp and sp.document_ids:
                    document_ids = sp.document_ids
            except Exception as sp_exc:
                logger.warning("run_graph: Không thể nạp document_ids từ space_id '%s': %s", space_id, sp_exc)

        # 2. Chuẩn bị Initial State
        initial_state: GraphState = {
            "session_id": session_id,
            "user_query": clean_message,
            "chat_history_summary": chat_history_summary,
            "mastery_context": mastery_context,
            "num_reflection_loops": 0,
            "reflection_history": [],
            "rewritten_queries": [],
            "retrieved_chunks": [],
            "citations": [],
            "answer_draft": "",
            "final_answer": "",
            "document_ids": document_ids,
        }

        # 3. Kích hoạt StateGraph
        compiled_graph = get_graph()
        try:
            final_state = await compiled_graph.ainvoke(initial_state, config={"recursion_limit": 25})
        except Exception as graph_exc:
            logger.exception("run_graph: Lỗi thực thi LangGraph StateGraph: %s", graph_exc)
            return {
                "intent": "qa",
                "answer": "Xin lỗi, đã xảy ra sự cố trong quá trình xử lý phản tư và tạo câu trả lời. Vui lòng thử lại sau.",
                "citations": [],
                "reflection": {
                    "context_grade": None,
                    "context_sufficient": None,
                    "rewritten_queries": [],
                    "answer_grade": None,
                    "answer_faithful": None,
                    "num_loops": 0,
                },
            }

        intent = final_state.get("intent", "qa")
        final_answer = final_state.get("final_answer") or final_state.get("answer_draft") or ""
        citations = final_state.get("citations") or []
        reflection_trace = _extract_reflection_trace(final_state)

        # 4. Ghi nhận nhật ký & cập nhật bộ nhớ sau Graph
        try:
            add_turn(
                db,
                session_id=db_session_id,
                role="assistant",
                content=final_answer,
                intent=intent,
                reflection_trace=reflection_trace,
            )
            # Tóm tắt ngầm nếu số turn mới tích lũy đủ ngưỡng
            maybe_summarize(db, session_id=db_session_id)
        except Exception as save_exc:
            logger.warning("run_graph: Không thể lưu Assistant turn vào DB (%s).", save_exc)

        return {
            "intent": intent,
            "answer": final_answer,
            "citations": citations,
            "reflection": reflection_trace,
        }

    finally:
        db.close()


async def stream_graph(
    session_id: str,
    user_message: str,
    document_ids: list[str] | None = None,
    space_id: str | None = None,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generator Server-Sent Events (SSE) phát tiến trình thực thi từng Node
    của LangGraph StateGraph về phía Frontend thời gian thực.
    """
    clean_message = (user_message or "").strip()
    if not clean_message:
        fallback = {
            "intent": "qa",
            "answer": "Xin chào! Bạn có thể đặt câu hỏi về tài liệu học tập, yêu cầu giảng bài hoặc làm bài trắc nghiệm.",
            "citations": [],
            "reflection": {
                "context_grade": None,
                "context_sufficient": None,
                "rewritten_queries": [],
                "answer_grade": None,
                "answer_faithful": None,
                "num_loops": 0,
            },
        }
        yield f"data: {json.dumps({'event': 'done', 'response': fallback})}\n\n"
        return

    db_session_id = _normalize_uuid(session_id)
    db_user_id = _normalize_uuid(user_id or "demo_user")
    db = SessionLocal()
    chat_history_summary = ""
    mastery_context: dict[str, Any] = {}

    try:
        yield f"data: {json.dumps({'event': 'status', 'step': 'init', 'message': 'Đang kết nối bộ nhớ ngữ cảnh và kiểm tra lịch sử...'})}\n\n"
        try:
            chat_session = get_or_create_session(db, session_id=db_session_id, user_id=db_user_id, space_id=space_id)
            user_id = chat_session.user_id or db_user_id
            add_turn(db, session_id=db_session_id, role="user", content=clean_message)
            chat_history_summary = build_context(db, session_id=db_session_id, token_budget=1000)
            weak_topics = get_weak_topics(db, user_id=user_id, limit=5)
            mastery_context = {
                "weak_topics": [
                    {
                        "topic": t.topic,
                        "mastery_score": t.mastery_score,
                        "streak": t.streak,
                        "last_reviewed_at": t.last_reviewed_at.isoformat() if t.last_reviewed_at else None,
                        "next_review_due_at": t.next_review_due_at.isoformat() if t.next_review_due_at else None,
                    }
                    for t in weak_topics
                ]
            }
        except Exception as db_exc:
            logger.warning("stream_graph: Không thể nạp memory từ DB (%s).", db_exc)

        if not document_ids and space_id:
            try:
                from app.models.db_models import StudySpace
                sp = db.query(StudySpace).filter(StudySpace.id == space_id).first()
                if sp and sp.document_ids:
                    document_ids = sp.document_ids
            except Exception as sp_exc:
                logger.warning("stream_graph: Không thể nạp document_ids từ space_id '%s': %s", space_id, sp_exc)

        initial_state: GraphState = {
            "session_id": session_id,
            "user_query": clean_message,
            "chat_history_summary": chat_history_summary,
            "mastery_context": mastery_context,
            "num_reflection_loops": 0,
            "reflection_history": [],
            "rewritten_queries": [],
            "retrieved_chunks": [],
            "citations": [],
            "answer_draft": "",
            "final_answer": "",
            "document_ids": document_ids,
        }

        compiled_graph = get_graph()
        accumulated_state: dict[str, Any] = dict(initial_state)

        async for chunk in compiled_graph.astream(initial_state, stream_mode="updates", config={"recursion_limit": 25}):
            for node_name, node_output in chunk.items():
                accumulated_state.update(node_output)

                if node_name == "orchestrator":
                    yield f"data: {json.dumps({'event': 'step', 'node': 'orchestrator', 'title': 'Phân Loại Ý Định', 'intent': node_output.get('intent'), 'review_topic': node_output.get('suggested_review_topic')})}\n\n"

                elif node_name == "retrieval":
                    chunks = node_output.get("retrieved_chunks") or []
                    yield f"data: {json.dumps({'event': 'step', 'node': 'retrieval', 'title': 'Truy Xuất Ngữ Cảnh', 'chunks_found': len(chunks), 'rewritten_queries': node_output.get('rewritten_queries', [])})}\n\n"

                elif node_name == "grade_context":
                    history = node_output.get("reflection_history") or []
                    latest = history[-1] if history else {}
                    yield f"data: {json.dumps({'event': 'step', 'node': 'grade_context', 'title': 'Reflection Critic: Chấm Ngữ Cảnh', 'grade': latest.get('context_grade'), 'sufficient': latest.get('context_sufficient'), 'feedback': latest.get('feedback', '')})}\n\n"

                elif node_name == "rewrite_query":
                    loops = node_output.get("num_reflection_loops", 1)
                    yield f"data: {json.dumps({'event': 'step', 'node': 'rewrite_query', 'title': 'CRAG Loop: Viết Lại Truy Vấn', 'loop': loops})}\n\n"

                elif node_name == "generation":
                    citations = node_output.get("citations") or []
                    yield f"data: {json.dumps({'event': 'step', 'node': 'generation', 'title': 'Sinh Câu Trả Lời Nháp', 'citations': citations})}\n\n"

                elif node_name == "grade_answer":
                    history = node_output.get("reflection_history") or []
                    latest = history[-1] if history else {}
                    yield f"data: {json.dumps({'event': 'step', 'node': 'grade_answer', 'title': 'Critic: Kiểm Tra Tính Trung Thực (Anti-Hallucination)', 'grade': latest.get('answer_grade'), 'faithful': latest.get('answer_faithful'), 'feedback': latest.get('feedback', '')})}\n\n"

                elif node_name == "retry_generation":
                    loops = node_output.get("num_reflection_loops", 1)
                    yield f"data: {json.dumps({'event': 'step', 'node': 'retry_generation', 'title': 'Self-RAG Loop: Sinh Lại Bản Thảo', 'loop': loops})}\n\n"

                elif node_name in ("accept_answer", "quiz_summary"):
                    yield f"data: {json.dumps({'event': 'step', 'node': node_name, 'title': 'Hoàn Tất Câu Trả Lời'})}\n\n"

        intent = accumulated_state.get("intent", "qa")
        final_answer = accumulated_state.get("final_answer") or accumulated_state.get("answer_draft") or ""
        citations = accumulated_state.get("citations") or []
        reflection_trace = _extract_reflection_trace(accumulated_state)

        try:
            add_turn(
                db,
                session_id=db_session_id,
                role="assistant",
                content=final_answer,
                intent=intent,
                reflection_trace=reflection_trace,
            )
            maybe_summarize(db, session_id=db_session_id)
        except Exception as save_exc:
            logger.warning("stream_graph: Không thể lưu Assistant turn (%s).", save_exc)

        final_response = {
            "intent": intent,
            "answer": final_answer,
            "citations": citations,
            "reflection": reflection_trace,
        }
        yield f"data: {json.dumps({'event': 'done', 'response': final_response})}\n\n"

    except Exception as exc:
        logger.exception("stream_graph: Lỗi trong quá trình stream: %s", exc)
        yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"
    finally:
        db.close()

