"""
app/agents/retrieval_agent.py

Retrieval Agent — ghép query_processing (rewrite/decompose) + embedder +
Qdrant hybrid search + BGE reranker thành 1 node hoàn chỉnh. Đây cũng là
node được quay lại bởi vòng lặp CRAG khi Reflection Agent báo context
chưa đủ.

Thiết kế idempotent (an toàn khi bị gọi lại nhiều lần trong 1 request):
- state["retrieved_chunks"] LUÔN bị GHI ĐÈ hoàn toàn mỗi lần chạy (không
  append) — kết quả cũ đã lỗi thời khi bị gọi lại.
- state["rewritten_queries"] được NỐI THÊM (append) qua các lần chạy — để
  hiển thị ra UI đầy đủ lịch sử các câu hỏi đã thử (ReflectionTrace.
  rewritten_queries trong app/models/schemas.py).
- Không có side-effect nào khác (không ghi DB, không gọi API có state
  ngoài) — hàm thuần map (state cũ) -> (state mới).

Phân biệt 2 concern khi "viết lại câu hỏi":
1. query_processing.rewrite_query(): resolve đại từ mơ hồ dựa vào lịch sử
   HỘI THOẠI (turn trước nói gì) — chạy ở lần đầu tiên, trước khi biết
   context có đủ hay không.
2. _reformulate_for_crag() (private, định nghĩa ngay trong file này): viết
   lại câu hỏi dựa vào FEEDBACK của Reflection Agent (context thiếu ý gì)
   — chạy ở các lần lặp CRAG sau, khi grade_context báo không đủ. Đây là
   concern khác (sửa lỗi retrieval theo phản hồi cụ thể, không phải resolve
   ngôn ngữ tự nhiên) nên KHÔNG tái dùng rewrite_query — heuristic bên đó
   chỉ trigger khi có đại từ mơ hồ, sẽ bỏ qua sai use case CRAG.

Ghi chú diễn giải phạm vi "summary" trong chat graph (do đề bài không nói
rõ graph lấy document_ids từ đâu khi user gõ "tóm tắt tài liệu X" trong
chat, trong khi GraphState không có field document_ids): coi intent
"summary" TỪ CHAT là tóm tắt Ở QUY MÔ HỘI THOẠI hiện tại (search rộng hơn
QA/teaching nhưng vẫn top-k, KHÔNG map-reduce toàn văn tài liệu) — bản tóm
tắt ĐẦY ĐỦ toàn văn 1 tài liệu cụ thể theo document_ids dùng
generate_summary() trong quiz_summary_agent.py, gọi qua REST endpoint
riêng. Tương tự cho "quiz".
"""
from __future__ import annotations

import logging
from typing import Any

from app.agents.constants import FINAL_TOP_K_BROAD, FINAL_TOP_K_QA
from app.agents.state import GraphState
from app.config import settings
from app.llm.utils import LLMCallError, acall_llm_text
from app.retrieval.expansion import aexpand_with_parents
from app.retrieval.helpers import aget_all_chunks_for_documents, arerank, normalize_chunk, retrieve_top_k
from app.retrieval.query_processing import decompose_query, rewrite_query

logger = logging.getLogger(__name__)

MAX_WEAK_TOPICS_PER_QUIZ_RETRIEVAL = 5


def _last_context_feedback(state: GraphState) -> str | None:
    """Tìm feedback GẦN NHẤT từ 1 lần grade_context — nếu lần gần nhất báo
    'chưa đủ', trả về feedback đó (đang trong CRAG retry). Nếu lần gần
    nhất ĐÃ đủ (hoặc chưa có lần nào), trả về None."""
    history = state.get("reflection_history") or []
    for entry in reversed(history):
        if "context_sufficient" in entry:
            return entry.get("feedback") or "" if entry.get("context_sufficient") is False else None
    return None


async def _reformulate_for_crag(original_query: str, feedback: str) -> str:
    """Viết lại câu hỏi để retrieve bổ sung đúng phần còn thiếu, dựa trên
    feedback cụ thể của Reflection Agent. Dùng task="rewrite" (cheap, temp
    thấp — cần bám sát ý gốc, không sáng tạo thêm)."""
    system_prompt = (
        "Bạn là trợ lý viết lại câu hỏi tìm kiếm cho hệ thống RAG. Câu hỏi "
        "gốc vừa được tìm kiếm nhưng ngữ cảnh trả về CHƯA ĐỦ để trả lời, "
        "theo phản hồi cụ thể dưới đây. Viết lại câu hỏi để tìm kiếm bổ "
        "sung đúng phần còn thiếu — có thể mở rộng từ khoá, thêm khái niệm "
        "liên quan được nhắc trong phản hồi. CHỈ trả về câu hỏi mới, không "
        "thêm giải thích, không thêm dấu ngoặc kép."
    )
    user_prompt = f"Câu hỏi gốc: {original_query}\n\nPhản hồi (đang thiếu gì): {feedback}"
    try:
        rewritten = await acall_llm_text("rewrite", system_prompt, user_prompt)
        rewritten = rewritten.strip().strip('"').strip("'")
        return rewritten or original_query
    except LLMCallError as exc:
        logger.warning("_reformulate_for_crag: LLM thất bại (%s) — dùng câu hỏi gốc.", exc)
        return original_query


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Loại candidate trùng khi gộp kết quả từ nhiều sub-query — so khớp
    theo (source_file, page, 200 ký tự đầu của text) vì chunk không chắc
    có id thống nhất giữa các lần search khác nhau."""
    seen: set[tuple] = set()
    deduped = []
    for c in candidates:
        key = (c.get("source_file"), c.get("page"), c.get("text", "")[:200])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def _round_robin(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Trộn xen kẽ kết quả các truy vấn con để khi cắt pool mỗi truy vấn đều còn đại diện."""
    merged: list[dict[str, Any]] = []
    for rank in range(max((len(g) for g in groups), default=0)):
        merged.extend(g[rank] for g in groups if rank < len(g))
    return merged


def _extract_weak_topic_names(state: GraphState) -> list[str]:
    mastery_context = state.get("mastery_context") or {}
    weak_topics = mastery_context.get("weak_topics") or []
    names = []
    for t in weak_topics:
        name = t.get("topic") if isinstance(t, dict) else getattr(t, "topic", None)
        if name:
            names.append(name)
    return names[:MAX_WEAK_TOPICS_PER_QUIZ_RETRIEVAL]


async def run_retrieval(state: GraphState) -> GraphState:
    """
    Cập nhật state["retrieved_chunks"] và state["rewritten_queries"].

    Luồng:
    1. Nếu đang trong vòng lặp CRAG (grade_context lần trước báo thiếu):
       viết lại câu hỏi theo feedback đó. Ngược lại: rewrite_query() để
       resolve đại từ mơ hồ dựa vào lịch sử hội thoại.
    2. Với intent="quiz": ưu tiên search theo (các) topic yếu trong
       mastery_context thay vì câu hỏi gốc. Với intent="qa"/"teaching":
       decompose_query() nếu câu hỏi (đã rewrite) có dấu hiệu multi-hop.
    3. Retrieve top-k cho từng (sub-)query, gộp + khử trùng lặp, rerank
       lần cuối theo câu hỏi chính để có thứ tự liên quan nhất.
    """
    user_query = (state.get("user_query") or "").strip()
    chat_history_summary = state.get("chat_history_summary") or ""
    intent = state.get("intent", "qa")
    rewritten_queries = list(state.get("rewritten_queries") or [])

    crag_feedback = _last_context_feedback(state)
    if crag_feedback:
        main_query = await _reformulate_for_crag(user_query, crag_feedback)
        logger.info("run_retrieval: CRAG retry, query mới = '%s'", main_query)
    elif user_query:
        # Orchestrator đã viết lại câu nối tiếp trong cùng lời gọi phân loại; nó lỗi thì dùng regex đại từ.
        main_query = (state.get("standalone_query") or "").strip() or await rewrite_query(
            user_query, chat_history_summary
        )
    else:
        main_query = ""

    if main_query and main_query != user_query:
        rewritten_queries.append(main_query)

    search_queries: list[str]
    final_top_k = FINAL_TOP_K_QA
    document_ids = state.get("document_ids") or []  # phạm vi tài liệu đã xác thực ở tầng API

    if intent == "quiz":
        weak_topics = _extract_weak_topic_names(state)
        if weak_topics:
            search_queries = weak_topics
        else:
            fallback_topic = state.get("suggested_review_topic")
            search_queries = [fallback_topic or main_query or "kiến thức chính"]
        final_top_k = FINAL_TOP_K_BROAD
    elif intent == "summary":
        search_queries = [main_query] if main_query else []
        search_queries.extend(["tổng quan kiến thức chính tóm tắt nội dung", "mục tiêu khái quát bài học"])
        final_top_k = FINAL_TOP_K_BROAD
    elif main_query:
        search_queries = await decompose_query(main_query)
    else:
        search_queries = []

    # Nhiều truy vấn con: chỉ search từng truy vấn, gộp lại rồi rerank MỘT lần ở dưới
    # (rerank CPU ~1s/cặp, rerank từng truy vấn rồi rerank lại là trả tiền hai lần).
    multi_query = len(search_queries) > 1
    per_query: list[list[dict[str, Any]]] = []
    for q in search_queries:
        # Số ứng viên trước rerank = RERANK_CANDIDATES (settings), nhưng không ít hơn final_top_k.
        n_candidates = max(settings.rerank_candidates, final_top_k)
        hits = await retrieve_top_k(
            q, top_k=final_top_k, hybrid_top_k=n_candidates, document_ids=document_ids, rerank=not multi_query
        )
        per_query.append([normalize_chunk(h) for h in hits])
    all_candidates = _round_robin(per_query)

    # Fallback: search không ra kết quả nào nhưng có tài liệu trong phạm vi (vd câu hỏi
    # quá chung chung) -> lấy trực tiếp chunk đầu của các tài liệu đó.
    if not all_candidates and document_ids:
        logger.info("run_retrieval: không có kết quả search — fallback lấy chunk trực tiếp từ %d tài liệu.", len(document_ids))
        try:
            raw_fallback = await aget_all_chunks_for_documents(document_ids)
        except Exception as exc:
            logger.warning("run_retrieval: fallback lấy chunk thất bại (%s).", exc)
            raw_fallback = []
        all_candidates.extend(normalize_chunk(c) for c in raw_fallback[:final_top_k * 2])

    pool = _dedupe_candidates(all_candidates)
    rerank_query = main_query or (search_queries[0] if search_queries else "")

    if settings.rerank_enabled and multi_query and pool and rerank_query:
        pool = pool[: max(settings.rerank_pool_max, final_top_k)]
        try:
            pool = await arerank(rerank_query, pool, top_k=final_top_k)
        except Exception as exc:
            logger.warning("run_retrieval: rerank cuối thất bại (%s) — cắt top_k không rerank.", exc)
            pool = pool[:final_top_k]
    else:
        pool = pool[:final_top_k]

    if intent in ("qa", "teaching") and settings.enable_parent_expansion:
        pool = await aexpand_with_parents(pool, settings.parent_context_max_tokens)

    logger.info(
        "run_retrieval: intent=%s, %d search-query(ies), %d chunk cuối cùng.",
        intent, len(search_queries), len(pool),
    )

    return {**state, "retrieved_chunks": pool, "rewritten_queries": rewritten_queries}
