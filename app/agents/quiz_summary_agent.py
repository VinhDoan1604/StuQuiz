"""
app/agents/quiz_summary_agent.py

Quiz & Summary Agent — 2 chức năng "structured content generation":

1. generate_quiz() / generate_summary(): hàm ĐỘC LẬP dùng cho REST endpoint
   riêng (vd POST /quiz/generate nhận QuizGenerateRequest, POST
   /documents/{id}/summary) — TỰ retrieve theo document_ids,
   generate_summary dùng ĐẦY ĐỦ nội dung tài liệu (map-reduce nếu vượt
   context window), không giới hạn top-k như QA.

2. run_quiz_summary(state) -> GraphState: node dùng trong LangGraph (Phase
   5) cho nhánh quiz/summary khi kích hoạt TỪ CHAT — tái dùng
   state["retrieved_chunks"] đã có sẵn (không retrieve lại), sinh nhanh 1
   batch quiz hoặc tóm tắt gọn dựa trên phạm vi đã truy xuất trong hội
   thoại hiện tại (không phải toàn bộ tài liệu — xem ghi chú tương tự
   trong app/agents/retrieval_agent.py).

--------------------------------------------------------------------------
SỬA SIGNATURE SO VỚI STUB GỐC — CẦN THIẾT, KHÔNG PHẢI TÙY CHỌN:
Stub gốc: `generate_quiz(document_ids, num_questions, prioritize_weak_topics)`
— THIẾU `db` và `user_id`. Không có 2 tham số này thì KHÔNG THỂ:
  - gọi get_weak_topics(db, user_id, ...) để ưu tiên topic yếu (chính yêu
    cầu của tham số `prioritize_weak_topics`!),
  - query QuizAttempt cũ theo user để tránh trùng lặp câu hỏi (yêu cầu rõ
    trong đề bài).
Đã thêm `db: Session` và `user_id: str` làm 2 tham số ĐẦU TIÊN, bắt buộc.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.llm.utils import LLMCallError, acall_llm_text, acall_structured
from app.retrieval.helpers import (
    aget_all_chunks_for_documents,
    format_chunks_for_prompt,
    normalize_chunk,
    retrieve_top_k,
)
from app.agents.state import GraphState
from app.memory.mastery_tracker import get_weak_topics
from app.models.db_models import Document, QuizAttempt
from app.models.schemas import QuizQuestion
from app.utils.tokens import count_tokens

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TOP_K = 6
EXISTING_QUESTIONS_LIMIT = 30     # giới hạn số câu cũ đưa vào prompt chống trùng, tránh prompt phình to
MAP_BATCH_TOKEN_LIMIT = 3000      # mỗi batch trong MAP phase của map-reduce tóm tắt
SINGLE_CALL_TOKEN_LIMIT = 6000    # dưới ngưỡng này -> tóm tắt 1 lần, không cần map-reduce

_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


class QuizBatch(BaseModel):
    questions: list[QuizQuestion]


_QUIZ_SYSTEM_PROMPT = """Bạn là giáo viên ra đề trắc nghiệm cho sinh viên,
CHỈ dựa vào context được cung cấp. Mỗi câu hỏi có:
- 4 lựa chọn, CHỈ 1 đáp án đúng.
- `correct_choice` PHẢI khớp NGUYÊN VĂN 1 trong 4 phần tử của `choices`.
- `explanation`: giải thích TẠI SAO đáp án đó đúng, dựa vào context.
- `difficulty`: "easy"/"medium"/"hard", ĐỘ KHÓ TĂNG DẦN qua các câu.
- `topic`: dùng ĐÚNG chuỗi chủ đề được cho trong prompt, không tự đổi.

TUYỆT ĐỐI KHÔNG lặp lại (dù diễn đạt khác đi) các câu đã liệt kê ở mục
"Câu hỏi cũ cần tránh trùng" nếu có."""

_SUMMARY_SYSTEM_PROMPT = """Bạn là trợ lý tóm tắt tài liệu học thuật.
Tóm tắt CHỈ dựa vào context được cung cấp, giữ CẤU TRÚC PHÂN CẤP theo
heading/mục nếu context có chỉ ra (dùng markdown ## cho mục lớn, - cho ý
con). KHÔNG chỉ nối câu — phải tổng hợp, cô đọng, giữ đúng thông tin quan
trọng nhất (định nghĩa, công thức, kết luận chính)."""

_REDUCE_SYSTEM_PROMPT = """Bạn nhận được nhiều bản tóm tắt TỪNG PHẦN của
CÙNG 1 tài liệu (do tài liệu quá dài phải tóm tắt theo từng đoạn riêng).
Nhiệm vụ: GỘP LẠI thành 1 bản tóm tắt THỐNG NHẤT, có cấu trúc phân cấp rõ
ràng (markdown ## cho mục lớn), loại bỏ trùng lặp giữa các phần, giữ mạch
logic xuyên suốt thay vì liệt kê rời rạc từng phần."""


# ---------------------------------------------------------------------------
# Core dùng chung: sinh quiz từ 1 tập chunk cho trước
# ---------------------------------------------------------------------------
async def _generate_quiz_questions(
    chunks: list[dict[str, Any]], topic_label: str, n: int, existing_questions: list[str]
) -> list[QuizQuestion]:
    if not chunks or n <= 0:
        return []

    context_block = format_chunks_for_prompt(chunks)
    avoid_block = ""
    if existing_questions:
        capped = existing_questions[-EXISTING_QUESTIONS_LIMIT:]
        avoid_block = "\n\nCâu hỏi cũ cần tránh trùng:\n" + "\n".join(f"- {q}" for q in capped)

    user_prompt = f"Chủ đề: {topic_label}\nSố câu cần sinh: {n}\n\nContext:\n{context_block}{avoid_block}"

    try:
        batch = await acall_structured("quiz", _QUIZ_SYSTEM_PROMPT, user_prompt, QuizBatch)
    except LLMCallError as exc:
        logger.error("_generate_quiz_questions(topic='%s'): thất bại (%s).", topic_label, exc)
        return []

    valid_questions = []
    for q in batch.questions:
        if q.correct_choice not in q.choices:
            logger.warning(
                "_generate_quiz_questions: bỏ câu hỏi vì correct_choice không khớp choices: %r",
                q.question[:80],
            )
            continue
        # Ép về ĐÚNG topic_label đã dùng để retrieve — QUAN TRỌNG: sau này
        # user trả lời, mastery_tracker.record_quiz_result(topic=...) phải
        # khớp CHÍNH XÁC chuỗi topic đã dùng ở get_weak_topics(), nếu để
        # LLM tự đặt tên (có thể lệch chính tả/cách viết hoa) sẽ tạo ra
        # MasteryRecord mới thay vì cập nhật đúng record cũ, làm sai cả
        # vòng lặp spaced repetition.
        q.topic = topic_label
        valid_questions.append(q)

    return valid_questions[:n]


def _sort_by_difficulty(questions: list[QuizQuestion]) -> list[QuizQuestion]:
    return sorted(questions, key=lambda q: _DIFFICULTY_ORDER.get(q.difficulty, 1))


def _render_quiz_markdown(questions: list[QuizQuestion]) -> str:
    if not questions:
        return "Xin lỗi, hệ thống chưa tạo được câu hỏi nào — có thể do thiếu context phù hợp."
    lines = ["# Câu hỏi ôn tập\n"]
    for i, q in enumerate(questions, start=1):
        lines.append(f"**Câu {i} ({q.difficulty}) — {q.topic}**")
        lines.append(q.question)
        for choice in q.choices:
            lines.append(f"- {choice}")
        lines.append(f"\n*Đáp án đúng: {q.correct_choice}*")
        lines.append(f"*Giải thích: {q.explanation}*\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core dùng chung: tóm tắt 1 tập chunk (map-reduce nếu vượt token limit)
# ---------------------------------------------------------------------------
def _pack_chunks_by_token_budget(chunks: list[dict[str, Any]], max_tokens: int) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for c in chunks:
        t = count_tokens(c.get("text", ""))
        if current and current_tokens + t > max_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(c)
        current_tokens += t
    if current:
        batches.append(current)
    return batches


async def _summarize_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "(Không có nội dung để tóm tắt.)"

    total_tokens = sum(count_tokens(c.get("text", "")) for c in chunks)

    if total_tokens <= SINGLE_CALL_TOKEN_LIMIT:
        context_block = format_chunks_for_prompt(chunks)
        try:
            return await acall_llm_text("summary", _SUMMARY_SYSTEM_PROMPT, f"Context:\n{context_block}")
        except LLMCallError as exc:
            logger.error("_summarize_chunks: thất bại (%s).", exc)
            return "Xin lỗi, hệ thống gặp lỗi khi tóm tắt nội dung này."

    # MAP: tóm tắt từng batch độc lập (chunk lớn hơn context window 1 lần gọi)
    batches = _pack_chunks_by_token_budget(chunks, MAP_BATCH_TOKEN_LIMIT)
    logger.info(
        "_summarize_chunks: %d token vượt ngưỡng %d -> map-reduce qua %d batch.",
        total_tokens, SINGLE_CALL_TOKEN_LIMIT, len(batches),
    )

    partial_summaries = []
    for i, batch in enumerate(batches, start=1):
        context_block = format_chunks_for_prompt(batch)
        try:
            partial = await acall_llm_text("summary", _SUMMARY_SYSTEM_PROMPT, f"Context:\n{context_block}")
        except LLMCallError as exc:
            logger.error("_summarize_chunks: batch %d/%d thất bại (%s).", i, len(batches), exc)
            partial = f"(Lỗi hệ thống khi tóm tắt phần {i} — bỏ qua phần này.)"
        partial_summaries.append(partial)

    # REDUCE: gộp các bản tóm tắt từng phần thành 1 bản thống nhất
    combined = "\n\n---\n\n".join(f"[Phần {i}]\n{s}" for i, s in enumerate(partial_summaries, start=1))
    try:
        return await acall_llm_text("summary", _REDUCE_SYSTEM_PROMPT, combined)
    except LLMCallError as exc:
        logger.error("_summarize_chunks: reduce phase thất bại (%s) — trả về nối thô các phần.", exc)
        return combined


# ---------------------------------------------------------------------------
# Public API #1: generate_quiz (REST-facing, tự retrieve theo document_ids)
# ---------------------------------------------------------------------------
async def generate_quiz(
    db: Session,
    user_id: str,
    document_ids: list[str],
    num_questions: int = 5,
    prioritize_weak_topics: bool = True,
    topic_prompt: str | None = None,
    space_id: str | None = None,
) -> list[QuizQuestion]:
    """
    Sinh `num_questions` câu trắc nghiệm (tối đa 20 câu).
    - Nếu `topic_prompt` được cung cấp: tập trung sinh câu hỏi xoay quanh chủ đề con cụ thể do người dùng yêu cầu.
    - Nếu `prioritize_weak_topics` và không có `topic_prompt`: ưu tiên các chủ đề yếu/quá hạn ôn tập (SM-2).
    - Tự động lưu `space_id` vào QuizAttempt để theo dõi lịch sử bài làm của Study Space.
    """
    # Ràng buộc số lượng câu hỏi từ 1 đến tối đa 20 câu (Góp ý #9)
    num_questions = max(1, min(int(num_questions or 5), 20))

    topic_plan: list[str] = []
    if topic_prompt and topic_prompt.strip():
        topic_plan = [topic_prompt.strip()]
    elif prioritize_weak_topics:
        weak = get_weak_topics(db, user_id, limit=min(num_questions, 8))
        topic_plan = [t.topic for t in weak]

    if not topic_plan:
        # Không có topic yếu (user mới, hoặc prioritize_weak_topics=False)
        # -> chế độ tổng hợp: dùng subject của tài liệu (nếu có) làm "topic".
        docs = db.query(Document).filter(Document.id.in_(document_ids)).all() if document_ids else []
        subjects = {d.subject for d in docs if d.subject}
        topic_plan = sorted(subjects) or ["Tổng hợp kiến thức"]

    n_topics = len(topic_plan)
    base, remainder = divmod(num_questions, n_topics)
    # Topic đầu tiên nhận phần dư — topic CẦN CHÚ Ý NHẤT thì được ra nhiều câu hỏi nhất.
    counts = [base + (1 if i < remainder else 0) for i in range(n_topics)]

    all_questions: list[QuizQuestion] = []
    for topic, n in zip(topic_plan, counts):
        if n <= 0:
            continue

        # Top-k retrieve: mở rộng theo số lượng câu hỏi cần sinh
        retrieval_k = max(DEFAULT_CONTEXT_TOP_K, n * 2)
        hits = await retrieve_top_k(topic, top_k=retrieval_k, document_ids=document_ids or None)
        chunks = [normalize_chunk(h) for h in hits]

        existing = [
            row[0]
            for row in db.query(QuizAttempt.question)
            .filter(QuizAttempt.user_id == user_id, QuizAttempt.topic == topic)
            .all()
        ]

        questions = await _generate_quiz_questions(chunks, topic, n, existing)

        for q in questions:
            db.add(
                QuizAttempt(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    space_id=space_id,
                    topic=q.topic,
                    question=q.question,
                    choices=q.choices,
                    correct_choice=q.correct_choice,
                    user_choice=None,
                    is_correct=None,
                )
            )
        all_questions.extend(questions)

    db.commit()

    result = _sort_by_difficulty(all_questions)[:num_questions]
    logger.info(
        "generate_quiz(user=%s, space=%s): sinh %d/%d câu qua %d topic %s (topic_prompt=%s).",
        user_id, space_id, len(result), num_questions, n_topics, topic_plan, topic_prompt,
    )
    return result


# ---------------------------------------------------------------------------
# Public API #2: generate_summary (REST-facing, full-scope + map-reduce)
# ---------------------------------------------------------------------------
async def generate_summary(db: Session, document_ids: list[str]) -> str:
    """
    Tóm tắt TOÀN VĂN các tài liệu trong `document_ids` (không giới hạn
    top-k như QA) — lấy TOÀN BỘ chunk của từng tài liệu qua
    `get_all_chunks_for_documents`, map-reduce qua strong model nếu vượt
    context window. Trả về markdown có cấu trúc theo từng tài liệu.
    """
    docs = db.query(Document).filter(Document.id.in_(document_ids)).all()
    if not docs:
        return "Không tìm thấy tài liệu nào để tóm tắt."

    sections = []
    for doc in docs:
        try:
            raw_chunks = await aget_all_chunks_for_documents([doc.id])
        except Exception as exc:
            logger.error("generate_summary: lấy chunk cho document '%s' lỗi: %s", doc.id, exc)
            raw_chunks = []

        chunks = [normalize_chunk(c) for c in raw_chunks]
        summary = await _summarize_chunks(chunks)
        sections.append(f"## {doc.filename}\n\n{summary}")

    header = f"# Tóm tắt {len(docs)} tài liệu\n" if len(docs) > 1 else "# Tóm tắt tài liệu\n"
    return header + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Graph node (Phase 5): tái dùng state["retrieved_chunks"], không retrieve lại
# ---------------------------------------------------------------------------
async def run_quiz_summary(state: GraphState) -> GraphState:
    """
    Node LangGraph cho nhánh quiz/summary khi kích hoạt TỪ CHAT. Khác với
    generate_quiz()/generate_summary(): dùng state["retrieved_chunks"] đã
    có sẵn (do retrieval node truy xuất trước đó trong graph) thay vì tự
    retrieve lại theo document_ids — nhanh hơn, đúng phạm vi hội thoại
    hiện tại thay vì toàn bộ tài liệu.

    Không log QuizAttempt vào DB ở đây (không có `db`/`user_id` trong
    GraphState) — nếu muốn quiz sinh từ chat cũng được tính vào mastery
    tracking, cần bổ sung `db`/`user_id` vào GraphState hoặc để lớp gọi
    run_graph() (app/graph/build_graph.py, Phase 5) tự log sau khi nhận
    kết quả trả về.
    """
    intent = state.get("intent")
    chunks = state.get("retrieved_chunks") or []

    if intent == "quiz":
        topic_label = state.get("suggested_review_topic") or "Tổng hợp"
        questions = await _generate_quiz_questions(chunks, topic_label, n=5, existing_questions=[])
        return {**state, "final_answer": _render_quiz_markdown(questions), "citations": []}

    if intent == "summary":
        if not chunks and state.get("document_ids"):
            try:
                from app.retrieval.helpers import aget_all_chunks_for_documents
                raw_fallback = await aget_all_chunks_for_documents(state["document_ids"])
                chunks = [normalize_chunk(c) for c in raw_fallback[:FINAL_TOP_K_BROAD]]
            except Exception as exc:
                logger.error("run_quiz_summary: fallback retrieval thất bại: %s", exc)

        summary = await _summarize_chunks(chunks)
        citations = [
            {"file": c.get("source_file"), "page": c.get("page"), "section": c.get("section")} for c in chunks
        ]
        return {**state, "final_answer": summary, "citations": citations}

    logger.warning("run_quiz_summary: intent '%s' không hợp lệ cho node này.", intent)
    return {**state, "final_answer": "", "citations": []}