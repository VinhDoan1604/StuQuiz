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

generate_quiz() cần `db` + `user_id` để ưu tiên topic yếu (get_weak_topics)
và tránh lặp lại câu hỏi user đã gặp (QuizAttempt cũ).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agents.constants import FINAL_TOP_K_BROAD
from app.agents.quiz_topics import build_section_topics, prioritize_topics
from app.agents.state import GraphState
from app.llm.utils import LLMCallError, acall_llm_text, acall_structured
from app.memory.mastery_tracker import WEAK_SCORE_THRESHOLD
from app.models.db_models import Document, MasteryRecord, ParentChunk, QuizAttempt
from app.models.schemas import QuizQuestion, QuizQuestionOut
from app.retrieval.expansion import make_snippet
from app.retrieval.helpers import (
    aget_all_chunks_for_documents,
    format_chunks_for_prompt,
    normalize_chunk,
    retrieve_top_k,
)
from app.utils.ids import new_id
from app.utils.time import utcnow
from app.utils.tokens import count_tokens

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TOP_K = 6
EXISTING_QUESTIONS_LIMIT = 30     # giới hạn số câu cũ đưa vào prompt chống trùng, tránh prompt phình to
MAP_BATCH_TOKEN_LIMIT = 3000      # mỗi batch trong MAP phase của map-reduce tóm tắt
SINGLE_CALL_TOKEN_LIMIT = 6000    # dưới ngưỡng này -> tóm tắt 1 lần, không cần map-reduce
TOPIC_CONTEXT_TOKEN_LIMIT = 2500  # context tối đa của 1 mục khi ra đề
QUIZ_LLM_CONCURRENCY = 4          # số mục sinh đề song song

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
        if (
            len(q.choices) < 2
            or len({c.strip().lower() for c in q.choices}) != len(q.choices)   # đáp án trùng nhau
            or q.correct_choice not in q.choices
        ):
            logger.warning("_generate_quiz_questions: bỏ câu hỏi lỗi (đáp án trùng/không khớp): %r", q.question[:80])
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
def _load_section_sources(db: Session, document_ids: list[str]) -> list[dict[str, Any]]:
    """Nội dung theo mục của các tài liệu: parent chunk (DB) — đủ cả section, không cần vector."""
    rows = (
        db.query(ParentChunk)
        .filter(ParentChunk.doc_id.in_(document_ids))
        .order_by(ParentChunk.doc_id, ParentChunk.page, ParentChunk.section)
        .all()
    )
    return [{"doc_id": r.doc_id, "section": r.section, "text": r.text, "page": r.page} for r in rows]


async def _load_section_sources_from_index(document_ids: list[str]) -> list[dict[str, Any]]:
    """Dự phòng khi tài liệu không có parent chunk: dùng child chunk trong Qdrant."""
    try:
        raw = await aget_all_chunks_for_documents(document_ids)
    except Exception as exc:
        logger.warning("generate_quiz: không lấy được chunk từ Qdrant (%s).", exc)
        return []
    return [
        {"doc_id": c.get("doc_id"), "section": c.get("section"), "text": c.get("text", ""), "page": c.get("page")}
        for c in raw
    ]


def _plan_topics(
    db: Session, user_id: str, document_ids: list[str], sources: list[dict[str, Any]],
    num_questions: int, prioritize_weak_topics: bool,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Chọn tối đa `num_questions` mục để ra đề, kèm nội dung của từng mục."""
    docs = db.query(Document).filter(Document.id.in_(document_ids)).all()
    doc_names = {str(d.id): d.filename for d in docs}
    topics = build_section_topics(sources, doc_names)
    if not topics:
        return []
    if prioritize_weak_topics:
        records = (
            db.query(MasteryRecord)
            .filter(MasteryRecord.user_id == user_id, MasteryRecord.topic.in_([t.label for t in topics]))
            .all()
        )
        topics = prioritize_topics(topics, {r.topic: r for r in records}, utcnow(), WEAK_SCORE_THRESHOLD)
    chosen = topics[:num_questions]
    return [
        (t.label, _pack_chunks_by_token_budget(t.chunks, TOPIC_CONTEXT_TOKEN_LIMIT)[0])
        for t in chosen
    ]


async def generate_quiz(
    db: Session,
    user_id: str,
    document_ids: list[str],
    num_questions: int = 5,
    prioritize_weak_topics: bool = True,
    topic_prompt: str | None = None,
    space_id: str | None = None,
) -> list[QuizQuestionOut]:
    """
    Sinh tối đa `num_questions` (<=20) câu trắc nghiệm, mỗi câu gắn với 1 QuizAttempt
    (attempt_id) để server tự chấm ở /quiz/answer.

    - topic_prompt: tập trung vào chủ đề người dùng gõ (context lấy bằng retrieval).
    - Không có topic_prompt: chủ đề = các MỤC của tài liệu (app/agents/quiz_topics.py),
      context = đúng nội dung mục đó. prioritize_weak_topics => ưu tiên mục quá hạn ôn,
      rồi mục chưa học, rồi mục còn yếu (chỉ xét trong các tài liệu đang chọn).
    """
    num_questions = max(1, min(int(num_questions or 5), 20))

    plan: list[tuple[str, list[dict[str, Any]]]] = []
    if topic_prompt and topic_prompt.strip():
        topic = topic_prompt.strip()
        hits = await retrieve_top_k(topic, top_k=max(DEFAULT_CONTEXT_TOP_K, num_questions * 2), document_ids=document_ids)
        plan = [(topic, [normalize_chunk(h) for h in hits])]
    elif document_ids:
        sources = _load_section_sources(db, document_ids) or await _load_section_sources_from_index(document_ids)
        plan = _plan_topics(db, user_id, document_ids, sources, num_questions, prioritize_weak_topics)

    if not plan:
        logger.warning("generate_quiz(user=%s): không có nội dung để ra đề.", user_id)
        return []

    n_topics = len(plan)
    base, remainder = divmod(num_questions, n_topics)
    counts = [base + (1 if i < remainder else 0) for i in range(n_topics)]  # mục ưu tiên nhất ra nhiều câu nhất

    # Câu cũ theo từng chủ đề (truy vấn DB tuần tự — Session không dùng chung giữa các coroutine)
    jobs = []
    for (topic, chunks), n in zip(plan, counts):
        existing = [
            row[0]
            for row in db.query(QuizAttempt.question)
            .filter(QuizAttempt.user_id == user_id, QuizAttempt.topic == topic)
            .all()
        ]
        jobs.append((chunks, topic, n, existing))

    semaphore = asyncio.Semaphore(QUIZ_LLM_CONCURRENCY)

    async def _run(job):
        async with semaphore:
            return await _generate_quiz_questions(*job)

    all_questions: list[QuizQuestion] = [q for batch in await asyncio.gather(*(_run(j) for j in jobs)) for q in batch]
    topic_plan = [t for t, _ in plan]

    selected = _sort_by_difficulty(all_questions)[:num_questions]

    outputs: list[QuizQuestionOut] = []
    for q in selected:
        attempt = QuizAttempt(
            id=new_id(),
            user_id=user_id,
            space_id=space_id,
            topic=q.topic,
            question=q.question,
            choices=q.choices,
            correct_choice=q.correct_choice,
            explanation=q.explanation,
            difficulty=q.difficulty,
            user_choice=None,
            is_correct=None,
        )
        db.add(attempt)
        outputs.append(
            QuizQuestionOut(
                attempt_id=attempt.id,
                question=q.question,
                choices=q.choices,
                topic=q.topic,
                difficulty=q.difficulty,
            )
        )
    db.commit()

    logger.info(
        "generate_quiz(user=%s, space=%s): %d/%d câu qua %d topic %s.",
        user_id, space_id, len(outputs), num_questions, n_topics, topic_plan,
    )
    return outputs


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
                raw_fallback = await aget_all_chunks_for_documents(state["document_ids"])
                chunks = [normalize_chunk(c) for c in raw_fallback[:FINAL_TOP_K_BROAD]]
            except Exception as exc:
                logger.error("run_quiz_summary: fallback retrieval thất bại: %s", exc)

        summary = await _summarize_chunks(chunks)
        citations = [
            {
                "index": i,
                "file": c.get("source_file"),
                "page": c.get("page"),
                "section": c.get("section"),
                "snippet": c.get("snippet") or make_snippet(c.get("text", "")),
            }
            for i, c in enumerate(chunks, start=1)
        ]
        return {**state, "final_answer": summary, "citations": citations}

    logger.warning("run_quiz_summary: intent '%s' không hợp lệ cho node này.", intent)
    return {**state, "final_answer": "", "citations": []}
