"""
app/agents/orchestrator_agent.py

Orchestrator Agent — node đầu tiên trong graph (Phase 5). Phân loại intent
(qa/teaching/quiz/summary) bằng cheap LLM + structured output, và chủ động
gợi ý ôn tập chủ đề yếu nếu câu hỏi của người dùng chung chung.

Quyết định thiết kế:
- Dùng task="rewrite" (không phải "qa") cho việc phân loại intent, dù đề
  bài cho phép cả 2. Lý do: TASK_TEMPERATURE_MAP trong router.py gán
  temperature=0.1 cho "rewrite" nhưng 0.5 cho "qa" — phân loại là tác vụ
  CẦN xác định (deterministic: cùng 1 câu hỏi phải luôn ra cùng 1 nhãn),
  temperature thấp cho kết quả ổn định hơn giữa các lần gọi. Task "grade"
  (temp 0.0) còn lý tưởng hơn nữa nhưng đề bài chỉ cho phép rewrite/qa cho
  việc này nên chọn cái gần nhất.
- Gộp 2 tín hiệu (intent + is_vague_or_generic) vào CÙNG 1 lời gọi LLM
  (1 lần structured output) thay vì 2 lời gọi riêng — giảm latency/chi phí
  cho node chạy ở đầu MỌI request.
- Chỉ gợi ý ôn tập khi topic ĐÃ QUÁ HẠN thật sự (next_review_due_at <=
  hiện tại), không phải chỉ vì "điểm thấp" — đúng tinh thần đề bài "có chủ
  đề yếu đang ĐẾN HẠN ôn tập", tránh làm phiền user với gợi ý không đúng lúc.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.llm.utils import LLMCallError, acall_structured
from app.agents.state import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Bạn là bộ phân loại ý định (intent classifier) cho trợ lý học tập AI.

Phân loại tin nhắn của sinh viên vào ĐÚNG 1 trong 4 nhãn:

- "qa": Hỏi đáp trực tiếp 1 câu hỏi kiến thức cụ thể, muốn câu trả lời
  ngắn gọn, chính xác. Ví dụ: "Đạo hàm của sin(x) là gì?", "Định lý Bayes
  phát biểu thế nào?"
- "teaching": Muốn được GIẢNG GIẢI/DẠY một khái niệm từ đầu, cần hiểu sâu
  bản chất, không chỉ 1 câu trả lời ngắn. Ví dụ: "Giải thích cho tôi về
  tích phân từng phần", "Dạy tôi về entropy trong nhiệt động lực học".
- "quiz": Muốn LUYỆN TẬP, làm bài kiểm tra, câu hỏi trắc nghiệm. Ví dụ:
  "Cho tôi vài câu trắc nghiệm về đạo hàm", "Kiểm tra kiến thức chương 1
  giúp tôi".
- "summary": Muốn TÓM TẮT tài liệu/chương/nhiều phần. Ví dụ: "Tóm tắt
  chương 2 giúp tôi", "Tổng hợp lại toàn bộ tài liệu".

Đồng thời đánh giá `is_vague_or_generic`: true nếu tin nhắn KHÔNG nhắm tới
1 khái niệm/chủ đề cụ thể nào (vd "tôi nên học gì tiếp theo?", "giúp tôi
ôn tập", "có gì cần làm không?", chào hỏi chung chung); false nếu tin
nhắn đã nhắm rõ vào 1 khái niệm/chủ đề cụ thể.

Nếu có tóm tắt hội thoại trước đó, dùng nó để hiểu ngữ cảnh (vd "cho tôi
quiz về phần đó" sau khi vừa học 1 khái niệm -> intent="quiz")."""


class IntentClassification(BaseModel):
    intent: Literal["qa", "teaching", "quiz", "summary"] = Field(
        description="Nhãn intent phù hợp nhất với tin nhắn của người dùng"
    )
    is_vague_or_generic: bool = Field(
        description="True nếu tin nhắn không nhắm tới 1 chủ đề cụ thể nào"
    )


def _parse_due_date(value) -> datetime | None:
    """mastery_context có thể tới từ state đã qua serialize JSON (API/graph
    checkpoint) nên next_review_due_at có thể là str ISO thay vì datetime —
    xử lý cả 2 dạng để không vỡ khi ranh giới serialize thay đổi."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _pick_review_topic(mastery_context: dict) -> str | None:
    """Chọn topic nên gợi ý ôn tập: topic ĐẦU TIÊN trong
    mastery_context["weak_topics"] (list dict giống MasteryTopic — xem
    app/memory/mastery_tracker.get_weak_topics, kết quả ĐÃ được sort ưu
    tiên quá hạn lâu nhất) MÀ THỰC SỰ đã quá hạn ôn tập. Nếu topic yếu
    nhất vẫn CHƯA tới hạn (vd mới học, chỉ đơn thuần điểm thấp), KHÔNG gợi
    ý — tránh nhắc ôn tập chưa đúng lúc."""
    weak_topics = mastery_context.get("weak_topics") or []
    now = datetime.utcnow()
    for t in weak_topics:
        topic = t.get("topic") if isinstance(t, dict) else getattr(t, "topic", None)
        due_raw = t.get("next_review_due_at") if isinstance(t, dict) else getattr(t, "next_review_due_at", None)
        due = _parse_due_date(due_raw)
        if topic and due is not None and due <= now:
            return topic
    return None


async def run_orchestrator(state: GraphState) -> GraphState:
    """
    Phân loại intent cho `state["user_query"]`, cập nhật:
        state["intent"]
        state["suggested_review_topic"]  (None nếu không có gợi ý)

    Idempotent: 2 field trên được tính lại HOÀN TOÀN từ input hiện tại mỗi
    lần gọi, không phụ thuộc giá trị cũ trong state.
    """
    user_query = (state.get("user_query") or "").strip()
    chat_history_summary = state.get("chat_history_summary") or ""
    mastery_context = state.get("mastery_context") or {}

    if not user_query:
        logger.warning("run_orchestrator: user_query rỗng — mặc định intent='qa'.")
        return {**state, "intent": "qa", "suggested_review_topic": None}

    user_prompt = f"Tin nhắn của sinh viên: {user_query}"
    if chat_history_summary.strip():
        user_prompt = f"Tóm tắt hội thoại trước đó:\n{chat_history_summary}\n\n{user_prompt}"

    try:
        result = await acall_structured("rewrite", _SYSTEM_PROMPT, user_prompt, IntentClassification)
        intent: Literal["qa", "teaching", "quiz", "summary"] = result.intent
        is_vague = result.is_vague_or_generic
    except LLMCallError as exc:
        logger.error("run_orchestrator: phân loại intent thất bại (%s) — fallback intent='qa'.", exc)
        intent = "qa"
        is_vague = False

    suggested_review_topic: str | None = None
    if is_vague:
        suggested_review_topic = _pick_review_topic(mastery_context)
        if suggested_review_topic:
            logger.info(
                "run_orchestrator: câu hỏi chung chung + có topic quá hạn -> gợi ý ôn '%s'.",
                suggested_review_topic,
            )

    return {**state, "intent": intent, "suggested_review_topic": suggested_review_topic}