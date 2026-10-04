"""
app/agents/reflection_agent.py

Reflection/Critic Agent — CRAG (Corrective RAG) + Self-RAG. Điểm nhấn kỹ
thuật quan trọng nhất của project.

Quyết định thiết kế đáng chú ý:

1. LLM-as-judge tự viết prompt, KHÔNG dùng RAGAS trực tiếp ở runtime, vì:
   - RAGAS tự gọi thêm LLM + embedding riêng cho mỗi metric -> cộng dồn
     latency/chi phí cho MỌI request chat (grade_context/grade_answer chạy
     trên hot path, có thể lặp tới max_reflection_loops lần/request).
   - RAGAS được thiết kế cho batch evaluation OFFLINE (mục 5.13), không
     phải để chấm real-time trong vòng lặp CRAG tương tác trực tiếp.
   Để 2 nơi (judge runtime ở đây và eval offline mục 5.13) không lệch
   nhau, nên DÙNG CHUNG ĐỊNH NGHĨA tiêu chí (relevance/sufficiency/
   faithfulness/completeness — xem docstring 2 hàm bên dưới) và định kỳ
   so sánh 2 bộ điểm để phát hiện lệch, thay vì ép dùng chung 1 thư viện.

2. Dùng task="grade" (temperature=0.0 trong TASK_TEMPERATURE_MAP của
   router.py) — chấm điểm cần khách quan, ổn định giữa các lần gọi cùng
   1 input, không cần sáng tạo.

3. Khi LLM/structured-output LỖI THẬT SỰ (không phải điểm thấp mà là
   exception — mạng, rate-limit, parse lỗi): FAIL-OPEN (coi như đạt) thay
   vì fail-closed. Lý do: fail-closed sẽ kích hoạt thêm 1 vòng lặp retry
   tốn token/latency cho khả năng cao là sự cố tạm thời, trong khi
   `num_reflection_loops` đã có sẵn giới hạn để chặn vòng lặp vô hạn — nên
   fail-open ở đây không tạo rủi ro nghiêm trọng, chỉ đánh đổi 1 lần bỏ
   sót chấm điểm (được log rõ để theo dõi tần suất xảy ra).

4. `num_reflection_loops` được TĂNG bởi graph (app/graph/build_graph.py,
   Phase 5) mỗi khi quay lại retrieval/generation — 2 hàm trong file này
   chỉ ĐỌC giá trị hiện tại để quyết định có "ép" pass hay không khi đã
   hết lượt, KHÔNG tự tăng biến đếm (tách đúng trách nhiệm: chấm điểm vs
   điều khiển luồng).
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.agents.constants import ANSWER_ACCEPT_THRESHOLD, CONTEXT_SUFFICIENT_THRESHOLD
from app.llm.utils import LLMCallError, acall_structured
from app.retrieval.helpers import format_chunks_for_prompt
from app.agents.state import GraphState, ReflectionResult
from app.config import settings

logger = logging.getLogger(__name__)

_CONTEXT_GRADE_SYSTEM_PROMPT = """Bạn là người chấm điểm khách quan cho hệ
thống RAG hỗ trợ học tập. Nhiệm vụ: đánh giá context truy xuất được có đủ
để trả lời câu hỏi hay không, theo 2 tiêu chí:

- relevance (0.0-1.0): context có LIÊN QUAN đến câu hỏi không.
- sufficiency (0.0-1.0): context có ĐẦY ĐỦ thông tin để trả lời TRỌN VẸN
  câu hỏi không (không chỉ liên quan mà còn phải đủ chi tiết).

Nếu sufficiency thấp, feedback PHẢI nêu CỤ THỂ đang thiếu ý/khía cạnh nào
của câu hỏi (để hệ thống viết lại câu hỏi tìm kiếm bổ sung đúng phần đó),
KHÔNG nhận xét chung chung như "chưa đủ thông tin"."""

_ANSWER_GRADE_SYSTEM_PROMPT = """Bạn là người chấm điểm khách quan cho hệ
thống RAG hỗ trợ học tập. Nhiệm vụ: đánh giá câu trả lời DỰA VÀO ĐÚNG
context đã cho, theo 2 tiêu chí:

- faithfulness (0.0-1.0): MỌI thông tin trong câu trả lời có được context
  hỗ trợ không, hay có phần BỊA ĐẶT/suy diễn ngoài context (hallucination).
  Đây là tiêu chí QUAN TRỌNG NHẤT — 1 câu bịa đặt cũng đủ để điểm thấp.
- completeness (0.0-1.0): câu trả lời có giải quyết TRỌN VẸN câu hỏi
  không, hay còn thiếu ý.

Nếu điểm thấp, feedback PHẢI nêu CỤ THỂ câu/ý nào bịa đặt hoặc câu hỏi
phần nào chưa được trả lời, KHÔNG nhận xét chung chung."""


class ContextGradeOutput(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    sufficiency: float = Field(ge=0.0, le=1.0)
    feedback: str = Field(description="Cụ thể thiếu ý gì nếu sufficiency thấp, rỗng nếu đã đủ")


class AnswerGradeOutput(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    feedback: str = Field(description="Cụ thể câu nào bịa đặt hoặc phần nào thiếu, rỗng nếu đã đạt")


def _loops_exhausted(state: GraphState) -> bool:
    return state.get("num_reflection_loops", 0) >= settings.max_reflection_loops


async def grade_context(state: GraphState) -> ReflectionResult:
    """
    Chấm độ liên quan + đầy đủ của state["retrieved_chunks"] so với câu
    hỏi hiện tại (rewritten_queries[-1] nếu có, else user_query).

    KHÔNG mutate state — caller (graph, Phase 5) chịu trách nhiệm append
    kết quả vào state["reflection_history"] và quyết định routing (loop
    lại retrieval hay đi tiếp generation).

    Khi đã hết `settings.max_reflection_loops` vòng lặp mà vẫn chưa đủ:
    LUÔN trả về context_sufficient=True (ép graph đi tiếp, tránh lặp vô
    hạn), kèm feedback đặc biệt để Generation Agent biết cần trả lời trung
    thực về phần thiếu thay vì cố "lấp đầy" bằng suy diễn — đúng yêu cầu
    "nếu hết vòng lặp mà vẫn không đủ -> trả lời tài liệu chưa đủ thông
    tin, KHÔNG bịa".
    """
    rewritten = state.get("rewritten_queries") or []
    query = rewritten[-1] if rewritten else state.get("user_query", "")
    chunks = state.get("retrieved_chunks") or []

    if not chunks:
        sufficient = _loops_exhausted(state)
        feedback = "Không truy xuất được context nào liên quan đến câu hỏi."
        if sufficient:
            feedback = f"[Đã hết số lần thử lại] {feedback} Hãy trả lời trung thực là tài liệu chưa đề cập."
        return ReflectionResult(context_grade=0.0, context_sufficient=sufficient, feedback=feedback)

    context_block = format_chunks_for_prompt(chunks)
    user_prompt = f"Câu hỏi: {query}\n\nContext:\n{context_block}"

    try:
        result = await acall_structured("grade", _CONTEXT_GRADE_SYSTEM_PROMPT, user_prompt, ContextGradeOutput)
        grade = (result.relevance + result.sufficiency) / 2
        sufficient = grade >= CONTEXT_SUFFICIENT_THRESHOLD
        feedback = result.feedback
    except LLMCallError as exc:
        logger.error("grade_context: chấm điểm thất bại (%s) — fail-open (coi như đủ).", exc)
        grade, sufficient, feedback = 1.0, True, ""

    if not sufficient and _loops_exhausted(state):
        logger.warning(
            "grade_context: hết %d vòng lặp mà context vẫn chưa đủ — ép sufficient=True để dừng loop.",
            settings.max_reflection_loops,
        )
        feedback = (
            f"[Đã hết số lần thử lại] {feedback} Hãy trả lời trung thực dựa trên context hiện có, "
            f"và nêu rõ phần nào tài liệu chưa đề cập thay vì suy diễn."
        )
        sufficient = True

    return ReflectionResult(context_grade=round(grade, 3), context_sufficient=sufficient, feedback=feedback)


async def grade_answer(state: GraphState) -> ReflectionResult:
    """
    Chấm faithfulness (chống hallucination) + completeness của
    state["answer_draft"] so với state["retrieved_chunks"] đã dùng để sinh
    câu trả lời đó. KHÔNG mutate state — xem docstring grade_context.

    answer_grade = 0.6 * faithfulness + 0.4 * completeness — faithfulness
    nặng ký hơn vì 1 câu bịa đặt nghiêm trọng hơn nhiều so với thiếu 1 ý,
    đúng tinh thần Self-RAG (ưu tiên KHÔNG hallucinate hơn là đầy đủ).
    """
    answer = state.get("answer_draft", "") or ""
    chunks = state.get("retrieved_chunks") or []

    if not answer.strip():
        return ReflectionResult(answer_grade=0.0, answer_faithful=False, feedback="Câu trả lời rỗng.")

    context_block = format_chunks_for_prompt(chunks)
    user_prompt = f"Context:\n{context_block}\n\nCâu trả lời cần chấm:\n{answer}"

    try:
        result = await acall_structured("grade", _ANSWER_GRADE_SYSTEM_PROMPT, user_prompt, AnswerGradeOutput)
        grade = 0.6 * result.faithfulness + 0.4 * result.completeness
        faithful = result.faithfulness >= ANSWER_ACCEPT_THRESHOLD
        feedback = result.feedback
    except LLMCallError as exc:
        logger.error("grade_answer: chấm điểm thất bại (%s) — fail-open (coi như đạt).", exc)
        grade, faithful, feedback = 1.0, True, ""

    if (not faithful or grade < ANSWER_ACCEPT_THRESHOLD) and _loops_exhausted(state):
        logger.warning(
            "grade_answer: hết %d vòng lặp mà câu trả lời vẫn chưa đạt — chấp nhận để dừng loop.",
            settings.max_reflection_loops,
        )
        feedback = f"[Đã hết số lần thử lại] {feedback} Chấp nhận bản trả lời hiện tại dù chưa hoàn hảo."
        faithful = True

    return ReflectionResult(answer_grade=round(grade, 3), answer_faithful=faithful, feedback=feedback)