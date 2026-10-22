"""
app/agents/generation_agent.py

Generation Agent — 2 mode dùng chung 1 node, khác prompt/model:
- QA mode: trả lời ngắn gọn, trích dẫn CHÍNH XÁC theo từng ý (đánh số
  [1][2] khớp với chunk được liệt kê trong prompt), dùng task="qa".
- Teaching mode: giải thích theo scaffolding (Khái niệm -> Bản chất ->
  Ví dụ minh hoạ) + 1 câu hỏi Socratic nhẹ ở cuối để kiểm tra hiểu bài,
  LUÔN dùng task="teaching" (strong model).

Nhận feedback từ lần chấm answer trước (Reflection Agent) qua
state["reflection_history"] — nếu có, prompt sẽ chèn RÕ feedback đó và
yêu cầu sửa cụ thể, không chỉ lặp lại cùng 1 prompt (tránh LLM trả lời y
hệt lần trước dù bị chấm rớt).

Output: LUÔN ghi state["answer_draft"] + state["citations"] (ứng viên).
KHÔNG ghi state["final_answer"] — việc "chấp nhận" bản draft này thành câu
trả lời cuối cùng là quyết định của graph (app/graph/build_graph.py, Phase
5) sau khi grade_answer() thông qua, đúng theo docstring gốc của stub.
"""
from __future__ import annotations

import logging
import re

from app.agents.constants import ANSWER_ACCEPT_THRESHOLD
from app.llm.utils import LLMCallError, acall_llm_text
from app.retrieval.helpers import format_chunks_for_prompt
from app.agents.state import GraphState
from app.retrieval.expansion import make_snippet

logger = logging.getLogger(__name__)

_QA_SYSTEM_PROMPT = """Bạn là trợ lý học tập và nghiên cứu học thuật StudyRAG, trả lời câu hỏi CHỈ dựa vào context được cung cấp bên dưới (mỗi đoạn context có đánh số [1], [2], ...).

QUY TẮC TRÌNH BÀY & ĐỊNH DẠNG (BẮT BUỘC):
1. **Không viết thành một khối văn bản dài dồn cục**: Tuyệt đối không viết 1 đoạn văn dày đặc, khó đọc. Phải chia câu trả lời thành các mục phân cấp rõ ràng bằng Markdown.
2. **In đậm từ khóa quan trọng**: BẮT BUỘC in đậm (`**thuật ngữ**`, **khái niệm then chốt**, **công thức**, **kết luận quan trọng**) để người học có thể đọc lướt (scan nhanh) nắm được ngay ý chính.
3. **Cấu trúc câu trả lời gồm 3 phần rõ ràng**:
   - `### 🎯 Trả lời trực tiếp`: 1-2 câu trả lời thẳng, cô đọng nhất vào trọng tâm câu hỏi.
   - `### 📌 Chi tiết & Cơ chế cốt lõi`: Dùng danh sách gạch đầu dòng (`- `) để phân tích từng luận điểm, mỗi gạch đầu dòng bắt đầu bằng từ khóa in đậm.
   - `### 💡 Điểm cần lưu ý / Ứng dụng`: Nêu ngoại lệ, lưu ý thực tế hoặc ví dụ ngắn gọn (nếu phù hợp).
4. **Trích dẫn chính xác**: Mỗi ý lấy từ context nào, chèn số tương ứng ngay sau câu, ví dụ: "Thuật toán có độ phức tạp là **O(V log V + E)** [1]." CHỈ dùng số đã được cấp, không tự bịa số mới.
5. **Trung thực tuyệt đối**: Nếu context không đủ để trả lời, phải nêu rõ ràng (ví dụ: "Tài liệu chưa cung cấp dữ kiện về..."), tuyệt đối không suy đoán hay bịa đặt ngoài tài liệu."""

_TEACHING_SYSTEM_PROMPT = """Bạn là giảng viên / gia sư AI StudyRAG, giảng giải kiến thức cho sinh viên theo phương pháp SCAFFOLDING (phân tầng sư phạm), CHỈ dựa vào context được cung cấp (mỗi đoạn có đánh số [1], [2], ...).

QUY TẮC TRÌNH BÀY & ĐỊNH DẠNG (BẮT BUỘC):
1. **Trình bày thoáng mắt, chia nhỏ mục**: Tuyệt đối không viết thành các đoạn văn dài dồn cục. Sử dụng tiêu đề Markdown (`###`), gạch đầu dòng (`- `) và khoảng trắng phân cách hợp lý.
2. **In đậm từ khóa trọng tâm**: BẮT BUỘC in đậm (`**khái niệm**`, **tính chất**, **bước tính toán**) để làm nổi bật các điểm sinh viên cần nhớ.
3. **Cấu trúc bài giảng chuẩn 4 phần**:
   - `### 1. 🎯 Khái niệm cốt lõi`: Định nghĩa súc tích, dễ hiểu nhất.
   - `### 2. 🔬 Bản chất & Nguyên lý hoạt động`: Phân tích sâu tại sao và cơ chế vận hành bên trong — giúp người học hiểu gốc rễ vấn đề.
   - `### 3. 📝 Ví dụ minh hoạ thực tế`: 1 ví dụ cụ thể kèm các bước suy luận / tính toán từng bước rõ ràng.
   - `### 4. 🤔 Câu hỏi suy ngẫm (Socratic Question)`: Kết thúc bằng ĐÚNG 1 câu hỏi gợi mở để sinh viên tự chiêm nghiệm, không tự trả lời ngay.
4. **Trích dẫn**: Khi dùng thông tin từ context nào, chèn số tương ứng như [1]. Nếu tài liệu thiếu ý nào, phải nói rõ phần đó chưa được đề cập thay vì tự suy diễn."""

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

_ERROR_FALLBACK_ANSWER = "Xin lỗi, hệ thống đang gặp sự cố khi tạo câu trả lời. Vui lòng thử lại sau ít phút."


def _last_answer_feedback(state: GraphState) -> str | None:
    """Tìm feedback GẦN NHẤT từ 1 lần grade_answer báo 'chưa đạt' — cho
    biết đây có phải lượt regenerate theo yêu cầu Reflection Agent hay
    không. Dùng chung ANSWER_ACCEPT_THRESHOLD với reflection_agent.py để
    2 bên luôn đồng bộ về "thế nào là đạt"."""
    history = state.get("reflection_history") or []
    for entry in reversed(history):
        if "answer_faithful" in entry:
            grade = entry.get("answer_grade")
            failed = entry.get("answer_faithful") is False or (
                grade is not None and grade < ANSWER_ACCEPT_THRESHOLD
            )
            return entry.get("feedback") if failed else None
    return None


def _extract_cited_indices(answer_text: str) -> set[int]:
    return {int(m) for m in _CITATION_PATTERN.findall(answer_text)}


def _citation(i: int, c: dict) -> dict:
    # `index` = số [n] ĐÚNG như LLM đã trích trong câu trả lời (trước đây UI đánh số lại
    # theo vị trí trong danh sách đã lọc => [3] trong text hiện thành "[1]" ở chip).
    return {
        "index": i,
        "file": c.get("source_file"),
        "page": c.get("page"),
        "section": c.get("section"),
        "snippet": c.get("snippet") or make_snippet(c.get("text", "")),
    }


def _build_qa_citations(chunks: list[dict], answer_text: str) -> list[dict]:
    cited = _extract_cited_indices(answer_text)
    return [_citation(i, c) for i, c in enumerate(chunks, start=1) if i in cited]


def _build_teaching_citations(chunks: list[dict]) -> list[dict]:
    return [_citation(i, c) for i, c in enumerate(chunks, start=1)]

async def run_generation(state: GraphState) -> GraphState:
    intent = state.get("intent", "qa")
    rewritten = state.get("rewritten_queries") or []
    query = rewritten[-1] if rewritten else state.get("user_query", "")
    chunks = state.get("retrieved_chunks") or []
    context_block = format_chunks_for_prompt(chunks)
    feedback = _last_answer_feedback(state)

    if intent == "teaching":
        task, system_prompt = "teaching", _TEACHING_SYSTEM_PROMPT
    else:
        task, system_prompt = "qa", _QA_SYSTEM_PROMPT

    user_prompt = f"Câu hỏi: {query}\n\nContext:\n{context_block}"
    if feedback:
        user_prompt += (
            f"\n\n--- LƯU Ý QUAN TRỌNG ---\nBài trả lời TRƯỚC của bạn có vấn đề, phản hồi từ "
            f'người chấm: "{feedback}". Hãy viết lại, khắc phục ĐÚNG vấn đề này, bám sát context hơn.'
        )
        logger.info("run_generation: regenerate theo feedback='%s'", feedback[:100])

    try:
        answer_text = await acall_llm_text(task, system_prompt, user_prompt)
    except LLMCallError as exc:
        logger.error("run_generation: LLM thất bại (%s).", exc)
        answer_text = _ERROR_FALLBACK_ANSWER

    if answer_text == _ERROR_FALLBACK_ANSWER:
        citations: list[dict] = []
    elif intent == "teaching":
        citations = _build_teaching_citations(chunks)
    else:
        citations = _build_qa_citations(chunks, answer_text)

    return {**state, "answer_draft": answer_text, "citations": citations}