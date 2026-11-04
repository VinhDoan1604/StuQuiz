"""
app/retrieval/query_processing.py

Query rewriting + decomposition cho Retrieval Agent, chạy bằng "cheap" LLM tier
(task "rewrite"/"decompose" trong app/llm/router.py) qua helper async dùng chung
app/llm/utils.py.

Cả 2 hàm có heuristic tiền kiểm tra để TRÁNH gọi LLM khi không cần — đây là
bước chạy trên gần như mọi request retrieval:
- rewrite_query: chỉ gọi LLM nếu câu hỏi có tham chiếu mơ hồ (đại từ,
  "phần trước"...) VÀ có chat_history_summary để resolve; ngược lại trả về
  nguyên văn câu hỏi.
- decompose_query: chỉ gọi LLM nếu câu hỏi có dấu hiệu multi-hop (so sánh,
  nhiều mệnh đề nối bằng "và"...); ngược lại trả về [query].

Mọi lỗi LLM đều fail-open về câu hỏi gốc (không làm hỏng retrieval).
"""
from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from app.llm.utils import LLMCallError, acall_llm_text, acall_structured

logger = logging.getLogger(__name__)

_REFERENTIAL_PATTERNS = re.compile(
    r"\b(nó|cái đó|cái này|điều đó|điều này|phần trước|phần trên|ở trên|"
    r"như trên|vừa nói|vừa rồi|định lý đó|công thức đó|ví dụ đó|khái niệm đó)\b",
    re.IGNORECASE,
)

_MULTIHOP_PATTERNS = re.compile(
    r"\b(so sánh|khác nhau|giống nhau|vừa .+ vừa|liên hệ giữa|mối quan hệ giữa|"
    r"và cho ví dụ|rồi giải thích|và giải thích|kết hợp)\b",
    re.IGNORECASE,
)

_MAX_SUBQUERIES = 4

_REWRITE_SYSTEM_PROMPT = (
    "Bạn là trợ lý viết lại câu hỏi cho hệ thống RAG hỗ trợ học tập. "
    "Nhiệm vụ: dựa vào tóm tắt hội thoại trước đó, viết lại câu hỏi của "
    "người dùng thành một câu hỏi ĐẦY ĐỦ NGỮ CẢNH, thay thế mọi đại từ/tham "
    "chiếu mơ hồ (nó, cái đó, phần trước...) bằng danh từ cụ thể được nhắc "
    "tới trong tóm tắt. CHỈ trả về câu hỏi đã viết lại, không thêm giải "
    "thích, không thêm dấu ngoặc kép."
)

_DECOMPOSE_SYSTEM_PROMPT = (
    "Bạn là trợ lý phân rã câu hỏi cho hệ thống RAG hỗ trợ học tập. "
    "Nếu câu hỏi của người dùng chứa nhiều ý độc lập (multi-hop, so sánh "
    f"nhiều khái niệm, yêu cầu nhiều việc), hãy tách thành tối đa {_MAX_SUBQUERIES} "
    "câu hỏi con, MỖI câu hỏi con phải tự đầy đủ nghĩa (không phụ thuộc câu "
    "khác) và dùng được để tìm kiếm tài liệu độc lập. Nếu câu hỏi vốn đã đơn "
    "giản, trả về chính nó trong list 1 phần tử."
)


class _SubQueries(BaseModel):
    sub_queries: list[str] = Field(description="Các câu hỏi con tự đầy đủ nghĩa")


def _needs_rewrite(query: str, chat_history_summary: str) -> bool:
    """Chỉ đáng rewrite nếu (a) có summary để resolve tham chiếu, và
    (b) câu hỏi thực sự chứa tham chiếu mơ hồ."""
    if not chat_history_summary.strip():
        return False
    return bool(_REFERENTIAL_PATTERNS.search(query))


def _looks_multihop(query: str) -> bool:
    if _MULTIHOP_PATTERNS.search(query):
        return True
    # Câu hỏi dài + nhiều dấu phẩy cũng là dấu hiệu multi-hop dù không khớp từ khoá
    return len(query.split()) > 25 and query.count(",") >= 2


async def rewrite_query(query: str, chat_history_summary: str = "") -> str:
    """Viết lại câu hỏi mơ hồ dựa vào tóm tắt hội thoại; không cần thì trả nguyên văn."""
    query = query.strip()
    if not query or not _needs_rewrite(query, chat_history_summary):
        return query

    user_prompt = f"Tóm tắt hội thoại trước đó:\n{chat_history_summary}\n\nCâu hỏi cần viết lại:\n{query}"
    try:
        rewritten = await acall_llm_text("rewrite", _REWRITE_SYSTEM_PROMPT, user_prompt)
    except LLMCallError as exc:
        logger.warning("rewrite_query: LLM thất bại (%s) — dùng câu hỏi gốc.", exc)
        return query

    rewritten = rewritten.strip().strip('"').strip("'")
    if not rewritten:
        return query
    logger.info("rewrite_query: '%s' -> '%s'", query, rewritten)
    return rewritten


async def decompose_query(query: str) -> list[str]:
    """Tách câu hỏi multi-hop thành các sub-query độc lập; không cần thì trả [query]."""
    query = query.strip()
    if not query:
        return []
    if not _looks_multihop(query):
        return [query]

    try:
        result = await acall_structured("decompose", _DECOMPOSE_SYSTEM_PROMPT, f"Câu hỏi: {query}", _SubQueries)
    except LLMCallError as exc:
        logger.warning("decompose_query: LLM thất bại (%s) — dùng câu hỏi gốc.", exc)
        return [query]

    sub_queries = [q.strip() for q in result.sub_queries if q and q.strip()][:_MAX_SUBQUERIES]
    if not sub_queries:
        return [query]
    logger.info("decompose_query: '%s' -> %d sub-query.", query, len(sub_queries))
    return sub_queries
