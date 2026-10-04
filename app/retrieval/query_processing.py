"""
app/retrieval/query_processing.py

Query rewriting + decomposition (mục 5.6), chạy bằng "cheap" LLM tier qua
app/llm/router.py — dùng cho Retrieval Agent, và được gọi lại khi
Reflection Agent báo "context chưa đủ -> rewrite rồi retrieve lại" (vòng
lặp CRAG, mục 5.8).

--------------------------------------------------------------------------
ĐỐI CHIẾU VỚI app/llm/router.py THẬT (đã nhận được, không còn là giả định):

    from app.llm.router import get_llm
    llm = get_llm(task)                    # task: Literal["rewrite","decompose",
                                            #   "grade","qa","teaching","quiz","summary"]
    response = llm.invoke([("system", ...), ("human", ...)])   # LangChain tuple messages
    text = response.content                # str

router.py đã tự phân loại "rewrite" và "decompose" vào CHEAP_TASKS — nghĩa
là 2 hàm trong file này ĐÚNG LÀ mục tiêu thiết kế của 2 task đó, không cần
thêm tham số "tier" nào. Gọi `get_llm("rewrite")` / `get_llm("decompose")`
trực tiếp là đủ.
--------------------------------------------------------------------------

Cả 2 hàm có bước heuristic tiền kiểm tra để TRÁNH gọi LLM khi không cần
thiết — tiết kiệm latency + chi phí, vì đây là bước chạy trên gần như mọi
request retrieval:
- rewrite_query: chỉ gọi LLM nếu câu hỏi có tham chiếu mơ hồ (đại từ,
  "phần trước"...) VÀ có chat_history_summary để resolve; ngược lại trả
  về nguyên văn câu hỏi.
- decompose_query: chỉ gọi LLM nếu câu hỏi có dấu hiệu multi-hop (so
  sánh, nhiều mệnh đề nối bằng "và"...); ngược lại trả về [query].
"""
from __future__ import annotations

import json
import logging
import re

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


def _call_llm(task: str, system_prompt: str, user_prompt: str) -> str:
    """Adapter DUY NHẤT gọi tới LLM router — task phải khớp TaskType của
    app/llm/router.py ("rewrite" hoặc "decompose" trong file này).
    Import trễ (bên trong hàm) để tránh cyclic import lúc module load."""
    from app.llm.router import get_llm

    llm = get_llm(task)
    response = llm.invoke([("system", system_prompt), ("human", user_prompt)])
    return getattr(response, "content", str(response)).strip()


def rewrite_query(query: str, chat_history_summary: str = "") -> str:
    """
    Viết lại câu hỏi mơ hồ/thiếu ngữ cảnh cho rõ nghĩa, dùng
    chat_history_summary để resolve tham chiếu kiểu "nó", "phần trước".

    Nếu câu hỏi không có dấu hiệu tham chiếu mơ hồ, hoặc không có
    chat_history_summary để dựa vào, trả về nguyên văn `query` mà KHÔNG
    gọi LLM.
    """
    query = query.strip()
    if not query:
        return query

    if not _needs_rewrite(query, chat_history_summary):
        return query

    system_prompt = (
        "Bạn là trợ lý viết lại câu hỏi cho hệ thống RAG hỗ trợ học tập. "
        "Nhiệm vụ: dựa vào tóm tắt hội thoại trước đó, viết lại câu hỏi của "
        "người dùng thành một câu hỏi ĐẦY ĐỦ NGỮ CẢNH, thay thế mọi đại từ/tham "
        "chiếu mơ hồ (nó, cái đó, phần trước...) bằng danh từ cụ thể được nhắc "
        "tới trong tóm tắt. CHỈ trả về câu hỏi đã viết lại, không thêm giải "
        "thích, không thêm dấu ngoặc kép."
    )
    user_prompt = f"Tóm tắt hội thoại trước đó:\n{chat_history_summary}\n\nCâu hỏi cần viết lại:\n{query}"

    try:
        rewritten = _call_llm("rewrite", system_prompt, user_prompt)
    except Exception as exc:
        logger.warning("rewrite_query: LLM call thất bại (%s) — dùng câu hỏi gốc.", exc)
        return query

    rewritten = rewritten.strip().strip('"').strip("'")
    if not rewritten:
        logger.warning("rewrite_query: LLM trả về chuỗi rỗng — dùng câu hỏi gốc.")
        return query

    logger.info("rewrite_query: '%s' -> '%s'", query, rewritten)
    return rewritten


def decompose_query(query: str) -> list[str]:
    """
    Tách câu hỏi multi-hop ("So sánh X và Y rồi cho ví dụ") thành list
    sub-query độc lập để retrieve riêng từng phần. Nếu câu hỏi không có
    dấu hiệu multi-hop, trả về [query] mà KHÔNG gọi LLM.
    """
    query = query.strip()
    if not query:
        return []

    if not _looks_multihop(query):
        return [query]

    system_prompt = (
        "Bạn là trợ lý phân rã câu hỏi cho hệ thống RAG hỗ trợ học tập. "
        "Nếu câu hỏi của người dùng chứa nhiều ý độc lập (multi-hop, so sánh "
        f"nhiều khái niệm, yêu cầu nhiều việc), hãy tách thành tối đa "
        f"{_MAX_SUBQUERIES} câu hỏi con, MỖI câu hỏi con phải tự đầy đủ nghĩa "
        "(không phụ thuộc câu khác) và dùng được để tìm kiếm tài liệu độc lập. "
        "Nếu câu hỏi vốn đã đơn giản, trả về chính nó trong list 1 phần tử. "
        'CHỈ trả về JSON thuần dạng {"sub_queries": ["...", "..."]}, không thêm '
        "giải thích, không dùng markdown code fence."
    )
    user_prompt = f"Câu hỏi: {query}"

    try:
        raw = _call_llm("decompose", system_prompt, user_prompt)
    except Exception as exc:
        logger.warning("decompose_query: LLM call thất bại (%s) — dùng câu hỏi gốc.", exc)
        return [query]

    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
        sub_queries = parsed["sub_queries"] if isinstance(parsed, dict) else parsed
        sub_queries = [str(q).strip() for q in sub_queries if str(q).strip()]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning(
            "decompose_query: parse JSON thất bại (%s), raw='%s' — dùng câu hỏi gốc.", exc, raw[:200]
        )
        return [query]

    if not sub_queries:
        return [query]

    sub_queries = sub_queries[:_MAX_SUBQUERIES]
    logger.info("decompose_query: '%s' -> %d sub-query.", query, len(sub_queries))
    return sub_queries