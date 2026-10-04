"""
app/llm/utils.py

Helper dùng chung cho mọi agent / service cần gọi LLM — 2 kiểu:

1. acall_llm_text(): trả về text thô (.content). Dùng cho sinh văn bản tự
   do (câu trả lời QA/teaching, tóm tắt).
2. acall_structured(): ép LLM trả về đúng 1 Pydantic model, dùng
   `BaseChatModel.with_structured_output()` — cơ chế structured
   output/function-calling NATIVE của provider (OpenAI tool-calling,
   Anthropic tool-use, Gemini function-calling), robust hơn nhiều so với
   tự parse chuỗi JSON bằng regex/strip code-fence, vì việc đảm bảo đúng
   schema được đẩy xuống tầng decode của provider thay vì đoán mò ở tầng
   ứng dụng. Dùng cho Orchestrator (phân loại intent) và Reflection Agent
   (chấm điểm) — 2 nơi cần output có cấu trúc chặt.

Cả 2 đều dùng `get_llm(task)` của app/llm/router.py — `task` PHẢI thuộc
Literal["rewrite","decompose","grade","qa","teaching","quiz","summary"].

Lưu ý về with_structured_output(): độ tin cậy có thể khác nhau giữa
provider (OpenAI/Anthropic qua tool-calling rất ổn định; Gemini qua
langchain-google-genai đôi khi kén schema lồng nhau phức tạp hơn). Vì vậy
MỌI lời gọi acall_structured() ở agent khác đều PHẢI có try/except với
fallback an toàn theo từng use case cụ thể — không có 1 giá trị fallback
chung đúng cho mọi trường hợp, nên fallback được quyết định ở nơi gọi,
không phải ở đây.
"""
from __future__ import annotations

import logging
from typing import TypeVar, Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMCallError(RuntimeError):
    """LLM call thất bại (lỗi mạng, rate limit, provider lỗi, output không
    khớp schema yêu cầu...). Caller PHẢI catch và quyết định fallback phù
    hợp với ngữ cảnh cụ thể của mình."""


def extract_text_from_response(response: Any) -> str:
    """Trích xuất chuỗi văn bản sạch từ LangChain AIMessage (hỗ trợ cả str và list[dict] từ multimodal/newer providers)."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


async def acall_llm_text(task: str, system_prompt: str, user_prompt: str) -> str:
    """Gọi LLM, trả về text thô. KHÔNG dùng cho output cần đúng cấu trúc
    (dùng acall_structured cho trường hợp đó)."""
    from app.llm.router import get_llm

    try:
        llm = get_llm(task)
        response = await llm.ainvoke([("system", system_prompt), ("human", user_prompt)])
    except Exception as exc:
        raise LLMCallError(f"Lỗi gọi LLM (task='{task}'): {exc}") from exc

    return extract_text_from_response(response)


async def acall_structured(task: str, system_prompt: str, user_prompt: str, output_model: type[T]) -> T:
    """Gọi LLM và ép trả về đúng instance của `output_model`."""
    from app.llm.router import get_llm

    try:
        llm = get_llm(task)
        structured_llm = llm.with_structured_output(output_model)
        result = await structured_llm.ainvoke([("system", system_prompt), ("human", user_prompt)])
    except Exception as exc:
        raise LLMCallError(
            f"Lỗi structured output (task='{task}', model={output_model.__name__}): {exc}"
        ) from exc

    if isinstance(result, output_model):
        return result
    # Một số provider trả về dict thay vì instance đã validate sẵn.
    try:
        return output_model.model_validate(result)
    except Exception as exc:
        raise LLMCallError(
            f"Structured output (task='{task}') không khớp schema {output_model.__name__}: {exc}"
        ) from exc
