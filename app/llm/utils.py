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

3. acall_llm_vision() / call_llm_vision(): prompt văn bản + ảnh (PNG/JPEG bytes), trả về
   text. Bản đồng bộ dành cho ingestion (parse chạy trong thread nền, không có event loop).

Tất cả đều dùng `get_llm(task)` của app/llm/router.py — `task` PHẢI thuộc
Literal["rewrite","decompose","grade","critic","qa","teaching","quiz","summary","vision"].

Lưu ý về with_structured_output(): độ tin cậy có thể khác nhau giữa
provider (OpenAI/Anthropic qua tool-calling rất ổn định; Gemini qua
langchain-google-genai đôi khi kén schema lồng nhau phức tạp hơn). Vì vậy
MỌI lời gọi acall_structured() ở agent khác đều PHẢI có try/except với
fallback an toàn theo từng use case cụ thể — không có 1 giá trị fallback
chung đúng cho mọi trường hợp, nên fallback được quyết định ở nơi gọi,
không phải ở đây.
"""
from __future__ import annotations

import base64
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


class LLMCallError(RuntimeError):
    """LLM call thất bại (lỗi mạng, rate limit, provider lỗi, output không
    khớp schema yêu cầu...). Caller PHẢI catch và quyết định fallback phù
    hợp với ngữ cảnh cụ thể của mình."""


def _content_text(response: Any) -> str:
    """Phần văn bản của AIMessage/AIMessageChunk, GIỮ khoảng trắng (cần khi nối token stream)."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Chỉ lấy block văn bản; bỏ qua block thinking/reasoning (vd Gemini 2.5
                # trả {'type': 'thinking', 'thinking': ''}) — trước đây bị str() dán vào câu trả lời.
                if item.get("type", "text") == "text" and "text" in item:
                    parts.append(str(item["text"]))
            elif hasattr(item, "text") and getattr(item, "type", "text") == "text":
                parts.append(str(getattr(item, "text")))
        return "".join(parts)
    return str(content)


def extract_text_from_response(response: Any) -> str:
    """Trích xuất chuỗi văn bản sạch từ LangChain AIMessage (hỗ trợ cả str và list[dict] từ multimodal/newer providers)."""
    return _content_text(response).strip()


async def _with_fallback(task: str, call: Callable[[Any], Awaitable[R]], error_label: str) -> R:
    """Chạy `call(llm)` với model chính; lỗi thì thử model dự phòng đúng 1 lần (nếu có cấu hình)."""
    from app.llm.router import get_fallback_llm, get_llm, record_fallback

    try:
        return await call(get_llm(task))
    except Exception as exc:
        fallback = get_fallback_llm(task)
        if fallback is None:
            raise LLMCallError(f"{error_label}: {exc}") from exc
        logger.warning("LLM chính lỗi (task=%s, %s): %s — chuyển sang model dự phòng.", task, type(exc).__name__, exc)

    record_fallback()
    try:
        return await call(fallback)
    except Exception as exc:
        raise LLMCallError(f"{error_label} (cả model dự phòng): {exc}") from exc


def _with_fallback_sync(task: str, call: Callable[[Any], R], error_label: str) -> R:
    """Bản đồng bộ của `_with_fallback` (cùng quy tắc: thử model dự phòng đúng 1 lần)."""
    from app.llm.router import get_fallback_llm, get_llm, record_fallback

    try:
        return call(get_llm(task))
    except Exception as exc:
        fallback = get_fallback_llm(task)
        if fallback is None:
            raise LLMCallError(f"{error_label}: {exc}") from exc
        logger.warning("LLM chính lỗi (task=%s, %s): %s — chuyển sang model dự phòng.", task, type(exc).__name__, exc)

    record_fallback()
    try:
        return call(fallback)
    except Exception as exc:
        raise LLMCallError(f"{error_label} (cả model dự phòng): {exc}") from exc


def image_mime_type(data: bytes) -> str:
    """Nhận dạng PNG/JPEG/WebP/GIF theo magic bytes (không tin đuôi file)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    raise ValueError("Ảnh không phải PNG/JPEG/WebP/GIF.")


def build_vision_messages(system_prompt: str, user_prompt: str, images: Sequence[bytes]) -> list[Any]:
    """[SystemMessage, HumanMessage(text + image_url data-URL)].

    Định dạng `{"type": "image_url", "image_url": {"url": "data:..."}}` là chuẩn chung của
    LangChain: ChatGoogleGenerativeAI, ChatOpenAI và ChatAnthropic đều tự chuyển sang định dạng
    riêng của provider, nên đổi provider không phải sửa chỗ gọi."""
    from langchain_core.messages import HumanMessage, SystemMessage

    if not images:
        raise ValueError("Cần ít nhất 1 ảnh.")
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for data in images:
        b64 = base64.b64encode(data).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{image_mime_type(data)};base64,{b64}"}})
    return [SystemMessage(content=system_prompt), HumanMessage(content=content)]


async def acall_llm_vision(task: str, system_prompt: str, user_prompt: str, images: Sequence[bytes]) -> str:
    """Gọi LLM đa phương thức (văn bản + ảnh), trả về text thô."""
    messages = build_vision_messages(system_prompt, user_prompt, images)
    response = await _with_fallback(task, lambda llm: llm.ainvoke(messages), f"Lỗi gọi VLM (task='{task}')")
    return extract_text_from_response(response)


def call_llm_vision(task: str, system_prompt: str, user_prompt: str, images: Sequence[bytes]) -> str:
    """Như `acall_llm_vision` nhưng đồng bộ — cho code chạy trong thread (parse tài liệu)."""
    messages = build_vision_messages(system_prompt, user_prompt, images)
    response = _with_fallback_sync(task, lambda llm: llm.invoke(messages), f"Lỗi gọi VLM (task='{task}')")
    return extract_text_from_response(response)


async def acall_llm_text(task: str, system_prompt: str, user_prompt: str) -> str:
    """Gọi LLM, trả về text thô. KHÔNG dùng cho output cần đúng cấu trúc
    (dùng acall_structured cho trường hợp đó)."""
    messages = [("system", system_prompt), ("human", user_prompt)]
    response = await _with_fallback(task, lambda llm: llm.ainvoke(messages), f"Lỗi gọi LLM (task='{task}')")
    return extract_text_from_response(response)


async def astream_llm_text(task: str, system_prompt: str, user_prompt: str, on_token: Callable[[str], None]) -> str:
    """Như acall_llm_text nhưng gọi `on_token` cho từng mảnh văn bản.

    Chỉ chuyển sang model dự phòng khi CHƯA phát token nào: đã hiện chữ cho người dùng
    thì không thể đổi model giữa chừng."""
    from app.llm.router import get_fallback_llm, get_llm, record_fallback

    messages = [("system", system_prompt), ("human", user_prompt)]
    parts: list[str] = []

    async def consume(llm: Any) -> None:
        async for chunk in llm.astream(messages):
            text = _content_text(chunk)
            if text:
                parts.append(text)
                on_token(text)

    try:
        await consume(get_llm(task, streaming=True))
    except Exception as exc:
        fallback = None if parts else get_fallback_llm(task, streaming=True)
        if fallback is None:
            raise LLMCallError(f"Lỗi stream LLM (task='{task}'): {exc}") from exc
        logger.warning("LLM chính lỗi khi stream (task=%s): %s — chuyển sang model dự phòng.", task, exc)
        record_fallback()
        try:
            await consume(fallback)
        except Exception as exc2:
            raise LLMCallError(f"Lỗi stream LLM (task='{task}', cả model dự phòng): {exc2}") from exc2

    return "".join(parts).strip()


async def acall_structured(task: str, system_prompt: str, user_prompt: str, output_model: type[T]) -> T:
    """Gọi LLM và ép trả về đúng instance của `output_model`."""
    messages = [("system", system_prompt), ("human", user_prompt)]
    result = await _with_fallback(
        task,
        lambda llm: llm.with_structured_output(output_model).ainvoke(messages),
        f"Lỗi structured output (task='{task}', model={output_model.__name__})",
    )

    if isinstance(result, output_model):
        return result
    # Một số provider trả về dict thay vì instance đã validate sẵn.
    try:
        return output_model.model_validate(result)
    except Exception as exc:
        raise LLMCallError(
            f"Structured output (task='{task}') không khớp schema {output_model.__name__}: {exc}"
        ) from exc
