"""
app/agents/llm_utils.py

[DEPRECATED / COMPATIBILITY SHIM]
Module này đã được chuyển về vị trí chuẩn kiến trúc: `app.llm.utils`.
File này giữ lại để đảm bảo tương thích ngược (re-export). Vui lòng import
trực tiếp từ `app.llm.utils`.
"""
from app.llm.utils import LLMCallError, acall_llm_text, acall_structured

__all__ = ["LLMCallError", "acall_llm_text", "acall_structured"]