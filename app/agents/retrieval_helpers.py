"""
app/agents/retrieval_helpers.py

[DEPRECATED / COMPATIBILITY SHIM]
Module này đã được chuyển về vị trí chuẩn kiến trúc: `app.retrieval.helpers`.
File này giữ lại để đảm bảo tương thích ngược (re-export). Vui lòng import
trực tiếp từ `app.retrieval.helpers`.
"""
from app.retrieval.helpers import (
    DEFAULT_HYBRID_TOP_K,
    aembed_query,
    ahybrid_search,
    aget_all_chunks_for_documents,
    arerank,
    retrieve_top_k,
    normalize_chunk,
    format_chunks_for_prompt,
)

__all__ = [
    "DEFAULT_HYBRID_TOP_K",
    "aembed_query",
    "ahybrid_search",
    "aget_all_chunks_for_documents",
    "arerank",
    "retrieve_top_k",
    "normalize_chunk",
    "format_chunks_for_prompt",
]