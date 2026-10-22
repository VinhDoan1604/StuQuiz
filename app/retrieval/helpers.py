"""
app/retrieval/helpers.py

Lớp adapter điều phối kết nối giữa embedder (BGE-M3) + Qdrant store + reranker
— dùng chung bởi retrieval_agent.py, quiz_summary_agent.py và các service khác.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

DEFAULT_HYBRID_TOP_K = 20


async def aembed_query(text: str) -> dict[str, Any]:
    """Nhúng text query qua BGE-M3 (app.embedding.bge_m3_embedder)."""
    from app.embedding.bge_m3_embedder import embed_query

    return await asyncio.to_thread(embed_query, text)


async def ahybrid_search(
    dense_vec: Any,
    sparse_vec: Any,
    top_k: int = DEFAULT_HYBRID_TOP_K,
    metadata_filter: dict | None = None,
    document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Gọi hybrid search trên Qdrant."""
    from app.indexing.qdrant_store import hybrid_search

    return await asyncio.to_thread(
        hybrid_search,
        dense_vec=dense_vec,
        sparse_vec=sparse_vec,
        top_k=top_k,
        metadata_filter=metadata_filter,
        document_ids=document_ids,
    )


async def aget_all_chunks_for_documents(document_ids: list[str]) -> list[dict[str, Any]]:
    """Lấy toàn bộ chunk của các document (phục vụ tóm tắt đầy đủ)."""
    from app.indexing.qdrant_store import get_all_chunks_for_documents

    return await asyncio.to_thread(get_all_chunks_for_documents, document_ids)


async def arerank(query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Chấm điểm lại ứng viên bằng BGE Reranker."""
    return await asyncio.to_thread(rerank, query, candidates, top_k)


async def retrieve_top_k(
    query: str,
    top_k: int = 5,
    hybrid_top_k: int = DEFAULT_HYBRID_TOP_K,
    metadata_filter: dict | None = None,
    document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Pipeline đầy đủ: embed -> hybrid search (top hybrid_top_k) -> rerank (xuống top_k).
    Dùng cho QA/teaching/quiz (cần độ liên quan cao với 1 query cụ thể)."""
    if not query or not query.strip():
        return []
    try:
        vec = await aembed_query(query)
        candidates = await ahybrid_search(
            vec["dense"],
            vec["sparse"],
            top_k=hybrid_top_k,
            metadata_filter=metadata_filter,
            document_ids=document_ids,
        )
    except Exception as exc:
        logger.error("retrieve_top_k('%s'): lỗi embed/hybrid_search: %s", query[:80], exc)
        return []

    if not candidates:
        return []

    try:
        return await arerank(query, candidates, top_k=top_k)
    except Exception as exc:
        logger.warning(
            "retrieve_top_k('%s'): rerank lỗi (%s) — dùng thứ tự hybrid search gốc.",
            query[:80],
            exc,
        )
        return candidates[:top_k]


def normalize_chunk(candidate: dict[str, Any]) -> dict[str, Any]:
    """Chuẩn hoá 1 candidate về shape phẳng, nhất quán. Idempotent."""
    metadata = candidate.get("metadata") or {}
    text = candidate.get("text") or (candidate.get("chunk") or {}).get("text", "")
    return {
        "text": text,
        "source_file": metadata.get("source_file", candidate.get("source_file", "")),
        "page": metadata.get("page", candidate.get("page")),
        "section": metadata.get("section", candidate.get("section", "")),
        "content_type": metadata.get("content_type", candidate.get("content_type", "content")),
        "doc_id": metadata.get("doc_id", candidate.get("doc_id")),
        "score": candidate.get("rerank_score", candidate.get("score", 0.0)),
        "parent_id": metadata.get("parent_id", candidate.get("parent_id")),
        "snippet": candidate.get("snippet"),
        "expanded": candidate.get("expanded", False),
    }


def format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    """Định dạng list chunk thành block đánh số để đưa vào prompt cho LLM."""
    if not chunks:
        return "(không có context nào được truy xuất)"
    blocks = []
    for i, c in enumerate(chunks, start=1):
        source = f"Nguồn: {c.get('source_file') or '?'}"
        if c.get("page") is not None:
            source += f", trang {c['page']}"
        if c.get("section"):
            source += f", mục \"{c['section']}\""
        blocks.append(f"[{i}] ({source})\n{c.get('text', '')}")
    return "\n\n".join(blocks)
