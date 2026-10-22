"""
app/retrieval/expansion.py

Parent-chunk expansion: child chunk (~200-400 token) giúp TÌM chính xác, nhưng
khi SINH câu trả lời ta cần ngữ cảnh rộng hơn => thay child bằng parent (cả
section, tới 1500 token) sau bước rerank.

- Parent được lưu ở bảng Postgres `parent_chunks` (không nhúng vector, không làm
  nhiễu tìm kiếm); child mang `parent_id` trong payload Qdrant.
- Nhiều child cùng parent => chỉ giữ 1 parent (khử trùng lặp).
- Có ngân sách token: parent nào làm vượt ngân sách thì giữ nguyên child.
- Mỗi chunk trả ra có `snippet` = đoạn child gốc (≤300 ký tự) để hiển thị
  trích dẫn chính xác, dù text gửi cho LLM là parent.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONTEXT_TOKENS = 3500
SNIPPET_CHARS = 300
_WS = re.compile(r"\s+")


def make_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = _WS.sub(" ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def expand_chunks_with_parents(
    chunks: list[dict[str, Any]],
    parents: dict[str, dict[str, Any]],
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    count_tokens: Callable[[str], int] | None = None,
) -> list[dict[str, Any]]:
    """Hàm thuần, giữ nguyên thứ tự xếp hạng; không mutate input."""
    if count_tokens is None:
        from app.utils.tokens import count_tokens as _ct

        count_tokens = _ct

    out: list[dict[str, Any]] = []
    used = 0
    seen_parents: set[str] = set()

    for c in chunks:
        pid = c.get("parent_id")
        if pid and pid in seen_parents:
            continue  # parent này đã đại diện cho child trước đó

        child_text = c.get("text", "")
        chosen = dict(c)
        chosen["snippet"] = make_snippet(child_text)

        parent = parents.get(pid) if pid else None
        if parent and parent.get("text"):
            p_tokens = count_tokens(parent["text"])
            if used + p_tokens <= max_context_tokens:
                chosen["text"] = parent["text"]
                chosen["child_text"] = child_text
                chosen["expanded"] = True
                used += p_tokens
                seen_parents.add(pid)
                out.append(chosen)
                continue

        used += count_tokens(child_text)
        out.append(chosen)
    return out


def fetch_parents(parent_ids: list[str]) -> dict[str, dict[str, Any]]:
    ids = sorted({p for p in parent_ids if p})
    if not ids:
        return {}
    from app.db.session import SessionLocal
    from app.models.db_models import ParentChunk

    with SessionLocal() as db:
        rows = db.query(ParentChunk).filter(ParentChunk.id.in_(ids)).all()
        return {r.id: {"text": r.text, "section": r.section, "page": r.page} for r in rows}


async def aexpand_with_parents(
    chunks: list[dict[str, Any]], max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
) -> list[dict[str, Any]]:
    """Lỗi DB không làm hỏng retrieval: lùi về child (vẫn có snippet)."""
    parents: dict[str, dict[str, Any]] = {}
    ids = [c["parent_id"] for c in chunks if c.get("parent_id")]
    if ids:
        try:
            parents = await asyncio.to_thread(fetch_parents, ids)
        except Exception as exc:
            logger.warning("aexpand_with_parents: không lấy được parent (%s) — dùng child.", exc)
    return expand_chunks_with_parents(chunks, parents, max_context_tokens)