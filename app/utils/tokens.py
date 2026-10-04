"""
app/utils/tokens.py

Đếm token dùng chung cho toàn bộ project (context window budget, chunking,
prompt budgeting...). Tách riêng module này để tránh lặp lại logic
load-tiktoken-với-fallback ở nhiều nơi.

Ghi chú: app/chunking/semantic_chunker.py (Phase 2) hiện đang tự định
nghĩa `_count_tokens` riêng — có thể refactor để import từ đây khi tiện,
không bắt buộc vì không ảnh hưởng tính đúng đắn.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _load_tiktoken_counter():
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encoder.encode(text))
    except ImportError:
        logger.warning("tiktoken chưa được cài — dùng ước lượng token xấp xỉ.")
    except Exception as exc:  # lỗi mạng khi tải BPE file lần đầu, cache hỏng, ...
        logger.warning(
            "tiktoken cài rồi nhưng không load được encoding (%s) — dùng ước lượng token "
            "xấp xỉ. Cân nhắc bake sẵn tiktoken cache vào Docker image cho production.",
            exc,
        )
    return None


_TOKEN_COUNTER = _load_tiktoken_counter()


def count_tokens(text: str) -> int:
    """Đếm số token của `text`. Dùng tiktoken (cl100k_base) nếu load được,
    fallback ước lượng ~1.3 token/từ nếu không (thiếu lib hoặc mạng bị chặn
    lúc tải BPE file lần đầu)."""
    if not text:
        return 0
    if _TOKEN_COUNTER is not None:
        return _TOKEN_COUNTER(text)
    return max(1, round(len(text.split()) * 1.3))