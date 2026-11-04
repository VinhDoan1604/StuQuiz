"""
app/utils/tokens.py

Đếm token dùng chung cho toàn bộ project (context window budget, chunking,
prompt budgeting...). tiktoken cl100k_base nếu load được; nếu không (thiếu
package hoặc không tải được file BPE lần đầu do chặn mạng) thì ước lượng
~1.3 token/từ. Docker image bake sẵn cache tiktoken (xem Dockerfile).
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
