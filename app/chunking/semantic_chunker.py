"""
app/chunking/semantic_chunker.py

chunk_document(doc) -> list[Chunk]

Chiến lược: Hierarchical + Structure-aware chunking.

- "Semantic" ở đây được hiện thực bằng ranh giới heading/section (cấu trúc
  thật của tài liệu, lấy từ parsers.py) thay vì cắt cứng theo token count.
  Đây là lựa chọn có chủ đích: rẻ, nhanh, không tốn thêm lời gọi embedding
  lúc ingest, và đặc biệt phù hợp với tài liệu học tập vốn có cấu trúc
  chương/mục/slide rõ ràng — khác với văn bản tự do (vd. bài báo) nơi
  embedding-similarity splitting mới thực sự cần thiết.
  TODO (nâng cấp có điều kiện): nếu sau này benchmark cho thấy cần tách
  ý trong CÙNG 1 section dài mà heading không đủ mịn, có thể thay
  `_split_into_sentences` + `_pack_sentences` bằng so sánh cosine
  similarity giữa embedding của các câu liền kề (kiểu SemanticChunker của
  LlamaIndex) — nhưng cần đo thêm chi phí/độ trễ ingest trước khi thêm.

- 2 tầng, đúng theo yêu cầu:
    parent chunk = toàn bộ 1 section (hoặc 1 phần nếu section quá dài),
                   dùng để mở rộng ngữ cảnh khi Reflection Agent thấy
                   child chunk "liên quan nhưng thiếu ngữ cảnh".
    child chunk  = ~200-400 token, dùng để retrieve chính xác, luôn có
                   `parent_id` trỏ về parent chunk chứa nó.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field

from app.models.schemas import BlockType, Chunk, ChunkMetadata, ParsedBlock, ParsedDocument

logger = logging.getLogger(__name__)

CHILD_MIN_TOKENS = 200
CHILD_MAX_TOKENS = 400
CHILD_OVERLAP_TOKENS = 50
PARENT_MAX_TOKENS = 1500

def _load_tiktoken_counter():
    """Thử load tiktoken cl100k_base. Có thể fail vì 2 lý do độc lập:
    (1) chưa cài package, hoặc (2) đã cài nhưng lần đầu chạy cần tải file
    BPE qua mạng và môi trường (CI, container air-gapped, ...) chặn mạng
    ra ngoài — network lỗi vẫn là Exception bình thường của thư viện,
    KHÔNG phải ImportError, nên phải bắt riêng để không crash cả module.
    Production nên bake sẵn cache tiktoken vào image (xem README) để
    tránh phụ thuộc mạng lúc runtime."""
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encoder.encode(text))
    except ImportError:
        logger.warning("tiktoken chưa được cài — dùng ước lượng token xấp xỉ (kém chính xác hơn).")
    except Exception as exc:  # lỗi mạng khi tải BPE file, cache hỏng, ...
        logger.warning(
            "tiktoken cài rồi nhưng không load được encoding (%s) — dùng ước lượng token "
            "xấp xỉ. Cân nhắc bake sẵn tiktoken cache vào Docker image cho production.",
            exc,
        )
    return None


_TOKEN_COUNTER = _load_tiktoken_counter()


def _count_tokens(text: str) -> int:
    if _TOKEN_COUNTER is not None:
        return _TOKEN_COUNTER(text)
    return max(1, round(len(text.split()) * 1.3))


_CONTENT_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(định nghĩa|khái niệm|là gì)", re.IGNORECASE), "definition"),
    (re.compile(r"(ví dụ|minh hoạ|minh họa|chẳng hạn|illustration)", re.IGNORECASE), "example"),
    (re.compile(r"(bài tập|câu hỏi|luyện tập|exercise|thực hành)", re.IGNORECASE), "exercise"),
]

# Tách câu: sau dấu kết câu (. ! ? …) + khoảng trắng + chữ hoa/số kế tiếp.
# Đơn giản nhưng đủ tốt cho văn bản học thuật; không cố xử lý mọi viết tắt
# tiếng Việt (vd "TS." "ThS.") vì rủi ro tách sai câu thấp hơn lợi ích của
# một sentence splitter phức tạp trong bối cảnh chunking (over-split ở
# viết tắt chỉ làm chunk hơi vụn, không làm sai retrieval).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-Ỹ0-9])")


def _classify_content_type(section_path: str, text_sample: str) -> str:
    haystack = f"{section_path} {text_sample[:300]}"
    for pattern, label in _CONTENT_TYPE_PATTERNS:
        if pattern.search(haystack):
            return label
    return "content"


def _split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


@dataclass
class _Section:
    path: str
    blocks: list[ParsedBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)

    @property
    def pages(self) -> list[int]:
        return sorted({b.page for b in self.blocks if b.page is not None})


def _build_sections(doc: ParsedDocument) -> list[_Section]:
    """Nhóm block theo ranh giới heading, giữ đường dẫn section dạng
    'Chương 1 > Mục 1.2' bằng một stack heading level."""
    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    current = _Section(path="(không có heading)")

    def flush() -> None:
        if current.blocks:
            sections.append(current)

    for block in doc.blocks:
        if block.type == BlockType.HEADING:
            flush()
            level = block.level or 1
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, block.text))
            path = " > ".join(t for _, t in heading_stack)
            current = _Section(path=path)
        else:
            current.blocks.append(block)

    flush()
    return sections


def _split_section_into_parents(section: _Section) -> list[_Section]:
    """Nếu section quá dài (> PARENT_MAX_TOKENS), tách thành nhiều parent
    con theo ranh giới block (không cắt giữa 1 paragraph/table/list-item)."""
    total_tokens = _count_tokens(section.text)
    if total_tokens <= PARENT_MAX_TOKENS:
        return [section]

    parts: list[_Section] = []
    buffer = _Section(path=section.path)
    buffer_tokens = 0

    for block in section.blocks:
        block_tokens = _count_tokens(block.text)
        if buffer_tokens + block_tokens > PARENT_MAX_TOKENS and buffer.blocks:
            parts.append(buffer)
            buffer = _Section(path=section.path)
            buffer_tokens = 0
        buffer.blocks.append(block)
        buffer_tokens += block_tokens

    if buffer.blocks:
        parts.append(buffer)

    logger.info(
        "Section '%s' dài %d token — tách thành %d parent chunk.", section.path, total_tokens, len(parts)
    )
    return parts


def _pack_sentences(
    sentences: list[str],
    max_tokens: int = CHILD_MAX_TOKENS,
    overlap_tokens: int = CHILD_OVERLAP_TOKENS,
    min_tokens: int = CHILD_MIN_TOKENS,
) -> list[str]:
    """Gộp câu thành các cửa sổ ~max_tokens, có overlap để không mất ngữ
    cảnh ở ranh giới chunk. Cửa sổ cuối cùng nếu quá nhỏ (< min_tokens) sẽ
    được gộp vào cửa sổ liền trước thay vì tồn tại như 1 chunk vụn."""
    windows: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)

        if current_tokens + sentence_tokens > max_tokens and current_sentences:
            windows.append(" ".join(current_sentences))
            overlap_sentences: list[str] = []
            overlap_count = 0
            for s in reversed(current_sentences):
                t = _count_tokens(s)
                if overlap_count + t > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += t
            current_sentences = overlap_sentences
            current_tokens = overlap_count

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    if current_sentences:
        windows.append(" ".join(current_sentences))

    if len(windows) > 1 and _count_tokens(windows[-1]) < min_tokens:
        windows[-2] = windows[-2] + " " + windows[-1]
        windows.pop()

    return windows


def chunk_document(doc: ParsedDocument, doc_id: str | None = None) -> list[Chunk]:
    """
    Điểm vào chính. Input là ParsedDocument ĐÃ qua `clean_document()`.

    Returns:
        list[Chunk] gồm cả parent chunk (level="parent", parent_id=None)
        và child chunk (level="child", parent_id trỏ về parent chứa nó).
        Reflection Agent tra parent bằng parent_id khi cần mở rộng ngữ cảnh
        quanh 1 child chunk retrieve được.
    """
    if not doc.blocks:
        logger.warning("'%s' không có block nào — trả về danh sách chunk rỗng.", doc.filename)
        return []

    doc_id = doc_id or str(uuid.uuid4())
    sections = _build_sections(doc)
    all_chunks: list[Chunk] = []
    chunk_index = 0

    for section in sections:
        for parent_section in _split_section_into_parents(section):
            parent_text = parent_section.text
            if not parent_text.strip():
                continue

            pages = parent_section.pages
            page_range = (pages[0], pages[-1]) if pages else None
            content_type = _classify_content_type(parent_section.path, parent_text)

            parent_chunk = Chunk(
                doc_id=doc_id,
                parent_id=None,
                level="parent",
                text=parent_text,
                token_count=_count_tokens(parent_text),
                chunk_index=chunk_index,
                metadata=ChunkMetadata(
                    source_file=doc.filename,
                    page=pages[0] if pages else None,
                    page_range=page_range,
                    section=parent_section.path,
                    content_type=content_type,
                ),
            )
            all_chunks.append(parent_chunk)
            chunk_index += 1

            sentences = _split_into_sentences(parent_text)
            for child_text in _pack_sentences(sentences):
                if _count_tokens(child_text) < 10:
                    # bỏ mảnh vụn quá nhỏ (thường do lỗi tách câu ở cuối section)
                    continue
                all_chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        parent_id=parent_chunk.id,
                        level="child",
                        text=child_text,
                        token_count=_count_tokens(child_text),
                        chunk_index=chunk_index,
                        metadata=ChunkMetadata(
                            source_file=doc.filename,
                            page=pages[0] if pages else None,
                            page_range=page_range,
                            section=parent_section.path,
                            content_type=content_type,
                        ),
                    )
                )
                chunk_index += 1

    logger.info(
        "chunk_document('%s'): %d section -> %d chunk (%d parent, %d child).",
        doc.filename, len(sections), len(all_chunks),
        sum(1 for c in all_chunks if c.level == "parent"),
        sum(1 for c in all_chunks if c.level == "child"),
    )
    return all_chunks