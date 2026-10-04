# STATUS: READY
"""
Pydantic schemas cho toàn bộ hệ thống StudyRAG:
- Ingestion models (ParsedBlock, ParsedDocument)
- Chunking models (Chunk, ChunkMetadata)
- API request/response DTOs (Chat, Quiz, Mastery, Reflection)
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ===========================================================================
# 1. Ingestion Schemas (pipeline: parsing -> cleaning -> chunking)
# ===========================================================================

class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST_ITEM = "list_item"
    SPEAKER_NOTES = "speaker_notes"


class ParsedBlock(BaseModel):
    """Một đơn vị nội dung nguyên tử, giữ đúng thứ tự xuất hiện trong tài liệu gốc."""

    type: BlockType
    text: str
    page: Optional[int] = None          # số trang (PDF) hoặc số slide (PPTX), 1-indexed
    level: Optional[int] = None         # heading level (1..6); None nếu không phải heading
    order: int = 0                      # thứ tự toàn cục trong document, dùng để sort ổn định
    extra: dict[str, Any] = Field(default_factory=dict)  # vd: {"source": "ocr", "ocr_confidence": 82.3}


class ParsedDocument(BaseModel):
    filename: str
    file_type: str                      # "pdf" | "docx" | "pptx"
    blocks: list[ParsedBlock] = Field(default_factory=list)
    total_pages: int = 0
    parsed_at: datetime = Field(default_factory=datetime.utcnow)
    warnings: list[str] = Field(default_factory=list)   # OCR confidence thấp, trang lỗi, ...
    doc_metadata: dict[str, Any] = Field(default_factory=dict)


# ===========================================================================
# 2. Chunking Schemas (dùng chung cho fixed & semantic chunkers)
# ===========================================================================

class ChunkMetadata(BaseModel):
    source_file: str
    page: Optional[int] = None                      # trang đầu tiên chunk xuất hiện
    page_range: Optional[tuple[int, int]] = None     # (trang đầu, trang cuối) chunk trải qua
    section: str = ""                                # đường dẫn heading, vd: "Chương 1 > Mục 1.2"
    content_type: Literal["definition", "example", "exercise", "content"] = "content"
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    parent_id: Optional[str] = None        # None nếu đây là parent/section chunk
    level: Literal["parent", "child"] = "child"
    text: str
    token_count: int = 0
    chunk_index: int = 0                   # thứ tự xuất hiện trong document, dùng để debug/hiển thị
    metadata: ChunkMetadata = Field(default_factory=lambda: ChunkMetadata(source_file=""))

    # Convenience properties đảm bảo tương thích ngược với qdrant_store.py và downstream
    @property
    def source_file(self) -> str:
        return self.metadata.source_file

    @property
    def page(self) -> Optional[int]:
        return self.metadata.page

    @property
    def section(self) -> str:
        return self.metadata.section

    @property
    def content_type(self) -> str:
        return self.metadata.content_type

    @property
    def subject(self) -> Optional[str]:
        return self.metadata.extra.get("subject")


# ===========================================================================
# 3. API DTOs (Request / Response) & Reflection Trace
# ===========================================================================

class ChatRequest(BaseModel):
    session_id: str
    message: str
    document_ids: Optional[list[str]] = None
    space_id: Optional[str] = None
    user_id: Optional[str] = "demo_user"


class ReflectionTrace(BaseModel):
    """Hiển thị trực tiếp ra UI — xem mục 3 tài liệu (transparency thật)."""
    context_grade: float | None = None
    context_sufficient: bool | None = None
    rewritten_queries: list[str] = []
    answer_grade: float | None = None
    answer_faithful: bool | None = None
    num_loops: int = 0


class ChatResponse(BaseModel):
    intent: Literal["qa", "teaching", "quiz", "summary"]
    answer: str
    citations: list[dict[str, Any]] = []   # [{"file": ..., "page": ...}]
    reflection: ReflectionTrace


class QuizQuestion(BaseModel):
    question: str
    choices: list[str]
    correct_choice: str
    explanation: str
    topic: str
    difficulty: Literal["easy", "medium", "hard"]


class QuizGenerateRequest(BaseModel):
    document_ids: list[str]
    user_id: Optional[str] = "demo_user"
    space_id: Optional[str] = None
    topic_prompt: Optional[str] = None
    num_questions: int = 5
    prioritize_weak_topics: bool = True


class MasteryTopic(BaseModel):
    topic: str
    mastery_score: int
    streak: int = 0
    last_reviewed_at: datetime | None
    next_review_due_at: datetime | None
