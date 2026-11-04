"""
app/agents/quiz_topics.py

Lập kế hoạch chủ đề quiz theo MỤC (heading) của tài liệu, để mastery /
spaced repetition theo dõi được từng phần kiến thức thay vì 1 chủ đề chung.

Heading của PDF được nhận diện bằng heuristic (app/ingestion/parsers.py) nên có
nhiễu: header bảng ("AUC (%)"), ký hiệu ("T", "O("), mảnh câu dài... Vì vậy:
1. Thành phần heading "rác" bị bỏ khỏi đường dẫn => nội dung gộp lên mục cha.
2. Mục quá ít nội dung (< MIN_TOPIC_TOKENS) gộp lên mục cha / cấp tài liệu.
3. Nhãn = tên mục đã bỏ số thứ tự + tên tài liệu (không trùng giữa các tài liệu),
   ổn định giữa các lần sinh quiz => khớp đúng MasteryRecord cũ.

Module thuần (không DB/LLM) để test được độc lập.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.utils.tokens import count_tokens

NO_HEADING = "(không có heading)"
MIN_TOPIC_TOKENS = 40
MAX_LABEL_CHARS = 80
_PATH_SEP = " > "

_NUMBERING_RE = re.compile(
    r"^\s*(?:(?:chương|chuong|phần|phan|mục|muc|bài|bai|chapter|section|part)\s+)?"
    r"(?:[ivxlc]+|\d+)(?:\.\d+)*[.):]?\s+",
    re.IGNORECASE,
)


@dataclass
class SectionTopic:
    label: str
    doc_id: str
    order: int
    chunks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(count_tokens(c.get("text", "")) for c in self.chunks)


def clean_heading(heading: str) -> str:
    """Bỏ số thứ tự đầu heading: '2.2 Domain Adaptation' -> 'Domain Adaptation'."""
    text = re.sub(r"\s+", " ", heading or "").strip()
    stripped = _NUMBERING_RE.sub("", text).strip(" .:-–—")
    return stripped or text


def is_meaningful_heading(heading: str) -> bool:
    """Heading đủ nghĩa để làm chủ đề học tập (lọc nhiễu từ heuristic nhận diện heading)."""
    text = clean_heading(heading)
    if not text or len(text) > MAX_LABEL_CHARS:
        return False
    letters = sum(ch.isalpha() for ch in text)
    visible = sum(not ch.isspace() for ch in text)
    if letters < 5 or letters / max(visible, 1) < 0.6:
        return False
    if text.count("(") != text.count(")") or any(ch in text for ch in "%=<>|"):
        return False
    # 1 từ viết hoa toàn bộ = từ viết tắt / header bảng (EAPIV, MMD, GAP), không phải chủ đề
    if len(text.split()) == 1 and not any(ch.islower() for ch in text):
        return False
    # Mảnh câu bị nhận nhầm là heading: có dấu chấm giữa câu
    return not re.search(r"[.!?]\s+\S", text)


def _normalized_path(section: str) -> tuple[str, ...]:
    if not section or section.strip() == NO_HEADING:
        return ()
    return tuple(p.strip() for p in section.split(_PATH_SEP) if is_meaningful_heading(p))


def _doc_title(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return stem.strip() or filename


def build_section_topics(
    sections: list[dict[str, Any]], doc_names: dict[str, str]
) -> list[SectionTopic]:
    """
    `sections`: mỗi phần tử {"doc_id", "section", "text", "page"?} theo thứ tự tài liệu.
    `doc_names`: doc_id -> filename.
    Trả về danh sách chủ đề theo thứ tự xuất hiện, mỗi chủ đề kèm các đoạn nội dung.
    """
    groups: dict[tuple[str, tuple[str, ...]], SectionTopic] = {}
    for order, s in enumerate(sections):
        doc_id = str(s["doc_id"])
        key = (doc_id, _normalized_path(s.get("section") or ""))
        topic = groups.get(key)
        if topic is None:
            topic = groups[key] = SectionTopic(label="", doc_id=doc_id, order=order)
        topic.chunks.append({
            "text": s.get("text", ""),
            "source_file": doc_names.get(doc_id, ""),
            "page": s.get("page"),
            "section": s.get("section") or "",
        })

    # Gộp mục ít nội dung lên cha, từng độ sâu từ sâu tới nông (mục cha mới tạo
    # ra cũng được xét ở vòng sau).
    max_depth = max((len(path) for _, path in groups), default=0)
    for depth in range(max_depth, 0, -1):
        for key in [k for k in groups if len(k[1]) == depth]:
            topic = groups[key]
            if topic.tokens >= MIN_TOPIC_TOKENS:
                continue
            doc_id, path = key
            parent = groups.get((doc_id, path[:-1]))
            if parent is None:
                topic.chunks = list(topic.chunks)
                groups[(doc_id, path[:-1])] = topic
            else:
                parent.chunks.extend(topic.chunks)
                parent.order = min(parent.order, topic.order)
            del groups[key]

    # Cùng nhãn (vd 1 file tải lên 2 lần, hoặc 2 mục cùng tên) => 1 chủ đề = 1 MasteryRecord
    by_label: dict[str, SectionTopic] = {}
    for (doc_id, path), topic in sorted(groups.items(), key=lambda kv: kv[1].order):
        title = _doc_title(doc_names.get(doc_id, doc_id))
        label = f"{clean_heading(path[-1])} — {title}" if path else title
        if label in by_label:
            kept = by_label[label].chunks
            seen = {c["text"] for c in kept}
            kept.extend(c for c in topic.chunks if c["text"] not in seen)
        else:
            topic.label = label
            by_label[label] = topic
    return list(by_label.values())


def prioritize_topics(
    topics: list[SectionTopic], mastery: dict[str, Any], now: datetime, weak_threshold: int
) -> list[SectionTopic]:
    """
    Thứ tự spaced repetition: quá hạn ôn (quá hạn lâu nhất trước) -> chưa học ->
    còn yếu (chưa tới hạn) -> phần còn lại (sắp tới hạn trước). Giữ thứ tự tài
    liệu trong cùng nhóm. `mastery`: label -> MasteryRecord (hoặc object cùng field).
    """
    def key(t: SectionTopic):
        rec = mastery.get(t.label)
        if rec is None:
            return (1, 0.0, t.order)
        due = rec.next_review_due_at
        if due is not None and due <= now:
            return (0, -(now - due).total_seconds(), t.order)
        if (rec.mastery_score or 0) < weak_threshold:
            return (2, float(rec.mastery_score or 0), t.order)
        return (3, (due - now).total_seconds() if due else 0.0, t.order)

    return sorted(topics, key=key)
