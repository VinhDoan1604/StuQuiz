"""
app/ingestion/cleaning.py

clean_document(doc) -> ParsedDocument

- Chuẩn hoá Unicode NFC cho toàn bộ text (một số pipeline OCR/scan hoặc
  macOS xuất text tiếng Việt ở dạng NFD — tổ hợp chữ cái + dấu rời — gây
  lỗi so khớp/search nếu không đồng nhất trước khi index).
- Khử header/footer lặp lại giữa các trang: so sánh dòng đầu tiên và dòng
  cuối cùng (theo thứ tự block) của mỗi trang; dòng nào lặp lại trên
  HEADER_FOOTER_THRESHOLD (mặc định 60%) số trang thì bị loại khỏi toàn bộ
  tài liệu.
- Xoá các dòng chỉ chứa số trang, và các pattern watermark rác phổ biến
  (danh sách này nên được mở rộng dần theo dữ liệu thực tế của project).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter

from app.models.schemas import ParsedDocument

logger = logging.getLogger(__name__)

# Dòng lặp lại trên tỉ lệ này (so với tổng số trang có nội dung) bị coi là
# header/footer rác (vd: tên trường/tên môn lặp lại mỗi trang slide bài giảng).
HEADER_FOOTER_THRESHOLD = 0.6

# Cần tối thiểu số trang này để thống kê tần suất lặp có ý nghĩa thống kê —
# tài liệu quá ngắn (1-2 trang) không nên áp dụng loại bỏ header/footer vì
# dễ nhầm nội dung thật thành boilerplate.
_MIN_PAGES_FOR_DETECTION = 3

_PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*\d{1,4}\s*$"),
    re.compile(r"^\s*(page|trang)\s+\d+(\s*(/|of|trên)\s*\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
]

# Watermark/boilerplate thường gặp. Mở rộng danh sách này khi phát hiện
# pattern mới trong dữ liệu thực tế (vd: tên miền website in mờ mỗi trang).
_WATERMARK_PATTERNS = [
    re.compile(r"^\s*(confidential|draft|bản nháp|tài liệu nội bộ|lưu hành nội bộ)\s*$", re.IGNORECASE),
]


def _normalize_for_comparison(text: str) -> str:
    """Chuẩn hoá nhẹ để so khớp header/footer: bỏ số cụ thể (vì header có
    thể chứa số trang thay đổi mỗi trang, vd 'Trang 3 - Môn Toán'), gộp
    khoảng trắng, hạ chữ thường."""
    text = unicodedata.normalize("NFC", text).strip().lower()
    text = re.sub(r"\d+", "#", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _is_page_number(text: str) -> bool:
    stripped = text.strip()
    return any(p.match(stripped) for p in _PAGE_NUMBER_PATTERNS)


def _is_watermark(text: str) -> bool:
    stripped = text.strip()
    return any(p.match(stripped) for p in _WATERMARK_PATTERNS)


def _detect_repeated_boundary_lines(doc: ParsedDocument) -> set[str]:
    """So sánh dòng đầu/cuối (theo `order`) của mỗi trang có nội dung; trả
    về tập text (đã normalize) xuất hiện ở > HEADER_FOOTER_THRESHOLD số
    trang -> coi là header/footer lặp cần loại bỏ."""
    pages: dict[int, list] = {}
    for block in doc.blocks:
        if block.page is None:
            continue
        pages.setdefault(block.page, []).append(block)

    if len(pages) < _MIN_PAGES_FOR_DETECTION:
        return set()

    boundary_counter: Counter[str] = Counter()
    for page_blocks in pages.values():
        page_blocks_sorted = sorted(page_blocks, key=lambda b: b.order)
        first_text = _normalize_for_comparison(page_blocks_sorted[0].text)
        last_text = _normalize_for_comparison(page_blocks_sorted[-1].text)
        if first_text:
            boundary_counter[first_text] += 1
        if last_text and last_text != first_text:
            boundary_counter[last_text] += 1

    total_pages = len(pages)
    return {
        text for text, count in boundary_counter.items()
        if count / total_pages > HEADER_FOOTER_THRESHOLD
    }


def clean_document(doc: ParsedDocument) -> ParsedDocument:
    """
    Trả về bản sao ParsedDocument đã làm sạch. Không mutate `doc` gốc, để
    giữ nguyên bản raw phục vụ audit/debug khi cần truy vết vì sao 1 chunk
    trong index lại có nội dung như vậy.
    """
    repeated_boundaries = _detect_repeated_boundary_lines(doc)
    if repeated_boundaries:
        logger.info(
            "'%s': phát hiện %d dòng header/footer lặp (>%.0f%% số trang), vd: %s",
            doc.filename, len(repeated_boundaries), HEADER_FOOTER_THRESHOLD * 100,
            list(repeated_boundaries)[:5],
        )

    cleaned_blocks = []
    removed_count = 0

    for block in doc.blocks:
        text = unicodedata.normalize("NFC", block.text).strip()
        if not text:
            removed_count += 1
            continue

        if _is_page_number(text) or _is_watermark(text):
            removed_count += 1
            continue

        if _normalize_for_comparison(text) in repeated_boundaries:
            removed_count += 1
            continue

        # Gộp khoảng trắng/newline thừa — thường gặp sau OCR hoặc export PDF lỗi font
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        cleaned_blocks.append(block.model_copy(update={"text": text}))

    warnings = list(doc.warnings)
    if removed_count:
        warnings.append(
            f"cleaning: đã loại {removed_count} block (header/footer lặp, số trang, watermark, hoặc rỗng sau normalize)."
        )
        logger.info("'%s': cleaning loại %d/%d block.", doc.filename, removed_count, len(doc.blocks))

    return doc.model_copy(update={"blocks": cleaned_blocks, "warnings": warnings})