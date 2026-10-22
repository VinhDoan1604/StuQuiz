"""
app/ingestion/parsers.py

parse_document(raw_bytes, filename) -> ParsedDocument

Trích xuất nội dung PDF / DOCX / PPTX, cố giữ lại cấu trúc ngữ nghĩa
(heading, bảng, list, speaker notes) thay vì chỉ lấy text thô — vì
semantic_chunker.py ở bước sau dùng chính cấu trúc này làm ranh giới chunk.

Giới hạn đã biết (ghi rõ để review không bị bất ngờ khi gặp edge case):
- pypdf không cung cấp font-size/font-weight, nên heading trong PDF được
  nhận diện bằng heuristic regex (số thứ tự chương/mục kiểu văn phong học
  thuật VN, dòng ngắn viết hoa toàn bộ...). Với PDF không theo văn phong
  này (vd. sách dịch, format nước ngoài), heuristic có thể bỏ sót heading
  -> khi đó cả trang sẽ rơi vào section "(không có heading)" và chunker
  vẫn hoạt động đúng (chunk theo token trong 1 section lớn), chỉ là mất
  lợi thế "chunk theo section" mà thôi — không crash, không mất dữ liệu.
- OCR fallback dùng PyMuPDF (fitz) chỉ để render trang PDF -> ảnh PNG
  (không dùng fitz để extract text, giữ đúng yêu cầu "pypdf để đọc text").
"""
from __future__ import annotations

import io
import logging
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.ingestion.ocr import OCR_MIN_CONFIDENCE, OCRError, ocr_page_image
from app.models.schemas import BlockType, ParsedBlock, ParsedDocument
from app.ingestion.text_parser import parse_text

logger = logging.getLogger(__name__)

# Trang được coi là "không có text layer" (khả năng là scan/ảnh) nếu số
# ký tự non-whitespace ít hơn ngưỡng này -> kích hoạt OCR cho trang đó.
MIN_CHARS_PER_PAGE = 20

# DPI dùng khi render trang PDF -> ảnh cho OCR. 200-300 là điểm cân bằng
# giữa độ chính xác OCR và thời gian xử lý/kích thước ảnh.
OCR_RENDER_DPI = 250


class UnsupportedFileTypeError(ValueError):
    """Đuôi file không nằm trong danh sách hỗ trợ (pdf/docx/pptx)."""


class ParsingError(RuntimeError):
    """File bị lỗi/hỏng, hoặc thiếu dependency cần thiết để parse."""


# ---------------------------------------------------------------------------
# Heading heuristics cho PDF (pypdf không có font metadata)
# ---------------------------------------------------------------------------
_HEADING_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"^\s*ch(ư|u)(ơ|o)ng\s+\d+", re.IGNORECASE), 1),
    (re.compile(r"^\s*ph(ầ|a)n\s+[ivxlc\d]+", re.IGNORECASE), 1),
    (re.compile(r"^\s*(m(ụ|u)c|b(à|a)i)\s+\d+", re.IGNORECASE), 2),
    (re.compile(r"^\s*\d+\.\d+\.\d+\.?\s+\S"), 3),   # vd: 1.2.3 Tiêu đề con
    (re.compile(r"^\s*\d+\.\d+\.?\s+\S"), 2),        # vd: 1.2 Tiêu đề
    (re.compile(r"^\s*\d+\.\s+\S"), 1),              # vd: 1. Tiêu đề
    (re.compile(r"^\s*[ivxlc]+\.\s+\S", re.IGNORECASE), 1),  # vd: I. Tiêu đề
]

_MAX_HEADING_LEN = 100  # dòng dài hơn mức này khó là heading thật


def _classify_line(line: str) -> int | None:
    """Trả về heading level (1-3) nếu dòng khớp pattern, None nếu không."""
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_LEN:
        return None
    for pattern, level in _HEADING_PATTERNS:
        if pattern.match(stripped):
            return level
    # Dòng toàn chữ hoa, ngắn, không kết thúc bằng dấu câu -> khả năng là heading
    if (
        stripped.isupper()
        and len(stripped.split()) <= 12
        and not stripped.endswith((".", ",", ":", ";"))
    ):
        return 2
    return None


def _render_pdf_page_to_png(raw_bytes: bytes, page_index: int, dpi: int = OCR_RENDER_DPI) -> bytes:
    """Render 1 trang PDF (0-indexed) thành ảnh PNG bytes, dùng PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ParsingError(
            "PyMuPDF (pymupdf) chưa được cài — cần để render trang PDF scan "
            "thành ảnh trước khi OCR. Chạy `pip install pymupdf`."
        ) from exc

    doc = fitz.open(stream=raw_bytes, filetype="pdf")
    try:
        page = doc[page_index]
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        return pix.tobytes("png")
    finally:
        doc.close()


def _parse_pdf(raw_bytes: bytes, filename: str) -> ParsedDocument:
    blocks: list[ParsedBlock] = []
    warnings: list[str] = []
    order = 0
    total_pages = 0

    # Ưu tiên dùng PyMuPDF (fitz) vì viết bằng C++, trích xuất cực nhanh (< 0.1s)
    # và chính xác hơn hẳn so với pypdf thuần Python.
    try:
        import fitz
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        total_pages = len(doc)

        for page_idx in range(1, total_pages + 1):
            page = doc[page_idx - 1]
            page_text = page.get_text() or ""
            stripped_len = len(re.sub(r"\s", "", page_text))

            if stripped_len < MIN_CHARS_PER_PAGE:
                # Chỉ kích hoạt OCR nếu trang thực sự có chứa ảnh/scan (tránh trang trắng tốn CPU OCR vô ích)
                images = page.get_images()
                if not images:
                    continue

                logger.info(
                    "Trang %d của %s có %d ảnh nhưng chỉ có %d ký tự text — kích hoạt OCR fallback.",
                    page_idx, filename, len(images), stripped_len,
                )
                try:
                    zoom = OCR_RENDER_DPI / 72
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                    image_bytes = pix.tobytes("png")
                    ocr_result = ocr_page_image(image_bytes, lang="vie")
                except (OCRError, ParsingError, Exception) as exc:
                    warnings.append(f"Trang {page_idx}: OCR fallback thất bại ({exc}).")
                    continue

                if ocr_result.low_confidence:
                    warnings.append(
                        f"Trang {page_idx}: OCR confidence thấp "
                        f"({ocr_result.confidence:.1f} < {OCR_MIN_CONFIDENCE}) — cần review thủ công."
                    )

                if ocr_result.text.strip():
                    blocks.append(
                        ParsedBlock(
                            type=BlockType.PARAGRAPH,
                            text=ocr_result.text,
                            page=page_idx,
                            order=order,
                            extra={
                                "source": "ocr",
                                "ocr_confidence": ocr_result.confidence,
                                "ocr_low_confidence": ocr_result.low_confidence,
                            },
                        )
                    )
                    order += 1
                continue

            for line in page_text.split("\n"):
                if not line.strip():
                    continue
                level = _classify_line(line)
                block_type = BlockType.HEADING if level else BlockType.PARAGRAPH
                blocks.append(
                    ParsedBlock(
                        type=block_type,
                        text=line.strip(),
                        page=page_idx,
                        level=level,
                        order=order,
                    )
                )
                order += 1

        doc.close()

    except Exception as fitz_err:
        logger.warning("PyMuPDF parser gặp lỗi (%s), chuyển sang fallback pypdf...", fitz_err)
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
        except PdfReadError as exc:
            raise ParsingError(f"File PDF lỗi hoặc không đọc được: {exc}") from exc

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ParsingError(
                    "PDF được mã hoá/bảo vệ mật khẩu và không thể trích xuất tự động."
                ) from exc

        total_pages = len(reader.pages)
        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("Lỗi extract_text() trang %d của %s: %s", page_idx, filename, exc)
                page_text = ""

            stripped_len = len(re.sub(r"\s", "", page_text))
            if stripped_len < MIN_CHARS_PER_PAGE:
                continue

            for line in page_text.split("\n"):
                if not line.strip():
                    continue
                level = _classify_line(line)
                block_type = BlockType.HEADING if level else BlockType.PARAGRAPH
                blocks.append(
                    ParsedBlock(
                        type=block_type,
                        text=line.strip(),
                        page=page_idx,
                        level=level,
                        order=order,
                    )
                )
                order += 1

    if not blocks:
        warnings.append("Không trích xuất được nội dung nào từ toàn bộ PDF.")

    return ParsedDocument(
        filename=filename,
        file_type="pdf",
        blocks=blocks,
        total_pages=total_pages,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------
_DOCX_HEADING_RE = re.compile(r"^Heading\s*(\d)$", re.IGNORECASE)


def _table_to_text(table) -> str:
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _parse_docx(raw_bytes: bytes, filename: str) -> ParsedDocument:
    try:
        import docx
        from docx.oxml.ns import qn
    except ImportError as exc:
        raise ParsingError("python-docx chưa được cài. Chạy `pip install python-docx`.") from exc

    try:
        document = docx.Document(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ParsingError(f"File DOCX lỗi hoặc không đọc được: {exc}") from exc

    blocks: list[ParsedBlock] = []
    order = 0

    # Duyệt theo đúng thứ tự xuất hiện thật trong document.xml (paragraph
    # và table có thể xen kẽ nhau) thay vì duyệt document.paragraphs rồi
    # document.tables riêng — tránh đảo lộn thứ tự gây sai ngữ cảnh chunk.
    paragraph_map = {p._element: p for p in document.paragraphs}
    table_map = {t._element: t for t in document.tables}

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p") and child in paragraph_map:
            para = paragraph_map[child]
            text = para.text.strip()
            if not text:
                continue
            match = _DOCX_HEADING_RE.match(para.style.name or "") if para.style else None
            if match:
                blocks.append(
                    ParsedBlock(type=BlockType.HEADING, text=text, level=int(match.group(1)), order=order)
                )
            else:
                blocks.append(ParsedBlock(type=BlockType.PARAGRAPH, text=text, order=order))
            order += 1
        elif child.tag == qn("w:tbl") and child in table_map:
            table_text = _table_to_text(table_map[child])
            if table_text.strip():
                blocks.append(ParsedBlock(type=BlockType.TABLE, text=table_text, order=order))
                order += 1

    warnings = [] if blocks else ["Không trích xuất được nội dung nào từ DOCX."]

    return ParsedDocument(filename=filename, file_type="docx", blocks=blocks, total_pages=0, warnings=warnings)


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------
def _parse_pptx(raw_bytes: bytes, filename: str) -> ParsedDocument:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise ParsingError("python-pptx chưa được cài. Chạy `pip install python-pptx`.") from exc

    try:
        prs = Presentation(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ParsingError(f"File PPTX lỗi hoặc không đọc được: {exc}") from exc

    blocks: list[ParsedBlock] = []
    order = 0

    for slide_idx, slide in enumerate(prs.slides, start=1):
        title_shape = slide.shapes.title
        if title_shape is not None:
            title_text = (title_shape.text or "").strip()
            if title_text:
                blocks.append(
                    ParsedBlock(type=BlockType.HEADING, text=title_text, page=slide_idx, level=1, order=order)
                )
                order += 1

        for shape in slide.shapes:
            if shape == title_shape or not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                text = "".join(run.text for run in para.runs).strip()
                if not text:
                    continue
                blocks.append(ParsedBlock(type=BlockType.LIST_ITEM, text=text, page=slide_idx, order=order))
                order += 1

        if slide.has_notes_slide:
            notes_text = (slide.notes_slide.notes_text_frame.text or "").strip()
            if notes_text:
                blocks.append(
                    ParsedBlock(
                        type=BlockType.SPEAKER_NOTES,
                        text=notes_text,
                        page=slide_idx,
                        order=order,
                        extra={"source": "speaker_notes"},
                    )
                )
                order += 1

    warnings = [] if blocks else ["Không trích xuất được nội dung nào từ PPTX."]

    return ParsedDocument(
        filename=filename,
        file_type="pptx",
        blocks=blocks,
        total_pages=len(prs.slides),
        warnings=warnings,
    )

def _parse_text(raw_bytes: bytes, filename: str) -> ParsedDocument:
    return parse_text(raw_bytes, filename, _classify_line)
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
_EXTENSION_MAP = {
    "pdf": _parse_pdf,
    "docx": _parse_docx,
    "pptx": _parse_pptx,
    "txt": _parse_text,
    "md": _parse_text,
}


def parse_document(raw_bytes: bytes, filename: str) -> ParsedDocument:
    """
    Điểm vào chính của module. Suy luận file_type từ phần mở rộng của
    `filename` và gọi parser tương ứng.

    Raises:
        UnsupportedFileTypeError: đuôi file không nằm trong pdf/docx/pptx.
        ParsingError: file bị lỗi/hỏng, hoặc thiếu dependency cần thiết.
    """
    if not raw_bytes:
        raise ParsingError("File rỗng, không có nội dung để xử lý.")

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    parser = _EXTENSION_MAP.get(ext)
    if parser is None:
        raise UnsupportedFileTypeError(
            f"Định dạng '.{ext}' chưa được hỗ trợ. Chỉ hỗ trợ: {list(_EXTENSION_MAP)}."
        )

    logger.info("Bắt đầu parse '%s' (%s, %d bytes).", filename, ext, len(raw_bytes))
    result = parser(raw_bytes, filename)
    logger.info(
        "Parse xong '%s': %d block(s), %d warning(s).", filename, len(result.blocks), len(result.warnings)
    )
    return result