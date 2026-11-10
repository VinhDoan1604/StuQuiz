"""
app/ingestion/parsers.py

parse_document(raw_bytes, filename) -> ParsedDocument

Trích xuất nội dung PDF / DOCX / PPTX, cố giữ lại cấu trúc ngữ nghĩa
(heading, bảng, list, speaker notes) thay vì chỉ lấy text thô — vì
semantic_chunker.py ở bước sau dùng chính cấu trúc này làm ranh giới chunk.

Giới hạn đã biết (ghi rõ để review không bị bất ngờ khi gặp edge case):
- PDF (PyMuPDF): heading được nhận diện theo FONT — dòng có cỡ chữ lớn hơn
  cỡ chữ thân bài (cỡ chiếm nhiều ký tự nhất) là heading, cỡ càng lớn level
  càng cao; dòng in đậm cùng cỡ thân bài, đứng riêng một block, ngắn và không
  kết thúc bằng dấu câu là heading cấp thấp nhất. Các dòng trong cùng một
  block được nối lại thành một đoạn (PDF ngắt dòng vật lý giữa câu); dòng mở
  đầu bằng bullet/số thứ tự là LIST_ITEM, KHÔNG phải heading.
- Regex số thứ tự ("1.", "1.2", "Chương 1"...) chỉ dùng khi PDF không có
  khác biệt font nào (font đồng nhất) hoặc khi phải lùi về pypdf (không có
  font metadata). Khi đó heuristic có thể bỏ sót/nhầm heading -> chunker
  vẫn đúng (chunk theo token), chỉ mất lợi thế "chunk theo section".
- OCR fallback dùng PyMuPDF (fitz) chỉ để render trang PDF -> ảnh PNG
  (không dùng fitz để extract text, giữ đúng yêu cầu "pypdf để đọc text").
"""
from __future__ import annotations

import io
import logging
import re
from collections import Counter
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import settings
from app.ingestion.ocr import OCR_MIN_CONFIDENCE, OCRError, ocr_page_image
from app.ingestion.text_parser import parse_text
from app.models.schemas import BlockType, ParsedBlock, ParsedDocument

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


# ---------------------------------------------------------------------------
# Heading theo font cho PDF (PyMuPDF có size/flags của từng span)
# ---------------------------------------------------------------------------
_HEADING_SIZE_RATIO = 1.12      # cỡ chữ >= 1.12 x thân bài -> heading
_MAX_FONT_HEADING_CHARS = 160   # heading xuống dòng 2 lần vẫn lọt
_MAX_BOLD_HEADING_CHARS = 100
_MAX_HEADING_LEVEL = 6
_BOLD_FLAG = 16
_LIST_ITEM_RE = re.compile(r"^\s*(?:[•◦▪▫●○■□➢►\-–*]|\d{1,3}[.)]|[a-zA-Z][.)])\s+\S")
_SENTENCE_END = (".", ",", ":", ";", "?", "!")
_PAGE_MARGIN_RATIO = 0.08       # block nằm trọn trong 8% trên/dưới trang = vùng header/footer


@dataclass
class _PdfLine:
    text: str
    size: float
    bold: bool
    in_margin: bool = False         # block chứa dòng nằm ở lề trên/dưới trang
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # bbox của block chứa dòng


@dataclass
class _FontStyle:
    body_size: float
    size_levels: dict[float, int]   # cỡ chữ heading -> level (1 = lớn nhất)
    bold_level: int                 # level cho heading in đậm cỡ thân bài

    @property
    def has_font_headings(self) -> bool:
        return bool(self.size_levels)


def _page_lines_by_block(page) -> list[list[_PdfLine]]:
    """Các dòng text của 1 trang, nhóm theo block, theo thứ tự đọc (trên->dưới)."""
    out: list[list[_PdfLine]] = []
    height = float(page.rect.height) or 1.0
    for block in page.get_text("dict", sort=True).get("blocks", []):
        if block.get("type") != 0:
            continue
        _, y0, _, y1 = block.get("bbox", (0, 0, 0, 0))
        in_margin = y1 <= height * _PAGE_MARGIN_RATIO or y0 >= height * (1 - _PAGE_MARGIN_RATIO)
        lines: list[_PdfLine] = []
        for line in block.get("lines", []):
            spans = [sp for sp in line.get("spans", []) if sp.get("text", "").strip()]
            if not spans:
                continue
            text = "".join(sp["text"] for sp in line["spans"]).strip()
            main = max(spans, key=lambda sp: len(sp["text"].strip()))
            bold = all((sp.get("flags", 0) & _BOLD_FLAG) or "bold" in sp.get("font", "").lower() for sp in spans)
            lines.append(_PdfLine(
                text=text, size=round(float(main["size"]), 1), bold=bold, in_margin=in_margin,
                bbox=tuple(block.get("bbox", (0, 0, 0, 0))),
            ))
        if lines:
            out.append(lines)
    return out


def _analyze_font_style(pages: list[list[list[_PdfLine]]]) -> _FontStyle:
    chars: Counter[float] = Counter()
    for page in pages:
        for block in page:
            for ln in block:
                chars[ln.size] += len(ln.text)
    if not chars:
        return _FontStyle(body_size=0.0, size_levels={}, bold_level=1)
    body = chars.most_common(1)[0][0]
    heading_sizes = sorted(
        {
            ln.size
            for page in pages for block in page for ln in block
            if ln.size >= body * _HEADING_SIZE_RATIO and len(ln.text) <= _MAX_FONT_HEADING_CHARS
        },
        reverse=True,
    )
    levels = {size: min(i + 1, _MAX_HEADING_LEVEL) for i, size in enumerate(heading_sizes)}
    return _FontStyle(body_size=body, size_levels=levels, bold_level=min(len(levels) + 1, _MAX_HEADING_LEVEL))


@dataclass
class _PdfTable:
    bbox: tuple[float, float, float, float]
    text: str


def _rows_to_text(rows: list[list[str | None]]) -> str:
    """Bảng -> text, mỗi hàng một dòng "ô | ô | ô" (cùng định dạng với bảng DOCX/PPTX)."""
    out = []
    for row in rows:
        cells = [re.sub(r"\s+", " ", cell or "").strip() for cell in row]
        if any(cells):
            out.append(" | ".join(cells))
    return "\n".join(out)


def _find_pdf_tables(page) -> list[_PdfTable]:
    """Bảng có kẻ khung trên trang (PyMuPDF find_tables). Bỏ "bảng" 1 hàng/1 cột —
    thường là khung viền quanh hộp chú thích chứ không phải bảng dữ liệu."""
    try:
        found = page.find_tables().tables
    except Exception as exc:  # noqa: BLE001 — nhận diện bảng lỗi thì vẫn parse text bình thường
        logger.warning("find_tables lỗi ở trang %d: %s", page.number + 1, exc)
        return []
    tables = []
    for tb in found:
        if tb.row_count < 2 or tb.col_count < 2:
            continue
        text = _rows_to_text(tb.extract())
        if text:
            tables.append(_PdfTable(bbox=tuple(tb.bbox), text=text))
    return tables


def _inside(bbox: tuple[float, float, float, float], box: tuple[float, float, float, float]) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return box[0] - 1 <= cx <= box[2] + 1 and box[1] - 1 <= cy <= box[3] + 1


def _join_lines(parts: list[str]) -> str:
    text = ""
    for part in parts:
        if not text:
            text = part
        elif text.endswith("-") and not text.endswith(" -"):
            text += part           # "non-" + "volatile" -> "non-volatile"
        else:
            text += " " + part
    return text


def _blocks_from_font_page(
    page_blocks: list[list[_PdfLine]], style: _FontStyle, page_idx: int, order: int,
    tables: list[_PdfTable] | None = None,
) -> tuple[list[ParsedBlock], int]:
    """Một trang -> ParsedBlock: heading theo font, đoạn văn = các dòng của 1 block nối lại,
    bảng = 1 block TABLE đặt đúng vị trí đọc (text nằm trong bảng không bị lặp lại)."""
    result: list[ParsedBlock] = []
    pending_tables = sorted(tables or [], key=lambda t: t.bbox[1])

    def emit(kind: BlockType, text: str, level: int | None = None) -> None:
        nonlocal order
        if text:
            result.append(ParsedBlock(type=kind, text=text, page=page_idx, level=level, order=order))
            order += 1

    for lines in page_blocks:
        if any(_inside(lines[0].bbox, t.bbox) for t in tables or []):
            continue
        while pending_tables and pending_tables[0].bbox[1] <= lines[0].bbox[1]:
            emit(BlockType.TABLE, pending_tables.pop(0).text)
        # Running header/footer (tên chương, số trang, "Access for free at ..."): chữ nhỏ hơn
        # thân bài ở lề trang. Bắt được cả header xen kẽ trang chẵn/lẻ mà cleaning bỏ sót.
        if lines[0].in_margin and max(ln.size for ln in lines) < style.body_size:
            continue
        block_text = _join_lines([ln.text for ln in lines])
        standalone_bold = (
            all(ln.bold for ln in lines)
            and len(lines) <= 2
            and abs(lines[0].size - style.body_size) < 0.6
            and len(block_text) <= _MAX_BOLD_HEADING_CHARS
            and not block_text.endswith(_SENTENCE_END)
        )
        if standalone_bold:
            emit(BlockType.HEADING, block_text, style.bold_level)
            continue

        heading_parts: list[str] = []
        heading_level: int | None = None
        para_parts: list[str] = []
        para_kind = BlockType.PARAGRAPH

        def flush_para() -> None:
            nonlocal para_parts, para_kind
            emit(para_kind, _join_lines(para_parts))
            para_parts, para_kind = [], BlockType.PARAGRAPH

        def flush_heading() -> None:
            nonlocal heading_parts, heading_level
            emit(BlockType.HEADING, _join_lines(heading_parts), heading_level)
            heading_parts, heading_level = [], None

        for ln in lines:
            level = style.size_levels.get(ln.size) if len(ln.text) <= _MAX_FONT_HEADING_CHARS else None
            if level is not None:
                flush_para()
                if heading_parts and level != heading_level:
                    flush_heading()
                heading_parts.append(ln.text)   # heading dài bị xuống dòng -> gộp lại
                heading_level = level
                continue
            flush_heading()
            if _LIST_ITEM_RE.match(ln.text):
                flush_para()
                para_kind = BlockType.LIST_ITEM
            para_parts.append(ln.text)
        flush_heading()
        flush_para()
    for table in pending_tables:
        emit(BlockType.TABLE, table.text)
    return result, order


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
        if settings.max_pdf_pages and total_pages > settings.max_pdf_pages:
            raise ParsingError(f"PDF có {total_pages} trang, vượt giới hạn {settings.max_pdf_pages} trang.")
        ocr_pages = 0
        page_lines = [_page_lines_by_block(doc[i]) for i in range(total_pages)]
        style = _analyze_font_style(page_lines)

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
                ocr_pages += 1
                if settings.max_ocr_pages and ocr_pages > settings.max_ocr_pages:
                    raise ParsingError(
                        f"PDF scan cần OCR quá {settings.max_ocr_pages} trang. Hãy tách file nhỏ hơn."
                    )
                try:
                    zoom = OCR_RENDER_DPI / 72
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                    image_bytes = pix.tobytes("png")
                    ocr_result = ocr_page_image(image_bytes, lang=settings.ocr_lang)
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

            if style.has_font_headings:
                page_blocks, order = _blocks_from_font_page(
                    page_lines[page_idx - 1], style, page_idx, order, _find_pdf_tables(page)
                )
                blocks.extend(page_blocks)
                continue

            # Font đồng nhất (không phân biệt được heading qua cỡ chữ) -> regex theo dòng
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

    except ParsingError:
        raise  # vượt giới hạn: không lùi về pypdf
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
    """Bảng DOCX/PPTX (python-docx / python-pptx đều có .rows[].cells[].text)."""
    return _rows_to_text([[cell.text for cell in row.cells] for row in table.rows])


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
# Placeholder lặp lại mỗi slide, không phải nội dung bài giảng
_PPTX_SKIP_PLACEHOLDERS = {"FOOTER", "SLIDE_NUMBER", "DATE"}
_MIN_OCR_IMAGE_PX = 150        # bỏ qua icon/logo nhỏ
_MIN_OCR_IMAGE_CHARS = 15      # OCR ra ít hơn số ký tự chữ/số này -> coi như ảnh không có chữ
_MIN_OCR_WORD_CONFIDENCE = 70  # chỉ giữ từ OCR chắc chắn; lưới ô/đường kẻ trong sơ đồ sinh nhiều từ rác


def _iter_pptx_shapes(shapes):
    """Duyệt shape theo thứ tự trên slide, đi vào cả group shape lồng nhau."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_pptx_shapes(shape.shapes)
        else:
            yield shape


def _ocr_slide_image(shape, slide_idx: int, warnings: list[str]) -> tuple[str, float] | None:
    """OCR chữ trong hình (sơ đồ, bảng chụp màn hình). None nếu ảnh nhỏ / không có chữ / OCR lỗi."""
    try:
        image = shape.image
        width, height = image.size
    except Exception:  # noqa: BLE001 — ảnh liên kết ngoài / định dạng lạ (EMF, WMF)
        return None
    if min(width, height) < _MIN_OCR_IMAGE_PX:
        return None
    try:
        result = ocr_page_image(image.blob, lang=settings.ocr_lang, min_word_confidence=_MIN_OCR_WORD_CONFIDENCE)
    except OCRError as exc:
        warnings.append(f"Slide {slide_idx}: không OCR được hình ({exc}).")
        return None
    except Exception as exc:  # noqa: BLE001 — định dạng ảnh Pillow không mở được
        logger.info("Slide %d: bỏ qua hình không đọc được (%s).", slide_idx, exc)
        return None
    text = result.text.strip()
    if result.low_confidence or len(re.findall(r"\w", text)) < _MIN_OCR_IMAGE_CHARS:
        return None
    return text, result.confidence


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
    warnings: list[str] = []
    order = 0

    def add(kind: BlockType, text: str, slide_idx: int, **kw) -> None:
        nonlocal order
        blocks.append(ParsedBlock(type=kind, text=text, page=slide_idx, order=order, **kw))
        order += 1

    for slide_idx, slide in enumerate(prs.slides, start=1):
        title_shape = slide.shapes.title
        if title_shape is not None:
            title_text = (title_shape.text or "").strip()
            if title_text:
                add(BlockType.HEADING, title_text, slide_idx, level=1)

        for shape in _iter_pptx_shapes(slide.shapes):
            if title_shape is not None and shape.shape_id == title_shape.shape_id:
                continue
            if shape.is_placeholder and shape.placeholder_format.type.name in _PPTX_SKIP_PLACEHOLDERS:
                continue
            if getattr(shape, "has_table", False) and shape.has_table:
                table_text = _table_to_text(shape.table)
                if table_text:
                    add(BlockType.TABLE, table_text, slide_idx)
                continue
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        add(BlockType.LIST_ITEM, text, slide_idx)
                continue
            if settings.ocr_slide_images and hasattr(shape, "image"):
                ocr = _ocr_slide_image(shape, slide_idx, warnings)
                if ocr:
                    add(BlockType.PARAGRAPH, ocr[0], slide_idx,
                        extra={"source": "ocr_image", "ocr_confidence": ocr[1]})

        if slide.has_notes_slide:
            notes_text = (slide.notes_slide.notes_text_frame.text or "").strip()
            if notes_text:
                add(BlockType.SPEAKER_NOTES, notes_text, slide_idx, extra={"source": "speaker_notes"})

    if not blocks:
        warnings.append("Không trích xuất được nội dung nào từ PPTX.")

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
