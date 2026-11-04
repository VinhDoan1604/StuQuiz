"""PPTX parser: bỏ footer/số slide, đọc group + bảng, OCR chữ trong hình."""
import copy
import io

from PIL import Image
from pptx import Presentation
from pptx.util import Inches

import app.ingestion.parsers as parsers
from app.ingestion.ocr import OCRResult
from app.ingestion.parsers import parse_document
from app.models.schemas import BlockType


def _png(w: int, h: int) -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    buf.seek(0)
    return buf


def _deck() -> bytes:
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])          # Title Only
    slide.shapes.title.text = "Earned value"
    box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(4), Inches(1))
    box.text_frame.text = "CPI = EV / AC"
    group = slide.shapes.add_group_shape()
    group.shapes.add_textbox(Inches(1), Inches(3), Inches(3), Inches(1)).text_frame.text = "Inside a group"
    table = slide.shapes.add_table(2, 2, Inches(1), Inches(4), Inches(4), Inches(1)).table
    for (r, c), text in {(0, 0): "Metric", (0, 1): "Meaning", (1, 0): "SPI", (1, 1): "EV / PV"}.items():
        table.cell(r, c).text = text
    slide.shapes.add_picture(_png(600, 400), Inches(5), Inches(1))   # sơ đồ có chữ
    slide.shapes.add_picture(_png(40, 40), Inches(8), Inches(1))     # icon nhỏ -> không OCR
    s2 = prs.slides.add_slide(prs.slide_layouts[1])          # Title and Content
    s2.shapes.title.text = "Second"
    s2.placeholders[1].text_frame.text = "Body bullet"
    out = io.BytesIO()
    prs.save(out)
    return out.getvalue()


def test_pptx_reads_groups_tables_and_ocrs_only_large_images(monkeypatch):
    calls = []

    def fake_ocr(blob, lang="vie", min_word_confidence=0.0):
        calls.append((lang, min_word_confidence))
        return OCRResult(text="Steering committee Project manager", confidence=88.0, word_count=4,
                         low_confidence=False)

    monkeypatch.setattr(parsers, "ocr_page_image", fake_ocr)
    blocks = parse_document(_deck(), "deck.pptx").blocks
    texts = [(b.type, b.text) for b in blocks if b.page == 1]

    assert (BlockType.HEADING, "Earned value") in texts
    assert (BlockType.LIST_ITEM, "CPI = EV / AC") in texts
    assert (BlockType.LIST_ITEM, "Inside a group") in texts
    assert (BlockType.TABLE, "Metric | Meaning\nSPI | EV / PV") in texts
    ocr_blocks = [b for b in blocks if b.extra.get("source") == "ocr_image"]
    assert [b.text for b in ocr_blocks] == ["Steering committee Project manager"]
    assert len(calls) == 1 and calls[0][1] >= 50      # icon 40px bị bỏ qua; OCR hình có lọc theo từ


def test_pptx_skips_footer_slide_number_and_date_placeholders():
    prs = Presentation(io.BytesIO(_deck()))
    slide = prs.slides[1]
    # Chép placeholder footer / số slide / ngày từ layout sang slide (như slide bài giảng thật)
    for ph in prs.slide_layouts[1].placeholders:
        if ph.placeholder_format.type.name in {"FOOTER", "SLIDE_NUMBER", "DATE"}:
            slide.shapes._spTree.append(copy.deepcopy(ph._element))
    for shape in slide.placeholders:
        if shape.placeholder_format.type.name == "FOOTER":
            shape.text_frame.text = "Ch9. Monitoring and control"
        elif shape.placeholder_format.type.name == "DATE":
            shape.text_frame.text = "Aug 2024"
    out = io.BytesIO()
    prs.save(out)
    texts = [b.text for b in parse_document(out.getvalue(), "deck.pptx").blocks if b.page == 2]
    assert "Body bullet" in texts
    assert "Ch9. Monitoring and control" not in texts and "Aug 2024" not in texts


def test_low_confidence_or_textless_image_is_dropped(monkeypatch):
    monkeypatch.setattr(parsers, "ocr_page_image", lambda *a, **k: OCRResult(
        text="4 ~\\", confidence=42.0, word_count=2, low_confidence=False))
    blocks = parse_document(_deck(), "deck.pptx").blocks
    assert not any(b.extra.get("source") == "ocr_image" for b in blocks)
