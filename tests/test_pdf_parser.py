"""PDF parser: heading theo font, gộp dòng thành đoạn, bỏ running header/footer."""
import fitz

from app.ingestion.parsers import parse_document
from app.models.schemas import BlockType

BODY = 11


def _pdf(pages: list[list[tuple]]) -> bytes:
    """Mỗi item: (y, text, fontsize, bold). Mỗi item = 1 lệnh insert_text = 1 block trong PDF."""
    doc = fitz.open()
    for items in pages:
        page = doc.new_page(width=595, height=842)
        for y, text, size, bold in items:
            page.insert_text((72, y), text, fontsize=size, fontname="hebo" if bold else "helv")
    data = doc.tobytes()
    doc.close()
    return data


def _body_page(y0: float = 200) -> list[tuple]:
    # Đủ chữ thân bài để cỡ BODY là cỡ chiếm đa số
    return [(y0 + i * 40, "Body text sentence that is long enough to dominate the size count.\n"
             "It continues on a second physical line of the same paragraph.", BODY, False) for i in range(8)]


def _blocks(data: bytes):
    return parse_document(data, "t.pdf").blocks


def test_font_size_headings_get_levels_and_numbered_items_are_list_items():
    data = _pdf([[
        (80, "Chapter 1 Graphs", 22, False),
        (120, "1.1 Shortest paths", 15, False),
        (150, "1. Read the question first and find the keywords in the table.", BODY, False),
        (170, "2. Long case questions follow the five step procedure.", BODY, False),
        *_body_page(),
    ]])
    blocks = _blocks(data)
    heads = [(b.text, b.level) for b in blocks if b.type == BlockType.HEADING]
    assert heads == [("Chapter 1 Graphs", 1), ("1.1 Shortest paths", 2)]
    items = [b.text for b in blocks if b.type == BlockType.LIST_ITEM]
    assert items[0].startswith("1. Read the question") and items[1].startswith("2. Long case")


def test_lines_of_one_block_are_joined_into_one_paragraph():
    data = _pdf([[(80, "Title", 20, False), *_body_page()]])
    paras = [b.text for b in _blocks(data) if b.type == BlockType.PARAGRAPH]
    assert paras[0] == ("Body text sentence that is long enough to dominate the size count. "
                        "It continues on a second physical line of the same paragraph.")


def test_standalone_bold_line_is_lowest_level_heading_but_bold_label_with_colon_is_not():
    data = _pdf([[
        (80, "Title", 20, False),
        (110, "Binary Search Trees", BODY, True),
        (140, "Common pitfalls:", BODY, True),
        *_body_page(),
    ]])
    blocks = _blocks(data)
    heads = {b.text: b.level for b in blocks if b.type == BlockType.HEADING}
    assert heads == {"Title": 1, "Binary Search Trees": 2}
    assert any(b.type == BlockType.PARAGRAPH and b.text == "Common pitfalls:" for b in blocks)


def test_small_running_header_and_footer_in_margins_are_dropped_but_small_caption_kept():
    pages = []
    for n in range(1, 5):  # header xen kẽ chẵn/lẻ như sách OpenStax
        header = f"{n} Chapter 3 Data Structures" if n % 2 == 0 else f"3.{n} Algorithms {n}"
        pages.append([
            (30, header, 7, True),
            (80, f"Section {n}", 18, False),
            (400, "Figure 3.1 A small caption in the middle of the page.", 7, False),
            *_body_page(),
            (820, "Access for free at openstax.org", 7, False),
        ])
    texts = [b.text for b in _blocks(_pdf(pages))]
    assert not any("openstax.org" in t or "Chapter 3 Data Structures" in t or "Algorithms" in t for t in texts)
    assert sum("small caption" in t for t in texts) == 4


def test_uniform_font_falls_back_to_regex_headings():
    data = _pdf([[(80, "Chuong 1 Gioi thieu", BODY, False), *_body_page()]])
    heads = [b for b in _blocks(data) if b.type == BlockType.HEADING]
    assert [h.text for h in heads] == ["Chuong 1 Gioi thieu"]


def test_ruled_table_becomes_one_table_block_in_reading_order_without_duplicate_text():
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "Lookup table", fontsize=20, fontname="helv")
    page.insert_text((72, 130), "Text before the table.", fontsize=BODY, fontname="helv")
    rows = [("Keyword", "Section"), ("data mart", "1.4"), ("granularity", "2.6")]
    x0, y0, w, h = 72, 150, 200, 24
    for r, row in enumerate(rows):          # vẽ khung kẻ ô để find_tables nhận diện
        for c, cell in enumerate(row):
            rect = fitz.Rect(x0 + c * w, y0 + r * h, x0 + (c + 1) * w, y0 + (r + 1) * h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 4, rect.y1 - 7), cell, fontsize=BODY, fontname="helv")
    for i in range(6):
        page.insert_text((72, 260 + i * 40), "Body text after the table that dominates the font size count.",
                         fontsize=BODY, fontname="helv")
    blocks = parse_document(doc.tobytes(), "t.pdf").blocks
    kinds = [b.type for b in blocks]
    table = next(b for b in blocks if b.type == BlockType.TABLE)
    assert table.text == "Keyword | Section\ndata mart | 1.4\ngranularity | 2.6"
    assert kinds.index(BlockType.TABLE) == 2          # sau heading + đoạn "Text before", trước thân bài
    assert not any(b.type != BlockType.TABLE and "granularity" in b.text for b in blocks)
