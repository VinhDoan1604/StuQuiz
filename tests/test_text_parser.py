"""Test parser .txt/.md. pytest tests/test_text_parser.py -v"""
from app.ingestion.text_parser import parse_text
from app.models.schemas import BlockType

MD = """# Chương 1
Đoạn một dòng đầu
dòng hai cùng đoạn.

## Mục 1.1
- ý một
- ý hai

| a | b |
|---|---|
| 1 | 2 |

```
code  giữ nguyên
  thụt lề
```
"""


def test_markdown_structure():
    doc = parse_text(MD.encode(), "x.md")
    kinds = [(b.type, b.level) for b in doc.blocks]
    assert kinds[0] == (BlockType.HEADING, 1)
    assert doc.blocks[1].type == BlockType.PARAGRAPH and "dòng hai" in doc.blocks[1].text
    assert (BlockType.HEADING, 2) in kinds
    assert [b.text for b in doc.blocks if b.type == BlockType.LIST_ITEM] == ["ý một", "ý hai"]
    assert any(b.type == BlockType.TABLE and "| 1 | 2 |" in b.text for b in doc.blocks)
    assert any("thụt lề" in b.text and "\n" in b.text for b in doc.blocks)
    assert [b.order for b in doc.blocks] == list(range(len(doc.blocks)))
    assert doc.file_type == "md" and not doc.warnings


def test_txt_uses_classifier_for_single_line_headings():
    classify = lambda s: 1 if s.startswith("1.") else None  # noqa: E731
    doc = parse_text("1. Giới thiệu\n\nNội dung đoạn văn.".encode(), "x.txt", classify)
    assert doc.blocks[0].type == BlockType.HEADING and doc.blocks[0].level == 1
    assert doc.blocks[1].type == BlockType.PARAGRAPH


def test_handles_bom_crlf_and_empty():
    doc = parse_text("\ufeffA\r\nB\r\n".encode("utf-8"), "x.txt")
    assert doc.blocks[0].text == "A B"
    empty = parse_text(b"   \n\n", "x.txt")
    assert empty.blocks == [] and empty.warnings


def test_unclosed_code_fence_is_kept():
    doc = parse_text("```\nabc".encode(), "x.md")
    assert doc.blocks[0].text == "abc"
