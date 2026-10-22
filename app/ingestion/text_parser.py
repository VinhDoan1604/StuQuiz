"""
app/ingestion/text_parser.py

Parser cho .txt / .md (UI đã cho chọn 2 định dạng này nhưng parsers.py chưa hỗ trợ
=> trước đây upload trả HTTP 500).

- Markdown: `#`..`######` -> HEADING (level đúng), `-`/`*`/`1.` -> LIST_ITEM,
  dòng bắt đầu bằng `|` -> TABLE, khối ``` -> 1 PARAGRAPH giữ nguyên xuống dòng.
- Văn bản thường: đoạn cách nhau bằng dòng trống; đoạn 1 dòng được thử nhận
  diện heading bằng `classify_line` (heuristic của parsers.py, truyền vào để
  tránh import vòng).
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from app.models.schemas import BlockType, ParsedBlock, ParsedDocument

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_MD_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*\S)\s*$")


def parse_text(
    raw_bytes: bytes,
    filename: str,
    classify_line: Optional[Callable[[str], Optional[int]]] = None,
) -> ParsedDocument:
    text = raw_bytes.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    is_md = filename.lower().endswith(".md")
    ext = "md" if is_md else "txt"

    blocks: list[ParsedBlock] = []
    order = 0
    para: list[str] = []
    table: list[str] = []
    code: list[str] = []
    in_code = False

    def add(btype: BlockType, content: str, level: Optional[int] = None) -> None:
        nonlocal order
        content = content.strip()
        if content:
            blocks.append(ParsedBlock(type=btype, text=content, level=level, order=order))
            order += 1

    def flush_para() -> None:
        if not para:
            return
        joined = " ".join(s.strip() for s in para)
        level = classify_line(joined) if (classify_line and len(para) == 1 and not is_md) else None
        add(BlockType.HEADING if level else BlockType.PARAGRAPH, joined, level)
        para.clear()

    def flush_table() -> None:
        if table:
            add(BlockType.TABLE, "\n".join(table))
            table.clear()

    for line in text.split("\n"):
        if is_md and line.strip().startswith("```"):
            if in_code:
                add(BlockType.PARAGRAPH, "\n".join(code))
                code.clear()
            else:
                flush_para()
                flush_table()
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
            continue

        if not line.strip():
            flush_para()
            flush_table()
            continue

        if is_md:
            m = _MD_HEADING.match(line)
            if m:
                flush_para()
                flush_table()
                add(BlockType.HEADING, m.group(2), level=len(m.group(1)))
                continue
            if line.lstrip().startswith("|"):
                flush_para()
                table.append(line.strip())
                continue
            m = _MD_LIST.match(line)
            if m:
                flush_para()
                flush_table()
                add(BlockType.LIST_ITEM, m.group(1))
                continue
        para.append(line)

    if in_code and code:  # fence không đóng
        add(BlockType.PARAGRAPH, "\n".join(code))
    flush_para()
    flush_table()

    warnings = [] if blocks else [f"Không trích xuất được nội dung nào từ {ext.upper()}."]
    return ParsedDocument(filename=filename, file_type=ext, blocks=blocks, total_pages=0, warnings=warnings)
