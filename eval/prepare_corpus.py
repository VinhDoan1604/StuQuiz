"""
eval/prepare_corpus.py

Cắt một khoảng trang từ PDF dày (vd. sách OpenStax ~900 trang) thành file nhỏ cho corpus eval.

    python -m eval.prepare_corpus book.pdf --pages 120-165 --out eval/corpus/openstax_cs_ch4.pdf
    python -m eval.prepare_corpus book.pdf --toc          # in mục lục kèm số trang để chọn chương

Số trang tính từ 1 (giống trình đọc PDF), khoảng bao gồm cả hai đầu.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF


def parse_range(spec: str, n_pages: int) -> tuple[int, int]:
    start_s, _, end_s = spec.partition("-")
    start, end = int(start_s), int(end_s or start_s)
    if not (1 <= start <= end <= n_pages):
        raise SystemExit(f"Khoảng trang {spec!r} không hợp lệ (PDF có {n_pages} trang).")
    return start, end


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--pages", help="vd. 120-165")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--toc", action="store_true", help="in mục lục (level 1-2) rồi thoát")
    args = ap.parse_args()

    src = fitz.open(args.pdf)
    if args.toc:
        for level, title, page in src.get_toc():
            if level <= 2:
                print(f"{'  ' * (level - 1)}{title}  ......  p.{page}")
        return
    if not args.pages or not args.out:
        raise SystemExit("Cần --pages và --out (hoặc dùng --toc).")

    start, end = parse_range(args.pages, src.page_count)
    out = fitz.open()
    out.insert_pdf(src, from_page=start - 1, to_page=end - 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out, garbage=4, deflate=True)
    print(f"Đã ghi {args.out} ({end - start + 1} trang, p.{start}-{end} của {args.pdf.name})")


if __name__ == "__main__":
    main()
