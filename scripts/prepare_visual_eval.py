"""
scripts/prepare_visual_eval.py

Chuẩn bị dữ liệu cho eval OCR/VLM từ corpus có sẵn (eval/corpus/), không gọi LLM:

1. figure_table/: render các trang có hình raster hoặc bảng (nội dung hình KHÔNG có trong text
   layer) + candidates.json (tài liệu, trang, caption) -> nguồn cho câu hỏi "đáp án nằm trong hình".
2. synthetic_scan/: trang ghi chú tiếng Việt render rồi làm xấu đi như bản scan (xoay nhẹ, mờ,
   nhiễu, JPEG chất lượng thấp). Ground truth lấy từ text layer gốc theo thứ tự đọc (*.gt.txt),
   nên có CER/WER miễn phí. Đây là scan GIẢ LẬP: báo cáo phải tách riêng với scan/ảnh chụp thật.
3. Tạo sẵn thư mục scan/ và phone_photo/ cho ảnh thật (tự chụp/scan, ground truth sửa tay).

    python scripts/prepare_visual_eval.py            # ghi vào eval/data/ocr/
    python scripts/prepare_visual_eval.py --dpi 200
"""
from __future__ import annotations

import argparse
import io
import json
import random
import unicodedata
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

CORPUS = Path("eval/corpus")
OUT = Path("eval/data/ocr")

# Trang có hình raster (caption "Figure ...") hoặc bảng — chọn tay sau khi quét corpus bằng PyMuPDF
# (get_images / find_tables). Số trang tính từ 1.
FIGURE_TABLE_PAGES: dict[str, list[int]] = {
    "openstax_intro_cs_ch3_data_structures_algorithms.pdf": [1, 13, 20, 24, 26, 28, 31, 32, 33, 34],
    "openstax_intro_cs_ch8_data_management.pdf": [1, 12, 21, 25, 27, 38, 40, 43, 50],
    "notes_co4031_data_warehouse_dss.pdf": [3, 8, 48, 58, 60],
}

# Trang ghi chú tiếng Việt nhiều chữ, dùng làm scan giả lập.
SYNTHETIC_SCAN_PAGES: dict[str, list[int]] = {
    "notes_co4031_data_warehouse_dss.pdf": [5, 7, 9, 13, 20, 32, 38, 40, 56, 65],
}

LEVELS = {
    # angle (độ), blur radius, noise sigma, jpeg quality, scale (so với dpi gốc)
    "light": (0.4, 0.3, 5, 75, 0.9),
    "medium": (0.9, 0.6, 10, 55, 0.8),
    "heavy": (1.5, 0.8, 14, 40, 0.7),
}


def render(page: pymupdf.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def degrade(img: Image.Image, level: str, rng: random.Random) -> Image.Image:
    angle, blur, sigma, quality, scale = LEVELS[level]
    g = img.convert("L")
    g = g.resize((int(g.width * scale), int(g.height * scale)), Image.Resampling.BILINEAR)
    g = g.rotate(rng.uniform(-angle, angle), resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    g = g.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(g, dtype=np.float32)
    arr += np.random.default_rng(rng.randrange(1 << 30)).normal(0, sigma, arr.shape)
    g = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    g.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue()))


def captions(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip().startswith(("Figure", "Table"))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for sub in ("figure_table", "synthetic_scan", "scan", "phone_photo"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    candidates = []
    for name, pages in FIGURE_TABLE_PAGES.items():
        doc = pymupdf.open(CORPUS / name)
        for n in pages:
            page = doc[n - 1]
            out = OUT / "figure_table" / f"{Path(name).stem}_p{n:03d}.png"
            render(page, args.dpi).save(out, optimize=True)
            try:
                n_tables = len(page.find_tables().tables)
            except Exception:
                n_tables = 0
            candidates.append({
                "image": out.as_posix(),
                "document": name,
                "page": n,
                "raster_images": len(page.get_images(full=True)),
                "tables": n_tables,
                "captions": captions(page.get_text()),
                "question_idea": "",  # điền tay: câu hỏi mà đáp án chỉ có trong hình/bảng
            })
    (OUT / "figure_table" / "candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = []
    level_names = list(LEVELS)
    for name, pages in SYNTHETIC_SCAN_PAGES.items():
        doc = pymupdf.open(CORPUS / name)
        for i, n in enumerate(pages):
            level = level_names[i % len(level_names)]
            stem = f"{Path(name).stem}_p{n:03d}_{level}"
            degrade(render(doc[n - 1], args.dpi), level, rng).save(OUT / "synthetic_scan" / f"{stem}.jpg")
            # sort=True: thứ tự đọc thị giác (trên->dưới, trái->phải). Mặc định get_text() theo thứ tự
            # content stream của PDF, có thể đặt tiêu đề trang/bảng ở cuối -> phạt oan mọi engine.
            gt = unicodedata.normalize("NFC", doc[n - 1].get_text(sort=True)).strip()
            (OUT / "synthetic_scan" / f"{stem}.gt.txt").write_text(gt + "\n", encoding="utf-8")
            manifest.append({"image": f"{stem}.jpg", "ground_truth": f"{stem}.gt.txt",
                             "document": name, "page": n, "level": level})
    (OUT / "synthetic_scan" / "manifest.json").write_text(
        json.dumps({"seed": args.seed, "dpi": args.dpi, "levels": LEVELS, "items": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"figure_table: {len(candidates)} trang · synthetic_scan: {len(manifest)} ảnh -> {OUT}")


if __name__ == "__main__":
    main()
