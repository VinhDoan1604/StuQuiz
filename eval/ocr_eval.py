"""
eval/ocr_eval.py — Tesseract vs VLM khi chép chữ trang scan, đo bằng CER/WER có CI 95 %.

    python scripts/prepare_visual_eval.py                                # tạo eval/data/ocr/ (1 lần)
    python -m eval.ocr_eval --engines tesseract --run-id ocr-tess        # miễn phí, không gọi LLM
    python -m eval.ocr_eval --engines tesseract vlm --run-id ocr-v1      # 1 lời gọi VLM / ảnh
    CHEAP_MODEL_PROVIDER=... CHEAP_MODEL_NAME=... python -m eval.ocr_eval --engines vlm --run-id ocr-v1-b
    python -m eval.ocr_eval --from-raw eval/results/ocr-v1 eval/results/ocr-v1-b --run-id ocr-v1-all  # gộp, không gọi lại

Dữ liệu: eval/data/ocr/synthetic_scan/manifest.json — trang ghi chú tiếng Việt render rồi làm
xấu như bản scan (light / medium / heavy), ground truth là text layer gốc. Đây là scan GIẢ LẬP:
không thay được ảnh scan / ảnh chụp thật, và text layer gốc có thể chứa ký tự mà ảnh render
không thể hiện rõ (công thức, ký hiệu) — sai số đó như nhau cho cả hai engine.

Engine gọi ĐÚNG đường code production: `ocr_page_image` (lang = OCR_LANG) và
`vlm_transcribe_page` (task "vision", CHEAP model). Mỗi ảnh chạy một lần, không chọn lại
prompt theo kết quả (không tinh chỉnh trên tập test).

Chỉ số (thấp hơn = tốt hơn):
- cer: lỗi ký tự, giữ dấu tiếng Việt, hoa/thường và dấu câu; khoảng trắng/ngắt dòng được gộp.
- cer_nodiac: như cer nhưng bỏ dấu — chênh lệch cer - cer_nodiac ~ phần lỗi do dấu.
- wer: lỗi từ (âm tiết).
CI: percentile bootstrap; hiệu VLM - Tesseract là paired bootstrap theo ảnh.

Kết quả: eval/results/<run-id>/{report.md, summary.json, meta.json, raw.jsonl}.
raw.jsonl chứa nguyên văn chữ chép từ tài liệu không được phân phối lại: KHÔNG commit.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.ingestion.ocr import ocr_page_image
from app.ingestion.vlm import VLM_TRANSCRIBE_SYSTEM_PROMPT, vlm_transcribe_page
from app.llm.router import begin_request_usage
from eval.metrics import bootstrap_ci, cer, paired_bootstrap_diff, wer

DATA_DIR = Path("eval/data/ocr/synthetic_scan")
RESULTS = Path("eval/results")
ENGINES = ("tesseract", "vlm")
METRICS = ("cer", "cer_nodiac", "wer")
LEVEL_ORDER = ("light", "medium", "heavy")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "describe", "--always", "--dirty"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def transcribe(engine: str, image: bytes) -> dict:
    """Chạy 1 engine trên 1 ảnh; trả về text + chi phí. Lỗi được ghi lại, không làm dừng cả run."""
    usage = begin_request_usage()
    started = time.perf_counter()
    try:
        if engine == "tesseract":
            res = ocr_page_image(image, lang=settings.ocr_lang)
            text, extra = res.text, {"tesseract_confidence": res.confidence}
        else:
            text, extra = vlm_transcribe_page(image).text, {}
        error = None
    except Exception as exc:  # noqa: BLE001
        text, extra, error = "", {}, f"{type(exc).__name__}: {exc}"
    return {
        "text": text, "error": error, "seconds": round(time.perf_counter() - started, 2),
        "tokens_in": usage["tokens_in"], "tokens_out": usage["tokens_out"], "fallbacks": usage["fallbacks"], **extra,
    }


def score(reference: str, hypothesis: str) -> dict[str, float]:
    return {
        "cer": round(cer(reference, hypothesis), 4),
        "cer_nodiac": round(cer(reference, hypothesis, strip_diacritics=True), 4),
        "wer": round(wer(reference, hypothesis), 4),
    }


def _fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} [{lo:.3f}–{hi:.3f}]"


def engine_label(engine: str) -> str:
    """'vlm' -> 'vlm:<model>' để gộp được nhiều VLM vào một báo cáo."""
    return f"vlm:{settings.cheap_model_name}" if engine == "vlm" else engine


def _engine_order(records: list[dict]) -> list[str]:
    seen = list(dict.fromkeys(r["engine"] for r in records))
    return sorted(seen, key=lambda e: (e != "tesseract", seen.index(e)))


def build_report(records: list[dict], meta: dict, n_boot: int = 2000) -> tuple[dict, str]:
    """records: 1 dòng / (ảnh, engine) với các khoá item, level, engine, cer, cer_nodiac, wer, seconds, ..."""
    engines = _engine_order(records)
    by = {e: [r for r in records if r["engine"] == e] for e in engines}
    summary: dict = {"run_id": meta["run_id"], "n_images": len({r["item"] for r in records}), "engines": {}}

    lines = [
        f"# OCR vs VLM on synthetic scans (`{meta['run_id']}`)", "",
        f"{summary['n_images']} synthetic scans of Vietnamese lecture notes (levels: "
        + ", ".join(f"{lv} {sum(1 for r in by[engines[0]] if r['level'] == lv)}" for lv in LEVEL_ORDER)
        + "). Lower is better. Brackets: 95 % bootstrap CI.",
        f"Tesseract lang `{meta['ocr_lang']}`; VLM engines are labelled by model (task `vision`, same prompt). "
        f"Commit `{meta['git_commit']}`.", "",
        "| Engine | CER | CER without diacritics | WER | Errors | s / page (median) | Tokens / page (mean) |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in engines:
        rows = by[e]
        stats = {m: bootstrap_ci([r[m] for r in rows], n_boot=n_boot) for m in METRICS}
        secs = [r["seconds"] for r in rows]
        tokens = [r["tokens_in"] + r["tokens_out"] for r in rows]
        errors = sum(1 for r in rows if r["error"])
        summary["engines"][e] = {
            **{m: {"mean": s[0], "lo": s[1], "hi": s[2]} for m, s in stats.items()},
            "errors": errors, "n": len(rows),
            "seconds_median": statistics.median(secs) if secs else None,
            "tokens_mean": statistics.mean(tokens) if tokens else None,
            "by_level": {
                lv: {m: statistics.mean([r[m] for r in rows if r["level"] == lv]) for m in METRICS}
                for lv in LEVEL_ORDER if any(r["level"] == lv for r in rows)
            },
        }
        lines.append(
            f"| `{e}` | {_fmt_ci(*stats['cer'])} | {_fmt_ci(*stats['cer_nodiac'])} | {_fmt_ci(*stats['wer'])} "
            f"| {errors}/{len(rows)} | {summary['engines'][e]['seconds_median']:.1f} "
            f"| {summary['engines'][e]['tokens_mean']:,.0f} |"
        )

    lines += ["", "**By degradation level** (mean CER; few images per level, so no CI):", "",
              "| Engine | " + " | ".join(LEVEL_ORDER) + " |", "|---|" + "---|" * len(LEVEL_ORDER)]
    for e in engines:
        lv = summary["engines"][e]["by_level"]
        lines.append(f"| `{e}` | " + " | ".join(f"{lv[x]['cer']:.3f}" if x in lv else "–" for x in LEVEL_ORDER) + " |")

    if "tesseract" in by:
        summary["paired_vs_tesseract"] = {}
        base = {m: {r["item"]: r[m] for r in by["tesseract"] if not r["error"]} for m in METRICS}
        for e in engines[1:]:
            summary["paired_vs_tesseract"][e] = {}
            lines += ["", f"**Paired difference, `{e}` − `tesseract`** (negative = fewer errors than Tesseract):", ""]
            for m in METRICS:
                a = {r["item"]: r[m] for r in by[e] if not r["error"]}
                d, lo, hi, n = paired_bootstrap_diff(a, base[m], n_boot=n_boot)
                summary["paired_vs_tesseract"][e][m] = {"diff": d, "lo": lo, "hi": hi, "n": n}
                verdict = "no demonstrated difference" if lo <= 0 <= hi else ("fewer errors" if hi < 0 else "more errors")
                lines.append(f"- `{m}`: {d:+.3f} [{lo:+.3f}, {hi:+.3f}], n = {n} → {verdict}")

    lines += ["", "Caveats: synthetic degradation of digital pages, not real scans or phone photos; one page set "
              "from one course; ground truth is the PDF text layer. Per-image rows are in `raw.jsonl` "
              "(not published: the source notes may not be redistributed)."]
    return summary, "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", nargs="+", choices=ENGINES, default=["tesseract"])
    ap.add_argument("--from-raw", nargs="+", type=Path, metavar="RUN_DIR",
                    help="không chạy engine: gộp raw.jsonl của các run đã có rồi viết báo cáo mới")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data", type=Path, default=DATA_DIR)
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu (kiểm tra nhanh)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    manifest_path = args.data / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"Thiếu {manifest_path}. Chạy: python scripts/prepare_visual_eval.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest["items"][: args.limit or None]

    run_dir = RESULTS / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.from_raw:
        merge_runs(args.from_raw, run_dir, args.run_id, args.bootstrap)
        return
    meta = {
        "run_id": args.run_id, "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "git_commit": _git_commit(),
        "engines": args.engines, "dataset": manifest_path.as_posix(), "n_images": len(items),
        "degradation": {"seed": manifest.get("seed"), "dpi": manifest.get("dpi"), "levels": manifest.get("levels")},
        "ocr_lang": settings.ocr_lang,
        "vlm_model": f"{settings.cheap_model_provider}:{settings.cheap_model_name}",
        "vlm_prompt_chars": len(VLM_TRANSCRIBE_SYSTEM_PROMPT),
        "metrics": "NFC, whitespace collapsed, case and punctuation kept; cer_nodiac strips Vietnamese diacritics",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    records = []
    with (run_dir / "raw.jsonl").open("w", encoding="utf-8") as raw:
        for it in items:
            image = (args.data / it["image"]).read_bytes()
            gt = (args.data / it["ground_truth"]).read_text(encoding="utf-8")
            for engine in args.engines:
                out = transcribe(engine, image)
                rec = {"item": it["image"], "level": it["level"], "engine": engine_label(engine),
                       **score(gt, out["text"]), **out}
                records.append(rec)
                raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{it['image']:<52} {engine:<9} cer={rec['cer']:.3f} wer={rec['wer']:.3f} "
                      f"{rec['seconds']:.1f}s{' ERROR ' + rec['error'] if rec['error'] else ''}", flush=True)

    summary, md = build_report(records, meta, n_boot=args.bootstrap)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    print("\n" + md)


def merge_runs(sources: list[Path], run_dir: Path, run_id: str, n_boot: int) -> None:
    """Gộp nhiều run (vd 2 VLM chạy riêng) thành 1 báo cáo. Engine trùng tên: giữ run đầu tiên."""
    records, metas, seen = [], [], set()
    for src in sources:
        metas.append(json.loads((src / "meta.json").read_text(encoding="utf-8")))
        for line in (src / "raw.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if (rec["item"], rec["engine"]) not in seen:
                seen.add((rec["item"], rec["engine"]))
                records.append(rec)
    meta = {
        "run_id": run_id, "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "merged_from": [m["run_id"] for m in metas],
        "git_commit": metas[0]["git_commit"], "ocr_lang": metas[0]["ocr_lang"], "dataset": metas[0]["dataset"],
        "degradation": metas[0]["degradation"], "metrics": metas[0]["metrics"],
        "sources": metas,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    summary, md = build_report(records, meta, n_boot=n_boot)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
