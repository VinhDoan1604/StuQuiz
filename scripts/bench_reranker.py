"""
scripts/bench_reranker.py — đo độ trễ reranker theo từng biến thể, KHÔNG cần Qdrant/LLM.

Lấy cặp (câu hỏi, chunk) thật từ 1 lần eval đã lưu (raw.jsonl có đủ text chunk): với mỗi
câu hỏi, gom các chunk mà mọi cấu hình đã truy xuất, bù thêm chunk của câu khác cho đủ N
ứng viên (giống "top-N rồi rerank" ở production). Với mỗi biến thể đo:
  - latency/câu (p50, p95, trung bình) sau warm-up;
  - độ khớp top-5 so với biến thể tham chiếu (biến thể đầu tiên) — int8 / max_length nhỏ
    có làm đổi thứ hạng không. Đây chỉ là tín hiệu nhanh; chất lượng thật phải đo bằng
    `python -m eval.run --retrieval-only` (cần Qdrant).

    python -m scripts.bench_reranker                       # mặc định: 4 biến thể
    python -m scripts.bench_reranker --limit 10 --variants fp32-20 int8-10
    python -m scripts.bench_reranker --out eval/results/rerank-bench.json

Tên biến thể: "<fp32|int8>-<số ứng viên>[-<max_length>]", vd int8-10-384.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

DEFAULT_VARIANTS = ["fp32-20", "fp32-10", "int8-20", "int8-10"]


def load_queries(raw_path: Path, n_candidates: int) -> list[tuple[str, list[str]]]:
    by_q: dict[str, tuple[str, list[str]]] = {}
    all_texts: list[str] = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        q, texts = by_q.setdefault(r["qid"], (r["question"], []))
        for c in r.get("chunks") or []:
            t = (c.get("text") or "").strip()
            if t and t not in texts:
                texts.append(t)
            if t and t not in all_texts:
                all_texts.append(t)
    out = []
    for i, (q, texts) in enumerate(by_q.values()):
        cands = list(texts)
        j = i
        while len(cands) < n_candidates and len(all_texts) > len(cands):
            t = all_texts[j % len(all_texts)]
            if t not in cands:
                cands.append(t)
            j += 7
        out.append((q, cands))
    return out


def parse_variant(name: str) -> tuple[bool, int, int]:
    parts = name.split("-")
    if parts[0] not in ("fp32", "int8") or len(parts) not in (2, 3):
        raise SystemExit(f"Tên biến thể không hợp lệ: {name}")
    return parts[0] == "int8", int(parts[1]), int(parts[2]) if len(parts) == 3 else 512


def run_variant(name: str, queries: list[tuple[str, list[str]]], threads: int | None) -> dict:
    import torch

    from app.config import settings
    from app.retrieval import reranker as rr

    if threads:
        torch.set_num_threads(threads)
    int8, n_cand, max_len = parse_variant(name)
    settings.reranker_int8 = int8
    settings.reranker_max_length = max_len
    settings.reranker_device = "cpu"
    rr.reset_reranker()

    t0 = time.perf_counter()
    rr.warmup()
    load_s = time.perf_counter() - t0
    rr.rerank("khởi động", [{"text": "khởi động hệ thống"}] * 4, top_k=1)  # warm-up kernel

    lat, tops = [], []
    for q, cands in queries:
        cands = [{"text": t} for t in cands[:n_cand]]
        t0 = time.perf_counter()
        ranked = rr.rerank(q, cands, top_k=5)
        lat.append(time.perf_counter() - t0)
        tops.append([c["text"][:120] for c in ranked])
        print(f"  [{name}] {len(lat)}/{len(queries)} {lat[-1]:.2f}s")
    lat_sorted = sorted(lat)
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(round(0.95 * (len(lat_sorted) - 1))))]
    return {
        "variant": name,
        "int8": int8,
        "candidates": n_cand,
        "max_length": max_len,
        "load_s": round(load_s, 1),
        "n": len(lat),
        "mean_s": round(statistics.mean(lat), 2),
        "p50_s": round(statistics.median(lat), 2),
        "p95_s": round(p95, 2),
        "torch_threads": torch.get_num_threads(),
        "_tops": tops,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="eval/results/full-v1/raw.jsonl")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--limit", type=int, default=20, help="số câu hỏi đo (mặc định 20)")
    ap.add_argument("--threads", type=int, help="torch.set_num_threads (mặc định để torch tự chọn)")
    ap.add_argument("--out", help="ghi kết quả JSON")
    args = ap.parse_args()

    max_cand = max(parse_variant(v)[1] for v in args.variants)
    queries = load_queries(Path(args.raw), max_cand)[: args.limit]
    print(f"{len(queries)} câu hỏi, tối đa {max_cand} ứng viên/câu")

    results = [run_variant(v, queries, args.threads) for v in args.variants]
    ref = results[0]["_tops"]
    for r in results:
        overlap = [len(set(a) & set(b)) / max(1, len(set(b))) for a, b in zip(r["_tops"], ref)]
        top1 = [bool(a and b and a[0] == b[0]) for a, b in zip(r["_tops"], ref)]
        r["top5_overlap_vs_ref"] = round(statistics.mean(overlap), 3)
        r["top1_same_vs_ref"] = round(sum(top1) / len(top1), 3)
        del r["_tops"]

    print(f"\nTham chiếu: {results[0]['variant']}")
    print("| Biến thể | ứng viên | max_len | p50 (s) | p95 (s) | TB (s) | top-5 trùng | top-1 trùng |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['variant']} | {r['candidates']} | {r['max_length']} | {r['p50_s']} | {r['p95_s']} | "
            f"{r['mean_s']} | {r['top5_overlap_vs_ref']} | {r['top1_same_vs_ref']} |"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
