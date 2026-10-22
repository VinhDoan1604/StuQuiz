"""
eval/run.py — chạy ablation + chấm điểm + sinh báo cáo.

    # 0) Nạp corpus & duyệt dataset (xem eval/README.md)
    # 1) Chỉ retrieval (không cần LLM, nhanh, rẻ):
    python -m eval.run --retrieval-only --configs dense hybrid hybrid_rerank
    # 2) Đầy đủ:
    EVAL_JUDGE_MODEL=<model khác model sinh> python -m eval.run
    # 3) Tiếp tục lần chạy bị ngắt (chạy lại các câu lỗi):
    python -m eval.run --run-id 20260101-120000

Chạy TUẦN TỰ có chủ đích (bộ đếm token/latency là global).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from app.config import settings
from eval.dataset import EvalItem, load_dataset
from eval.metrics import evidence_match, normalize_text, retrieval_metrics
from eval.pipelines import ALL_CONFIGS, FINAL_TOP_K, GRAPH_CONFIGS, RETRIEVAL_CONFIGS, build_graphs, run_pipeline, warmup
from eval.report import build_report


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


async def preflight_evidence(items: list[EvalItem], doc_ids: list[str], allow_orphans: bool) -> None:
    """Mỗi gold_evidence PHẢI khớp >=1 chunk trong index; nếu không, câu đó không thể 'hit'
    dù hệ thống hoàn hảo (gõ sai / bằng chứng vắt qua 2 chunk) và sẽ làm thấp điểm oan."""
    from app.retrieval.helpers import aget_all_chunks_for_documents

    chunks = await aget_all_chunks_for_documents(doc_ids)
    if not chunks:
        raise SystemExit("Index rỗng cho các doc_id trong manifest — chạy `python -m eval.ingest_corpus` trước.")
    norm = [normalize_text(c["text"]) for c in chunks]
    orphans = [
        (it.id, ev[:70])
        for it in items
        for ev in it.gold_evidence
        if not any(evidence_match(t, ev, normalized=True) for t in norm)
    ]
    if orphans:
        msg = "gold_evidence không khớp chunk nào (hãy rút ngắn/sửa cho đúng nguyên văn):\n" + "\n".join(
            f"  - {q}: {e!r}" for q, e in orphans
        )
        if not allow_orphans:
            raise SystemExit(msg + "\n(Dùng --allow-orphans để bỏ qua — không khuyến khích.)")
        print("[cảnh báo] " + msg)
    print(f"[preflight] {len(items)} câu / {len(chunks)} chunks: OK")


def _load_latest(raw_path: Path) -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[(r["config"], r["qid"])] = r  # bản ghi sau thắng bản ghi trước
    return latest


def _is_done(rec: dict | None, retrieval_only: bool) -> bool:
    if rec is None or rec.get("error"):
        return False
    return retrieval_only or not rec.get("judge_error")


async def judge_record(llm, item: EvalItem, res, rec: dict) -> None:
    from eval.judge import CORRECTNESS_SCORE, JudgeError, faithfulness_score, judge_correctness, judge_faithfulness

    try:
        fv = await judge_faithfulness(llm, item.question, res.chunks, res.answer)
        rec["abstained"] = fv.abstained
        rec["faithfulness"] = faithfulness_score(fv)
        rec["claims"] = [c.model_dump() for c in fv.claims]
        if item.answerable:
            cv = await judge_correctness(llm, item.question, item.gold_answer or "", res.answer)
            rec["correctness"] = CORRECTNESS_SCORE[cv.verdict]
            rec["correctness_verdict"] = cv.verdict
            rec["judge_reason"] = cv.reason
        rec["judge_error"] = None
    except JudgeError as exc:
        rec["judge_error"] = str(exc)


async def main_async(args) -> None:
    configs = args.configs or (list(RETRIEVAL_CONFIGS) if args.retrieval_only else list(ALL_CONFIGS))
    if args.retrieval_only and any(c in GRAPH_CONFIGS for c in configs):
        raise SystemExit("--retrieval-only chỉ hỗ trợ: " + ", ".join(RETRIEVAL_CONFIGS))
    if configs[0] != "dense":
        print("[lưu ý] Baseline được lấy là cấu hình đầu tiên; khuyến nghị đặt 'dense' đầu.")

    items = load_dataset(args.dataset, include_unreviewed=args.include_unreviewed)
    if args.limit:
        items = items[: args.limit]
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    doc_ids = [d["doc_id"] for d in manifest["documents"]]
    await preflight_evidence(items, doc_ids, args.allow_orphans)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.jsonl"

    judge_llm, judge_meta = (None, None)
    if not args.retrieval_only:
        from eval.judge import build_judge_llm

        judge_llm, judge_meta = build_judge_llm()

    meta = {
        "run_id": run_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "git_commit": _git_commit(),
        "retrieval_only": args.retrieval_only,
        "configs": configs,
        "final_top_k": FINAL_TOP_K,
        "embedding_model": manifest.get("embedding_model"),
        "chunking": manifest.get("chunking"),
        "corpus": manifest["documents"],
        "generator": {"provider": settings.cheap_model_provider, "model": settings.cheap_model_name},
        "judge": judge_meta,
    }
    meta_path = run_dir / "meta.json"
    if meta_path.exists():  # resume: chặn trộn kết quả từ model khác nhau
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if (old.get("generator"), old.get("judge")) != (meta["generator"], meta["judge"]):
            raise SystemExit("Resume với model sinh/judge khác lần chạy trước — tạo --run-id mới.")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    latest = _load_latest(raw_path)
    graphs = build_graphs(configs)
    print("[warmup] nạp model embedding/reranker…")
    await warmup(doc_ids)

    for config in configs:
        for n, item in enumerate(items, start=1):
            if _is_done(latest.get((config, item.id)), args.retrieval_only):
                continue
            res = await run_pipeline(config, item.question, doc_ids, graphs, retrieval_only=args.retrieval_only)
            rec = {
                "config": config, "qid": item.id, "question": item.question, "answerable": item.answerable,
                "answer": res.answer,
                "chunks": [{"source_file": c.get("source_file"), "page": c.get("page"), "text": c.get("text", "")[:600]} for c in res.chunks],
                "retrieval": retrieval_metrics(res.chunks, item.gold_evidence) if item.answerable and not res.error else None,
                "latency_s": res.latency_s, "llm_calls": res.llm_calls, "tokens": res.tokens,
                "loops": res.loops, "rewritten_queries": res.rewritten_queries, "error": res.error,
                "faithfulness": None, "abstained": None, "correctness": None, "judge_error": None,
            }
            if judge_llm and not res.error:
                await judge_record(judge_llm, item, res, rec)

            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            latest[(config, item.id)] = rec
            flag = f"LỖI: {res.error[:60]}" if res.error else f"{res.latency_s:.1f}s"
            print(f"[{config}] {n}/{len(items)} {item.id} {flag}")

    records: dict[str, dict[str, dict]] = {c: {} for c in configs}
    for (c, q), r in latest.items():
        if c in records:
            records[c][q] = r

    summary, md = build_report(records, items, meta, n_boot=args.bootstrap)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== Xong. Báo cáo: {run_dir / 'report.md'} ===\n")
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval/data/questions.jsonl")
    ap.add_argument("--manifest", default="eval/data/corpus_manifest.json")
    ap.add_argument("--configs", nargs="+", choices=ALL_CONFIGS)
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--run-id")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--include-unreviewed", action="store_true")
    ap.add_argument("--allow-orphans", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()