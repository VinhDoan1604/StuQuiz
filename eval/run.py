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
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from app.config import settings
from eval.dataset import EvalItem, load_dataset
from eval.metrics import evidence_match, normalize_text, retrieval_metrics, split_alternatives
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
    # Kiểm tra TỪNG phương án của "A || B" để bắt lỗi gõ sai ở phương án phụ.
    orphans = [
        (it.id, alt[:70])
        for it in items
        for ev in it.gold_evidence
        for alt in split_alternatives(ev)
        if not any(evidence_match(t, alt, normalized=True) for t in norm)
    ]
    if orphans:
        msg = "gold_evidence không khớp chunk nào (hãy rút ngắn/sửa cho đúng nguyên văn):\n" + "\n".join(
            f"  - {q}: {e!r}" for q, e in orphans
        )
        if not allow_orphans:
            raise SystemExit(msg + "\n(Dùng --allow-orphans để bỏ qua — không khuyến khích.)")
        print("[cảnh báo] " + msg)
    print(f"[preflight] {len(items)} câu / {len(chunks)} chunks: OK")


def _take_limit(items: list[EvalItem], limit: int) -> list[EvalItem]:
    """Lấy `limit` câu nhưng giữ tỉ lệ câu không có đáp án (ít nhất 1 nếu dataset có),
    để chạy smoke vẫn kiểm tra được đường từ chối. Giữ thứ tự gốc của dataset."""
    if limit >= len(items):
        return items
    ans = [i for i in items if i.answerable]
    unans = [i for i in items if not i.answerable]
    n_un = min(len(unans), max(1, round(limit * len(unans) / len(items)))) if unans else 0
    n_un = min(n_un, limit)
    chosen = {i.id for i in ans[: limit - n_un] + unans[:n_un]}
    return [i for i in items if i.id in chosen]


def _keep_awake() -> None:
    """Windows: chặn máy ngủ trong lúc chạy (latency sẽ sai hoàn toàn nếu máy ngủ
    giữa 1 câu). Tự hết hiệu lực khi process kết thúc."""
    if sys.platform == "win32":
        try:
            import ctypes

            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        except Exception as exc:  # không chặn được thì vẫn chạy
            print(f"[cảnh báo] không chặn được chế độ ngủ: {exc}")


def _load_latest(raw_path: Path) -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[(r["config"], r["qid"])] = r  # bản ghi sau thắng bản ghi trước
    return latest


def _is_done(rec: dict | None, retrieval_only: bool) -> bool:
    """Pipeline đã chạy xong cho bản ghi này chưa. Bản ghi chỉ lỗi JUDGE mà có đủ context
    (chunks_full) coi như xong — được chấm lại bằng rejudge_pending, không sinh lại câu trả lời."""
    if rec is None or rec.get("error"):
        return False
    return retrieval_only or not rec.get("judge_error") or bool(rec.get("chunks_full"))


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


async def rejudge_pending(judge_llm, items_by_id: dict[str, EvalItem], latest: dict, raw_path: Path, configs: list[str]) -> int:
    """Chấm lại các bản ghi chỉ lỗi judge, dùng câu trả lời + context đã lưu. Trả về số còn lỗi."""
    pending = [
        r for (c, q), r in latest.items()
        if c in configs and q in items_by_id and not r.get("error") and r.get("judge_error") and r.get("chunks_full")
    ]
    for rec in pending:
        res = SimpleNamespace(answer=rec["answer"], chunks=rec["chunks"])
        await judge_record(judge_llm, items_by_id[rec["qid"]], res, rec)
        with raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status = "OK" if not rec.get("judge_error") else "vẫn lỗi"
        print(f"[chấm lại] {rec['config']} {rec['qid']}: {status}")
    return sum(1 for r in pending if r.get("judge_error"))


async def main_async(args) -> None:
    configs = args.configs or (list(RETRIEVAL_CONFIGS) if args.retrieval_only else list(ALL_CONFIGS))
    if args.retrieval_only and any(c in GRAPH_CONFIGS for c in configs):
        raise SystemExit("--retrieval-only chỉ hỗ trợ: " + ", ".join(RETRIEVAL_CONFIGS))
    if configs[0] != "dense":
        print("[lưu ý] Baseline được lấy là cấu hình đầu tiên; khuyến nghị đặt 'dense' đầu.")

    items = load_dataset(args.dataset, include_unreviewed=args.include_unreviewed)
    if args.qids:
        wanted = set(args.qids)
        missing = wanted - {i.id for i in items}
        if missing:
            raise SystemExit(f"Không có trong dataset: {', '.join(sorted(missing))}")
        items = [i for i in items if i.id in wanted]
    if args.limit:
        items = _take_limit(items, args.limit)
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
    _keep_awake()
    print("[warmup] nạp model embedding/reranker" + ("" if args.retrieval_only else " + LLM") + "…")
    await warmup(doc_ids, with_llm=not args.retrieval_only)
    items_by_id = {i.id: i for i in items}
    if judge_llm:
        await rejudge_pending(judge_llm, items_by_id, latest, raw_path, configs)

    for config in configs:
        for n, item in enumerate(items, start=1):
            if _is_done(latest.get((config, item.id)), args.retrieval_only):
                continue
            res = await run_pipeline(config, item.question, doc_ids, graphs, retrieval_only=args.retrieval_only)
            rec = {
                "config": config, "qid": item.id, "question": item.question, "answerable": item.answerable,
                "answer": res.answer,
                # Lưu ĐỦ context (không cắt) để có thể chấm lại judge mà không sinh lại câu trả lời.
                "chunks": [{"source_file": c.get("source_file"), "page": c.get("page"), "section": c.get("section"),
                            "text": c.get("text", "")} for c in res.chunks],
                "chunks_full": True,
                # Chấm trên chunk con (cùng đơn vị cho mọi cấu hình); context_retrieval = bằng chứng
                # có nằm trong context thật gửi LLM không (parent mở rộng sẽ cao hơn — không so trực tiếp).
                "retrieval": retrieval_metrics(res.child_chunks, item.gold_evidence) if item.answerable and not res.error else None,
                "context_retrieval": retrieval_metrics(res.chunks, item.gold_evidence) if item.answerable and not res.error else None,
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

    if judge_llm:
        left = sum(1 for (c, q), r in latest.items() if c in configs and q in items_by_id and r.get("judge_error"))
        if left:
            print(f"[chấm lại] {left} bản ghi lỗi judge — chờ 120s rồi chấm lại…")
            await asyncio.sleep(120)
            left = await rejudge_pending(judge_llm, items_by_id, latest, raw_path, configs)
        if left:
            print(f"[cảnh báo] còn {left} bản ghi lỗi judge — chạy lại đúng lệnh này (cùng --run-id) để chấm lại.")

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
    ap.add_argument("--qids", nargs="+", help="Chỉ chạy các câu này (vd: --qids q057 q061 q065)")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--include-unreviewed", action="store_true")
    ap.add_argument("--allow-orphans", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()