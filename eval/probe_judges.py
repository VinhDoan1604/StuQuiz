"""
eval/probe_judges.py — thử các model ứng viên làm judge TRƯỚC khi chạy full eval.

    python -m eval.probe_judges                                   # dùng eval/results/check-v5
    python -m eval.probe_judges --run eval/results/smoke-v3 --n 8
    python -m eval.probe_judges --candidates shopaikey_anthropic:claude-sonnet-4-6 shopaikey_openai:gpt-5-mini

Với mỗi ứng viên, chấm lại các câu trả lời ĐÃ CÓ trong run (không sinh lại) bằng đúng prompt
của eval/judge.py, mỗi lời gọi chỉ thử 1 lần (để đo độ ổn định thật của kênh), rồi in:
  - tỉ lệ gọi thành công + lỗi điển hình (403 quota, parse...)
  - độ trễ p50 mỗi lời gọi
  - verdict correctness + faithfulness từng câu, và độ đồng thuận correctness giữa các model.
Đầu tiên in danh sách model key được dùng (/v1/models) để kiểm tra tên model.

Tiêu chí chọn judge: 100% gọi thành công; KHÁC họ model sinh; verdict hợp lý khi đọc tay
những câu các model bất đồng.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from eval.dataset import load_dataset
from eval.judge import (
    _CORRECTNESS_SYSTEM,
    _FAITHFULNESS_SYSTEM,
    CorrectnessVerdict,
    FaithfulnessVerdict,
    _structured,
    faithfulness_score,
    make_judge_llm,
)
from eval.metrics import percentile

DEFAULT_CANDIDATES = [
    "shopaikey_anthropic:claude-sonnet-4-6",
    "shopaikey_openai:gpt-5-mini",
    "shopaikey_openai:gpt-5.4",
]
_KEYWORDS = ("gpt-5", "gpt-4.1", "o3", "o4", "claude", "gemini", "deepseek", "qwen")


def list_models() -> None:
    import httpx

    from app.llm.providers import get_provider

    spec = get_provider("shopaikey_openai")
    url = (spec.base_url() or "").rstrip("/") + "/models"
    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {spec.api_key()}"}, timeout=30)
        r.raise_for_status()
        ids = sorted(m["id"] for m in r.json().get("data", []))
    except Exception as exc:
        print(f"[models] không lấy được danh sách ({type(exc).__name__}: {exc})\n")
        return
    rel = [i for i in ids if any(k in i.lower() for k in _KEYWORDS)]
    print(f"[models] key được dùng {len(ids)} model; liên quan ({len(rel)}):")
    for i in range(0, len(rel), 4):
        print("   " + "  ".join(f"{m:<34}" for m in rel[i : i + 4]))
    print()


def load_records(run_dir: Path, n: int) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["config"], r["qid"])] = r
    recs = [r for r in latest.values() if not r.get("error") and r.get("answer")]
    return recs[:n]


async def probe(spec: str, recs: list[dict], items: dict) -> dict:
    provider, model = spec.split(":", 1)
    out = {"ok": 0, "total": 0, "lat": [], "errors": [], "verdicts": {}}
    try:
        llm = make_judge_llm(provider, model)
    except Exception as exc:
        out["errors"].append(f"khởi tạo: {exc}")
        return out
    from app.retrieval.helpers import format_chunks_for_prompt

    for r in recs:
        item = items[r["qid"]]
        key = f"{r['config']}:{r['qid']}"
        v: dict = {}
        calls = [("faith", FaithfulnessVerdict, _FAITHFULNESS_SYSTEM,
                  f"Câu hỏi: {item.question}\n\nCONTEXT:\n{format_chunks_for_prompt(r['chunks'])}\n\nCÂU TRẢ LỜI:\n{r['answer']}")]
        if item.answerable:
            calls.append(("corr", CorrectnessVerdict, _CORRECTNESS_SYSTEM,
                          f"Câu hỏi: {item.question}\n\nĐÁP ÁN CHUẨN:\n{item.gold_answer}\n\nCÂU TRẢ LỜI CẦN CHẤM:\n{r['answer']}"))
        for name, cls, system, user in calls:
            out["total"] += 1
            t0 = time.perf_counter()
            try:
                res = await _structured(llm, cls, system, user, retries=1)
                out["ok"] += 1
                out["lat"].append(time.perf_counter() - t0)
                if name == "faith":
                    v["abst"] = res.abstained
                    v["faith"] = faithfulness_score(res)
                else:
                    v["corr"] = res.verdict
            except Exception as exc:
                out["errors"].append(str(exc)[:140])
                v[name] = "LỖI"
        out["verdicts"][key] = v
        print(f"  {spec:<42} {key:<16} {v}")
    return out


async def main_async(args) -> None:
    list_models()
    items = {i.id: i for i in load_dataset(args.dataset, include_unreviewed=True)}
    recs = [r for r in load_records(Path(args.run), args.n) if r["qid"] in items]
    if not recs:
        raise SystemExit(f"Không có bản ghi dùng được trong {args.run}")
    if any(not r.get("chunks_full") for r in recs):
        print("[cảnh báo] run này lưu context bị cắt 600 ký tự (trước khi có chunks_full) -> điểm "
              "faithfulness ở đây thấp giả tạo; chỉ dùng tỉ lệ thành công + correctness để so sánh.\n")
    print(f"Chấm lại {len(recs)} câu trả lời từ {args.run} bằng {len(args.candidates)} ứng viên…\n")
    results = {c: await probe(c, recs, items) for c in args.candidates}

    print("\n=== Tổng hợp ===")
    print(f"{'model':<42} {'thành công':>11} {'p50 (s)':>8}  faith TB   lỗi điển hình")
    for c, o in results.items():
        faiths = [v["faith"] for v in o["verdicts"].values() if isinstance(v.get("faith"), float)]
        p50 = percentile(o["lat"], 50) if o["lat"] else float("nan")
        mf = f"{sum(faiths) / len(faiths):.3f}" if faiths else "  n/a"
        err = o["errors"][0][:70] if o["errors"] else "—"
        print(f"{c:<42} {o['ok']:>4}/{o['total']:<6} {p50:>8.1f}  {mf:>7}   {err}")

    names = list(results)
    print("\nĐồng thuận correctness (cùng verdict / số câu cả hai chấm được):")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs = [
                (results[a]["verdicts"][k].get("corr"), results[b]["verdicts"].get(k, {}).get("corr"))
                for k in results[a]["verdicts"]
            ]
            pairs = [(x, y) for x, y in pairs if x not in (None, "LỖI") and y not in (None, "LỖI")]
            agree = sum(x == y for x, y in pairs)
            diff = [k for k in results[a]["verdicts"]
                    if results[a]["verdicts"][k].get("corr") not in (None, "LỖI")
                    and results[b]["verdicts"].get(k, {}).get("corr") not in (None, "LỖI", results[a]["verdicts"][k].get("corr"))]
            print(f"  {a} vs {b}: {agree}/{len(pairs)}" + (f"  — bất đồng: {', '.join(diff)}" if diff else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="eval/results/check-v5")
    ap.add_argument("--dataset", default="eval/data/questions.jsonl")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
