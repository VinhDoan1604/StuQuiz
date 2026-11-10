"""
eval/orchestrator_eval.py

Đo bước Orchestrator, phần mà ablation (eval/pipelines.py) cố ý bỏ qua:
1. Độ chính xác phân loại ý định (qa/teaching/quiz/summary).
2. Viết lại câu nối tiếp: bản viết lại có nêu đúng chủ đề không, và câu đã tự đủ nghĩa
   có được giữ nguyên không. So với cách cũ (regex đại từ) để biết cách cũ bỏ sót bao nhiêu.

Chỉ cần LLM, không cần Qdrant:
    python -m eval.orchestrator_eval --run-id orchestrator-v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
from collections import Counter
from pathlib import Path

from app.agents.orchestrator_agent import run_orchestrator
from app.retrieval.query_processing import _needs_rewrite

CASES = Path(__file__).parent / "data" / "orchestrator_cases.json"
RESULTS = Path(__file__).parent / "results"
CONCURRENCY = 4


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson 95% cho tỉ lệ k/n."""
    if n == 0:
        return 0.0, 0.0
    p, denom = k / n, 1 + z * z / n
    centre, margin = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def _history_text(history: list[list[str]]) -> str:
    lines = [line for user, answer in history for line in (f"Người dùng: {user}", f"Trợ lý: {answer}")]
    return "[Lịch sử gần đây]\n" + "\n".join(lines)


def _normalize(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


async def _classify(sem: asyncio.Semaphore, case: dict) -> dict:
    async with sem:
        out = await run_orchestrator({"user_query": case["message"], "chat_history_summary": ""})
    return {**case, "predicted": out["intent"]}


async def _rewrite(sem: asyncio.Semaphore, case: dict) -> dict:
    history = _history_text(case["history"])
    async with sem:
        out = await run_orchestrator({"user_query": case["message"], "chat_history_summary": history})
    rewritten = out.get("standalone_query", "")
    if case.get("standalone"):
        passed = _normalize(rewritten) == _normalize(case["message"])
    else:
        passed = all(term in rewritten.lower() for term in case["must_contain"])
    return {**case, "rewritten": rewritten, "passed": passed, "old_regex_fires": _needs_rewrite(case["message"], history)}


async def main_async(run_id: str) -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    sem = asyncio.Semaphore(CONCURRENCY)
    intents = await asyncio.gather(*(_classify(sem, c) for c in cases["intents"]))
    rewrites = await asyncio.gather(*(_rewrite(sem, c) for c in cases["followups"]))

    correct = sum(r["predicted"] == r["intent"] for r in intents)
    confusion = Counter((r["intent"], r["predicted"]) for r in intents if r["predicted"] != r["intent"])
    followups = [r for r in rewrites if not r.get("standalone")]
    standalone = [r for r in rewrites if r.get("standalone")]
    fu_pass, st_pass = sum(r["passed"] for r in followups), sum(r["passed"] for r in standalone)
    regex_fires = sum(r["old_regex_fires"] for r in followups)

    summary = {
        "intent_accuracy": correct / len(intents), "intent_correct": correct, "intent_n": len(intents),
        "intent_ci95": _wilson(correct, len(intents)),
        "intent_errors": [{"true": t, "predicted": p, "count": c} for (t, p), c in confusion.items()],
        "followup_pass": fu_pass, "followup_n": len(followups), "followup_ci95": _wilson(fu_pass, len(followups)),
        "standalone_kept": st_pass, "standalone_n": len(standalone),
        "old_regex_would_rewrite": regex_fires,
    }
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "raw.json").write_text(json.dumps({"intents": intents, "rewrites": rewrites}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for r in rewrites:
        if not r["passed"]:
            print(f"FAIL {r['id']}: {r['message']!r} -> {r['rewritten']!r}")
    for r in intents:
        if r["predicted"] != r["intent"]:
            print(f"INTENT {r['intent']} -> {r['predicted']}: {r['message']!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="orchestrator-v1")
    asyncio.run(main_async(parser.parse_args().run_id))
