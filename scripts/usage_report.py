"""
scripts/usage_report.py

Tổng hợp số liệu vận hành từ các lượt chat đã lưu: độ trễ, số lời gọi LLM, token theo
intent, thời gian từng node, tỉ lệ fallback và đánh giá 👍/👎.

    python scripts/usage_report.py --days 7
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict
from datetime import timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import SessionLocal  # noqa: E402
from app.models.db_models import ChatTurn  # noqa: E402
from app.utils.time import utcnow  # noqa: E402


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * q), len(ordered) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    days = parser.parse_args().days

    with SessionLocal() as db:
        turns = db.query(ChatTurn).filter(ChatTurn.role == "assistant", ChatTurn.created_at >= utcnow() - timedelta(days=days)).all()

    rows = [(t, (t.reflection_trace or {}).get("metrics")) for t in turns]
    rows = [(t, m) for t, m in rows if m]
    if not rows:
        print("Chưa có lượt chat nào có số liệu.")
        return

    by_intent: dict[str, list] = defaultdict(list)
    nodes: dict[str, list[float]] = defaultdict(list)
    for t, m in rows:
        by_intent[t.intent or "?"].append(m)
        for node, ms in m["nodes_ms"].items():
            nodes[node].append(ms / 1000)

    print(f"{len(rows)} lượt trong {days} ngày")
    print(f"{'intent':10} {'n':>4} {'p50 s':>7} {'p95 s':>7} {'LLM':>5} {'token':>7}")
    for intent, ms in sorted(by_intent.items()):
        totals = [x["total_ms"] / 1000 for x in ms]
        print(
            f"{intent:10} {len(ms):>4} {_percentile(totals, .5):>7.1f} {_percentile(totals, .95):>7.1f} "
            f"{statistics.mean(x['llm_calls'] for x in ms):>5.1f} {statistics.mean(x['tokens_in'] + x['tokens_out'] for x in ms):>7.0f}"
        )
    print("\nThời gian theo node (p50 giây):")
    for node, values in sorted(nodes.items(), key=lambda kv: -statistics.median(kv[1])):
        print(f"  {node:18} {statistics.median(values):.2f}")

    rated = [t.feedback for t, _ in rows if t.feedback]
    print(f"\nFallback: {sum(1 for _, m in rows if m['fallbacks'])}/{len(rows)} lượt")
    print(f"Đánh giá: {sum(1 for r in rated if r > 0)} 👍, {sum(1 for r in rated if r < 0)} 👎")


if __name__ == "__main__":
    main()
