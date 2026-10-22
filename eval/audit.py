"""
eval/audit.py — kiểm tra judge có đáng tin không (bước nhà tuyển dụng rất thích hỏi).

    python -m eval.audit eval/results/<run_id> --n 20          # xuất audit.csv
    # Mở audit.csv, điền cột human_correctness: correct | partial | incorrect
    python -m eval.audit eval/results/<run_id> --score          # tính độ đồng thuận

Báo cáo kết quả kiểu: "judge đồng thuận 85% (17/20) với người chấm" — kèm hạn chế.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

FIELDS = ["config", "qid", "question", "gold_answer", "answer", "judge_correctness", "human_correctness"]
LABEL = {1.0: "correct", 0.5: "partial", 0.0: "incorrect"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    csv_path = run_dir / "audit.csv"

    if args.score:
        rows = [r for r in csv.DictReader(csv_path.open(encoding="utf-8")) if r["human_correctness"].strip()]
        if not rows:
            raise SystemExit("Chưa có dòng nào được điền human_correctness.")
        agree = sum(r["judge_correctness"] == r["human_correctness"].strip().lower() for r in rows)
        print(f"Đồng thuận judge–người: {agree}/{len(rows)} = {agree / len(rows):.0%}")
        return

    gold = {}
    ds = Path("eval/data/questions.jsonl")
    if ds.exists():
        for line in ds.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                gold[o["id"]] = o.get("gold_answer", "")

    latest = {}
    for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["config"], r["qid"])] = r
    pool = [r for r in latest.values() if r.get("correctness") is not None]
    random.Random(args.seed).shuffle(pool)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in pool[: args.n]:
            w.writerow({
                "config": r["config"], "qid": r["qid"], "question": r["question"],
                "gold_answer": gold.get(r["qid"], ""), "answer": r["answer"],
                "judge_correctness": LABEL[r["correctness"]], "human_correctness": "",
            })
    print(f"Đã xuất {min(args.n, len(pool))} dòng -> {csv_path}. Điền cột human_correctness rồi chạy --score.")


if __name__ == "__main__":
    main()