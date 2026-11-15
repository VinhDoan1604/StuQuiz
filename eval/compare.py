"""
eval/compare.py — so sánh ghép cặp giữa các cấu hình của MỘT lần chạy, theo từng nhóm câu hỏi.

report.py chỉ so với baseline (cấu hình đầu tiên). File này trả lời các câu hỏi như
"crag có hơn production không, riêng trên câu viết kiểu sinh viên?" mà không gọi LLM.

    python -m eval.compare eval/results/<run_id> --dataset eval/data/questions_v2.jsonl \
        --base production --vs crag full --out eval/results/<run_id>/compare.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.dataset import EvalItem, load_dataset
from eval.metrics import bootstrap_ci, paired_bootstrap_diff
from eval.report import collect

METRICS = ("mrr", "hit@5", "faithfulness", "correctness", "false_abstain", "abstain_correct", "tokens", "llm_calls", "latency_s")
GROUPS = {
    "all": lambda it: True,
    "v1": lambda it: not ({"student-phrasing", "hard-unanswerable"} & set(it.tags)),
    "student-phrasing": lambda it: "student-phrasing" in it.tags,
    "hard-unanswerable": lambda it: "hard-unanswerable" in it.tags,
}


def load_latest(run_dir: Path) -> dict[str, dict[str, dict]]:
    records: dict[str, dict[str, dict]] = {}
    for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            records.setdefault(rec["config"], {})[rec["qid"]] = rec
    return records


def in_group(values: dict[str, float], wanted: set[str]) -> dict[str, float]:
    return {q: v for q, v in values.items() if q in wanted}


def build(records: dict[str, dict[str, dict]], items: list[EvalItem], base: str, others: list[str], n_boot: int) -> str:
    by_id = {i.id: i for i in items}
    values = collect(records, by_id)
    lines: list[str] = []
    for group, pred in GROUPS.items():
        wanted = {i.id for i in items if pred(i)}
        if not wanted:
            continue
        lines.append(f"## {group} ({len(wanted)} questions)\n")
        lines.append("| Metric | " + " | ".join(f"`{c}`" for c in [base, *others]) + " |")
        lines.append("|---|" + "---|" * (len(others) + 1))
        for metric in METRICS:
            cells = []
            for config in [base, *others]:
                vals = list(in_group(values[config].get(metric, {}), wanted).values())
                if not vals:
                    cells.append("n/a")
                    continue
                mean, lo, hi = bootstrap_ci(vals, n_boot=n_boot)
                cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}] (n={len(vals)})")
            if any(c != "n/a" for c in cells):
                lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"| Paired vs `{base}` | Metric | Δ | 95% CI | n | CI excludes 0 |")
        lines.append("|---|---|---|---|---|---|")
        for config in others:
            for metric in METRICS:
                a = in_group(values[config].get(metric, {}), wanted)
                b = in_group(values[base].get(metric, {}), wanted)
                diff, lo, hi, n = paired_bootstrap_diff(a, b, n_boot=n_boot)
                if n:
                    flag = "yes" if lo > 0 or hi < 0 else "no"
                    lines.append(f"| `{config}` | {metric} | {diff:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {n} | {flag} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--dataset", default="eval/data/questions_v2.jsonl")
    ap.add_argument("--base", required=True)
    ap.add_argument("--vs", nargs="+", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    records = load_latest(run_dir)
    missing = {args.base, *args.vs} - set(records)
    if missing:
        raise SystemExit(f"Không có cấu hình trong raw.jsonl: {', '.join(sorted(missing))}")
    wanted = {q for c in (args.base, *args.vs) for q in records[c]}
    items = [i for i in load_dataset(args.dataset) if i.id in wanted]
    md = build(records, items, args.base, args.vs, args.bootstrap)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
