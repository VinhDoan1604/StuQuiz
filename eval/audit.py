"""
eval/audit.py — kiểm tra judge có đáng tin không (bước nhà tuyển dụng rất thích hỏi).

    python -m eval.audit eval/results/<run_id> --n 20          # xuất audit.csv
    # Mở audit.csv, điền cột human_correctness: correct | partial | incorrect
    python -m eval.audit eval/results/<run_id> --score          # tính độ đồng thuận

    # Audit faithfulness theo từng khẳng định (điền human_supported: true | false):
    python -m eval.audit eval/results/<run_id> --kind faithfulness --config production --n 30
    python -m eval.audit eval/results/<run_id> --kind faithfulness --score

Báo cáo kết quả kiểu: "judge đồng thuận 85% (17/20) với người chấm" — kèm hạn chế.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

FIELDS = ["config", "qid", "question", "gold_answer", "answer", "judge_correctness", "human_correctness"]
FAITH_FIELDS = ["config", "qid", "question", "claim", "context", "judge_supported", "human_supported"]
LABEL = {1.0: "correct", 0.5: "partial", 0.0: "incorrect"}


def _read_audit(path: Path) -> list[dict[str, str]]:
    """Đọc audit.csv kể cả sau khi Excel lưu lại: có/không BOM, phân cách ',' hoặc ';'."""
    text = path.read_text(encoding="utf-8-sig")
    header = text.split("\n", 1)[0]
    delimiter = ";" if header.count(";") > header.count(",") else ","
    return list(csv.DictReader(text.splitlines(keepends=True), delimiter=delimiter))


def _latest(run_dir: Path) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["config"], r["qid"])] = r
    return list(latest.values())


def _guard_labels(path: Path, column: str) -> None:
    if path.exists() and any((r.get(column) or "").strip() for r in _read_audit(path)):
        raise SystemExit(f"{path} đã có nhãn người chấm — không ghi đè. Xoá/đổi tên file nếu muốn xuất lại.")


def score_faithfulness(rows: list[dict[str, str]]) -> dict[str, int]:
    """Đồng thuận trên các khẳng định đã gán nhãn, tách theo hướng lệch để biết judge dễ hay khó."""
    judged = [r for r in rows if (r.get("human_supported") or "").strip()]
    pairs = [(r["judge_supported"].strip().lower() == "true", r["human_supported"].strip().lower() == "true") for r in judged]
    return {
        "n": len(pairs),
        "agree": sum(j == h for j, h in pairs),
        "judge_lenient": sum(j and not h for j, h in pairs),
        "judge_strict": sum(h and not j for j, h in pairs),
    }


def export_faithfulness(run_dir: Path, path: Path, n: int, seed: int, config: str | None) -> None:
    _guard_labels(path, "human_supported")
    rows = [
        {"config": r["config"], "qid": r["qid"], "question": r["question"], "claim": c["claim"],
         "context": "\n\n".join(f"[{i}] {ch.get('text', '')}" for i, ch in enumerate(r["chunks"], start=1)),
         "judge_supported": str(bool(c["supported"])).lower(), "human_supported": ""}
        for r in _latest(run_dir) if (config is None or r["config"] == config)
        for c in (r.get("claims") or [])
    ]
    random.Random(seed).shuffle(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FAITH_FIELDS)
        w.writeheader()
        w.writerows(rows[:n])
    print(f"Đã xuất {min(n, len(rows))}/{len(rows)} khẳng định -> {path}. Điền human_supported rồi chạy --score.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--kind", choices=["correctness", "faithfulness"], default="correctness")
    ap.add_argument("--config", help="Chỉ lấy mẫu từ cấu hình này")
    ap.add_argument("--dataset", default="eval/data/questions_v2.jsonl", help="Nguồn gold_answer (v2 chứa cả câu v1)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if args.kind == "faithfulness":
        path = run_dir / "audit_faithfulness.csv"
        if args.score:
            s = score_faithfulness(_read_audit(path))
            if not s["n"]:
                raise SystemExit("Chưa có dòng nào được điền human_supported.")
            print(f"Đồng thuận judge–người: {s['agree']}/{s['n']} = {s['agree'] / s['n']:.0%} "
                  f"(judge dễ hơn: {s['judge_lenient']}, khó hơn: {s['judge_strict']})")
            return
        export_faithfulness(run_dir, path, args.n, args.seed, args.config)
        return

    csv_path = run_dir / "audit.csv"

    if args.score:
        rows = [r for r in _read_audit(csv_path) if (r.get("human_correctness") or "").strip()]
        if not rows:
            raise SystemExit("Chưa có dòng nào được điền human_correctness.")
        agree = sum(r["judge_correctness"] == r["human_correctness"].strip().lower() for r in rows)
        print(f"Đồng thuận judge–người: {agree}/{len(rows)} = {agree / len(rows):.0%}")
        return

    gold = {}
    ds = Path(args.dataset)
    if ds.exists():
        for line in ds.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                o = json.loads(line)
                gold[o["id"]] = o.get("gold_answer", "")

    pool = [r for r in _latest(run_dir)
            if r.get("correctness") is not None and (args.config is None or r["config"] == args.config)]
    random.Random(args.seed).shuffle(pool)

    _guard_labels(csv_path, "human_correctness")

    # utf-8-sig (BOM) để Excel trên Windows hiển thị đúng tiếng Việt.
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
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
