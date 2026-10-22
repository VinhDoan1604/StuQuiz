"""
eval/report.py — tổng hợp raw.jsonl thành báo cáo Markdown + summary.json.

Nguyên tắc trung thực:
- Mọi số đều kèm khoảng tin cậy bootstrap 95% và cỡ mẫu n.
- So sánh với baseline bằng bootstrap GHÉP CẶP; CI chứa 0 => ghi rõ "chưa đủ bằng chứng".
- Câu pipeline lỗi / judge lỗi KHÔNG bị thay bằng hằng số; được đếm và hiển thị.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

from eval.dataset import EvalItem
from eval.metrics import bootstrap_ci, paired_bootstrap_diff, percentile

RETRIEVAL_METRICS = ["hit@1", "hit@3", "hit@5", "recall@5", "mrr"]
# (khoá, nhãn, "cao hơn là tốt hơn?")
QUALITY_METRICS = [
    ("faithfulness", "Faithfulness ↑", True),
    ("correctness", "Correctness ↑", True),
    ("false_abstain", "Từ chối nhầm ↓", False),
    ("abstain_correct", "Từ chối đúng (câu không có đáp án) ↑", True),
]


def _fmt(vals: Iterable[Optional[float]], n_boot: int, digits: int = 3) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    m, lo, hi = bootstrap_ci(vals, n_boot=n_boot)
    return f"{m:.{digits}f} [{lo:.{digits}f}–{hi:.{digits}f}] (n={len(vals)})"


def collect(records: dict[str, dict[str, dict]], items: dict[str, EvalItem]) -> dict[str, dict[str, dict[str, float]]]:
    """-> values[config][metric][qid]"""
    values: dict[str, dict[str, dict[str, float]]] = {}
    for config, recs in records.items():
        v: dict[str, dict[str, float]] = {}

        def put(metric: str, qid: str, val: Optional[float]) -> None:
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                v.setdefault(metric, {})[qid] = float(val)

        for qid, rec in recs.items():
            item = items.get(qid)
            if item is None or rec.get("error"):
                continue
            if item.answerable:
                for m in RETRIEVAL_METRICS:
                    put(m, qid, (rec.get("retrieval") or {}).get(m))
                put("faithfulness", qid, rec.get("faithfulness"))
                put("correctness", qid, rec.get("correctness"))
                if rec.get("abstained") is not None:
                    put("false_abstain", qid, 1.0 if rec["abstained"] else 0.0)
            elif rec.get("abstained") is not None:
                put("abstain_correct", qid, 1.0 if rec["abstained"] else 0.0)

            put("latency_s", qid, rec.get("latency_s"))
            put("llm_calls", qid, rec.get("llm_calls"))
            if rec.get("tokens"):
                put("tokens", qid, rec.get("tokens"))
            put("loops", qid, rec.get("loops"))
        values[config] = v
    return values


def build_report(
    records: dict[str, dict[str, dict]], items: list[EvalItem], meta: dict[str, Any], n_boot: int = 2000
) -> tuple[dict, str]:
    items_by_id = {i.id: i for i in items}
    configs = list(records)
    baseline = configs[0]
    values = collect(records, items_by_id)
    n_ans = sum(i.answerable for i in items)
    n_unans = len(items) - n_ans
    retrieval_only = meta.get("retrieval_only", False)

    L: list[str] = []
    L.append(f"# Báo cáo đánh giá StudyRAG — run `{meta['run_id']}`\n")
    L.append(f"- Ngày: {meta['date']} · commit: `{meta.get('git_commit', 'n/a')}`")
    L.append(f"- Dataset: **{len(items)} câu** ({n_ans} có đáp án, {n_unans} không có đáp án)")
    L.append(f"- Corpus: {len(meta['corpus'])} tài liệu · embedding `{meta['embedding_model']}` · top-k={meta['final_top_k']}")
    if not retrieval_only:
        g = meta["generator"]
        j = meta["judge"]
        L.append(f"- Model sinh/chấm trong pipeline: `{g['provider']}/{g['model']}` · Judge: `{j['provider']}/{j['model']}`")
        if j["same_as_generator"]:
            L.append("- ⚠️ **Judge trùng model sinh** → điểm faithfulness/correctness có thể bị thổi phồng.")
    L.append(f"- Khoảng tin cậy: percentile bootstrap 95% ({n_boot} lần); baseline = `{baseline}`\n")

    # --- Retrieval
    L.append("## 1. Retrieval (chỉ câu có đáp án)\n")
    L.append("| Cấu hình | " + " | ".join(RETRIEVAL_METRICS) + " |")
    L.append("|---|" + "---|" * len(RETRIEVAL_METRICS))
    for c in configs:
        L.append(f"| `{c}` | " + " | ".join(_fmt(values[c].get(m, {}).values(), n_boot) for m in RETRIEVAL_METRICS) + " |")

    # --- Generation
    summary_quality: dict[str, Any] = {}
    if not retrieval_only:
        L.append("\n## 2. Chất lượng câu trả lời\n")
        L.append("| Cấu hình | " + " | ".join(lbl for _, lbl, _ in QUALITY_METRICS) + " |")
        L.append("|---|" + "---|" * len(QUALITY_METRICS))
        for c in configs:
            L.append(f"| `{c}` | " + " | ".join(_fmt(values[c].get(k, {}).values(), n_boot) for k, _, _ in QUALITY_METRICS) + " |")
        L.append(
            "\n*Faithfulness chỉ tính trên câu trả lời có ≥1 khẳng định (câu từ chối không tính; "
            "chúng được đo ở cột 'Từ chối').*"
        )

    # --- Efficiency
    L.append("\n## 3. Chi phí & độ trễ\n")
    L.append("| Cấu hình | Latency p50 (s) | p95 (s) | LLM calls TB | Token TB | Vòng lặp TB | Tỉ lệ lỗi |")
    L.append("|---|---|---|---|---|---|---|")
    eff: dict[str, Any] = {}
    for c in configs:
        lat = list(values[c].get("latency_s", {}).values())
        calls = list(values[c].get("llm_calls", {}).values())
        toks = list(values[c].get("tokens", {}).values())
        loops = list(values[c].get("loops", {}).values())
        errs = sum(1 for r in records[c].values() if r.get("error"))
        tot = len(records[c])
        mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
        eff[c] = {"p50": percentile(lat, 50), "p95": percentile(lat, 95), "calls": mean(calls),
                  "tokens": mean(toks), "loops": mean(loops), "error_rate": errs / tot if tot else float("nan")}
        e = eff[c]
        tok_cell = f"{e['tokens']:.0f}" if toks else "n/a"
        L.append(f"| `{c}` | {e['p50']:.2f} | {e['p95']:.2f} | {e['calls']:.1f} | {tok_cell} | {e['loops']:.2f} | {e['error_rate']:.1%} ({errs}/{tot}) |")
    L.append("\n*Latency đo tuần tự trên 1 máy, sau warm-up, không cache embedding giữa các lần chạy.*")

    # --- Paired deltas
    L.append(f"\n## 4. Chênh lệch so với baseline `{baseline}` (bootstrap ghép cặp)\n")
    delta_metrics = ["hit@5", "mrr"] + ([] if retrieval_only else ["faithfulness", "correctness", "abstain_correct"])
    L.append("| Cấu hình | Chỉ số | Δ | 95% CI | n cặp | Kết luận |")
    L.append("|---|---|---|---|---|---|")
    deltas: dict[str, Any] = {}
    for c in configs[1:]:
        for m in delta_metrics:
            d, lo, hi, n = paired_bootstrap_diff(values[c].get(m, {}), values[baseline].get(m, {}), n_boot=n_boot)
            if n == 0:
                continue
            verdict = "khác biệt có ý nghĩa" if (lo > 0 or hi < 0) else "chưa đủ bằng chứng"
            deltas[f"{c}:{m}"] = {"diff": d, "lo": lo, "hi": hi, "n": n, "significant": lo > 0 or hi < 0}
            L.append(f"| `{c}` | {m} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {n} | {verdict} |")

    # --- Failure analysis
    last = configs[-1]
    misses = [
        q for q, val in values[last].get("hit@5", {}).items() if val == 0.0
    ]
    L.append(f"\n## 5. Phân tích lỗi (`{last}`)\n")
    L.append(f"- Retrieval trượt hoàn toàn (hit@5=0): **{len(misses)}** câu: {', '.join(sorted(misses)[:15]) or '—'}")
    if not retrieval_only:
        bad = [q for q, val in values[last].get("correctness", {}).items() if val == 0.0]
        L.append(f"- Trả lời sai hẳn (correctness=0): **{len(bad)}** câu: {', '.join(sorted(bad)[:15]) or '—'}")
        jerr = sum(1 for r in records[last].values() if r.get("judge_error"))
        L.append(f"- Judge lỗi (không chấm được, đã loại khỏi trung bình): {jerr} câu")

    L.append("\n## 6. Hạn chế\n")
    L.append("- Cỡ mẫu nhỏ: CI rộng; chênh lệch nhỏ không nên được diễn giải là cải thiện.")
    L.append("- Judge là LLM, chưa hiệu chuẩn tuyệt đối — xem `audit.py` để đo độ đồng thuận với người chấm.")
    L.append("- Câu hỏi sinh tự động có xu hướng trùng từ vựng với tài liệu (retrieval dễ hơn thực tế).")
    L.append("- Orchestrator bị bỏ qua ở mọi cấu hình (intent cố định 'qa').")

    summary = {"meta": meta, "efficiency": eff, "deltas": deltas,
               "metrics": {c: {m: bootstrap_ci(v.values(), n_boot=n_boot) for m, v in values[c].items()} for c in configs}}
    return summary, "\n".join(L) + "\n"