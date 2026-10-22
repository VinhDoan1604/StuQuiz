"""Test các hàm thuần của eval (không cần LLM/Qdrant). Chạy: pytest tests/test_eval_metrics.py -v"""
import json

import pytest

from eval.dataset import DatasetError, load_dataset
from eval.metrics import (
    bootstrap_ci,
    evidence_match,
    normalize_text,
    paired_bootstrap_diff,
    percentile,
    retrieval_metrics,
)

CHUNK = (
    "Thuật toán Dijkstra hoạt động theo chiến lược tham lam. Nó giả định rằng khoảng cách\n"
    "ngắn nhất đến một đỉnh đã chốt là tối ưu vĩnh viễn, nên không xử lý được cạnh âm."
)


# ------------------------------------------------------------ evidence_match
def test_normalize_collapses_case_punct_whitespace():
    assert normalize_text("  Xin,  CHÀO!\n Bạn ") == "xin chào bạn"


def test_evidence_exact_substring_across_linebreak():
    ev = "giả định rằng khoảng cách ngắn nhất đến một đỉnh đã chốt là tối ưu vĩnh viễn"
    assert evidence_match(CHUNK, ev)


def test_evidence_fuzzy_trigram_tolerates_small_ocr_error():
    ev = "Nó giả định rằng khoảng cách ngắn nhất đến một đỉnh đã chốt là tối ưu vĩnh viễn nên không xử lý được cạnh âm"
    noisy = CHUNK.replace("vĩnh viễn", "vĩnh viển")
    assert evidence_match(noisy, ev, min_containment=0.8)


def test_evidence_not_present():
    assert not evidence_match(CHUNK, "Bellman-Ford có độ phức tạp O(V*E) và phát hiện chu trình âm")


def test_short_evidence_requires_exact_substring():
    assert evidence_match(CHUNK, "tham lam")
    assert not evidence_match(CHUNK, "tham ăn")


# --------------------------------------------------------- retrieval_metrics
def test_retrieval_metrics_rank_and_recall():
    chunks = [{"text": "không liên quan gì cả"}, {"text": CHUNK}, {"text": "cũng không liên quan"}]
    evs = ["khoảng cách ngắn nhất đến một đỉnh đã chốt là tối ưu vĩnh viễn", "Bellman-Ford phát hiện chu trình âm"]
    m = retrieval_metrics(chunks, evs, ks=(1, 3))
    assert m["hit@1"] == 0.0
    assert m["hit@3"] == 1.0
    assert m["recall@3"] == 0.5  # mới phủ 1/2 bằng chứng
    assert m["mrr"] == pytest.approx(0.5)


def test_retrieval_metrics_no_match_is_zero():
    m = retrieval_metrics([{"text": "abc def"}], ["một câu hoàn toàn khác biệt nằm ngoài tài liệu"])
    assert m["mrr"] == 0.0 and m["hit@5"] == 0.0


def test_retrieval_metrics_requires_evidence():
    with pytest.raises(ValueError):
        retrieval_metrics([{"text": "x"}], [])


# ---------------------------------------------------------------- thống kê
def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([5], 95) == 5


def test_bootstrap_ci_deterministic_and_contains_mean():
    vals = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    a = bootstrap_ci(vals, n_boot=500, seed=1)
    b = bootstrap_ci(vals, n_boot=500, seed=1)
    assert a == b
    mean, lo, hi = a
    assert lo <= mean <= hi and mean == pytest.approx(0.7)


def test_bootstrap_ignores_none_and_handles_empty():
    assert bootstrap_ci([None, None])[0] != bootstrap_ci([None, None])[0]  # NaN
    assert bootstrap_ci([1.0, None, 1.0])[0] == 1.0


def test_paired_diff_detects_clear_improvement():
    a = {f"q{i}": 1.0 for i in range(30)}
    b = {f"q{i}": 0.0 for i in range(30)}
    d, lo, hi, n = paired_bootstrap_diff(a, b, n_boot=300)
    assert d == 1.0 and lo > 0 and n == 30


def test_paired_diff_no_effect_ci_contains_zero():
    a = {f"q{i}": float(i % 2) for i in range(40)}
    b = {f"q{i}": float((i + 1) % 2) for i in range(40)}
    d, lo, hi, _ = paired_bootstrap_diff(a, b, n_boot=500)
    assert lo <= 0 <= hi


def test_paired_diff_only_uses_common_keys():
    d, _, _, n = paired_bootstrap_diff({"a": 1.0, "b": 1.0}, {"b": 0.0, "c": 0.0}, n_boot=50)
    assert n == 1 and d == 1.0


# ------------------------------------------------------------------ dataset
def _write(tmp_path, *objs):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in objs), encoding="utf-8")
    return p


def test_dataset_valid_and_unreviewed_skipped(tmp_path):
    p = _write(
        tmp_path,
        {"id": "a", "question": "Q?", "gold_answer": "A", "gold_evidence": ["e e e e e e"]},
        {"id": "b", "question": "Q2?", "gold_answer": "A", "gold_evidence": ["e"], "reviewed": False},
        {"id": "c", "question": "Q3?", "answerable": False},
    )
    items = load_dataset(p)
    assert [i.id for i in items] == ["a", "c"]
    assert len(load_dataset(p, include_unreviewed=True)) == 3


def test_dataset_rejects_answerable_without_evidence(tmp_path):
    p = _write(tmp_path, {"id": "a", "question": "Q?", "gold_answer": "A", "gold_evidence": []})
    with pytest.raises(DatasetError):
        load_dataset(p)


def test_dataset_rejects_unanswerable_with_evidence(tmp_path):
    p = _write(tmp_path, {"id": "a", "question": "Q?", "answerable": False, "gold_evidence": ["x"]})
    with pytest.raises(DatasetError):
        load_dataset(p)


def test_dataset_rejects_duplicate_ids(tmp_path):
    o = {"id": "a", "question": "Q?", "answerable": False}
    with pytest.raises(DatasetError):
        load_dataset(_write(tmp_path, o, o))