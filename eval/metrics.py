"""
eval/metrics.py

Các hàm THUẦN (không gọi LLM/DB) để đo retrieval và tổng hợp thống kê.

Gold của retrieval là *đoạn bằng chứng* (gold_evidence: trích nguyên văn từ tài
liệu), KHÔNG phải chunk id. Nhờ vậy nhãn vẫn đúng khi bạn đổi chiến lược
chunking — điều bắt buộc nếu muốn so sánh các cấu hình công bằng.
"""
from __future__ import annotations

import math
import random
import re
import unicodedata
from typing import Iterable, Optional, Sequence

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _trigrams(tokens: list[str]) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)}


def evidence_match(
    chunk_text: str,
    evidence: str,
    min_containment: float = 0.9,
    *,
    normalized: bool = False,
) -> bool:
    """
    Chunk có chứa bằng chứng không?
    - Khớp chuỗi con sau chuẩn hoá (bỏ dấu câu/khoảng trắng thừa), hoặc
    - >= min_containment số trigram từ của bằng chứng có mặt trong chunk
      (chịu được xuống dòng giữa câu / lỗi gạch nối của PDF).
    Bằng chứng ngắn (< 5 từ) chỉ chấp nhận khớp chuỗi con.
    """
    ch = chunk_text if normalized else normalize_text(chunk_text)
    ev = normalize_text(evidence)
    if not ev or not ch:
        return False
    if ev in ch:
        return True
    ev_tokens = ev.split()
    if len(ev_tokens) < 5:
        return False
    ev_tri = _trigrams(ev_tokens)
    if not ev_tri:
        return False
    return len(ev_tri & _trigrams(ch.split())) / len(ev_tri) >= min_containment


def retrieval_metrics(
    chunks: Sequence[dict], evidences: Sequence[str], ks: Sequence[int] = (1, 3, 5)
) -> dict[str, float]:
    """
    chunks: danh sách chunk theo thứ tự xếp hạng cuối cùng pipeline trả về.
    - hit@k    : có >= 1 bằng chứng nằm trong top-k.
    - recall@k : tỉ lệ bằng chứng (trong số cần thiết) được phủ bởi top-k.
    - mrr      : 1 / hạng của chunk liên quan đầu tiên (0 nếu không có).
    """
    if not evidences:
        raise ValueError("retrieval_metrics cần ít nhất 1 gold_evidence.")

    norm_texts = [normalize_text(c.get("text", "")) for c in chunks]
    matched: list[set[int]] = [
        {j for j, e in enumerate(evidences) if evidence_match(t, e, normalized=True)}
        for t in norm_texts
    ]

    first = next((i for i, m in enumerate(matched) if m), None)
    out: dict[str, float] = {"mrr": 1.0 / (first + 1) if first is not None else 0.0}
    for k in ks:
        covered: set[int] = set().union(*matched[:k]) if matched[:k] else set()
        out[f"hit@{k}"] = float(bool(covered))
        out[f"recall@{k}"] = len(covered) / len(evidences)
    return out


# ---------------------------------------------------------------------------
# Thống kê
# ---------------------------------------------------------------------------
def percentile(values: Iterable[float], p: float) -> float:
    vals = sorted(values)
    if not vals:
        return float("nan")
    k = (len(vals) - 1) * p / 100.0
    f = int(math.floor(k))
    c = min(f + 1, len(vals) - 1)
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def _boot_means(vals: list[float], n_boot: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(vals)
    return sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))


def bootstrap_ci(
    values: Iterable[Optional[float]], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, lo, hi) bằng percentile bootstrap. Bỏ qua None. Rỗng -> NaN."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        nan = float("nan")
        return nan, nan, nan
    means = _boot_means(vals, n_boot, seed)
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return sum(vals) / len(vals), lo, hi


def paired_bootstrap_diff(
    a: dict[str, Optional[float]],
    b: dict[str, Optional[float]],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """
    Chênh lệch trung bình (a - b) CÙNG TẬP CÂU HỎI (ghép cặp theo qid) kèm CI.
    Trả về (diff, lo, hi, n_pairs). CI chứa 0 => không đủ bằng chứng có khác biệt.
    """
    keys = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    diffs = [float(a[k]) - float(b[k]) for k in keys]
    if not diffs:
        nan = float("nan")
        return nan, nan, nan, 0
    mean, lo, hi = bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
    return mean, lo, hi, len(diffs)