"""
app/retrieval/reranker.py

rerank(query, candidates, top_k=5) — dùng BAAI/bge-reranker-v2-m3
(FlagEmbedding.FlagReranker) để chấm điểm chính xác cặp (query, chunk_text)
từ top-N hybrid search (thường top-20) xuống top-k cuối cùng đưa vào
Generation Agent.

Ghi chú vận hành quan trọng:
- Model được load 1 LẦN DUY NHẤT (lazy singleton, thread-safe qua
  double-checked locking) vì load checkpoint ~570M tham số tốn vài giây —
  KHÔNG load lại mỗi request.
- Đây là bước chạy trên MỌI request QA/Teaching -> nếu chạy CPU, latency
  rerank 20 cặp có thể là điểm nghẽn (thường vài trăm ms tới ~1s tuỳ CPU).
  Code tự bật fp16 khi có GPU. Nếu benchmark cho thấy CPU quá chậm, cân
  nhắc: (a) giảm số candidate đưa vào rerank (top-10 thay top-20),
  (b) dùng bản ONNX/quantized của bge-reranker-v2-m3, hoặc (c) tách rerank
  thành service riêng có GPU.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

_reranker_instance = None
_reranker_lock = threading.Lock()


class RerankerError(RuntimeError):
    """Model không load được, hoặc lỗi khi compute_score."""


def _detect_use_fp16() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _get_reranker():
    """Lazy singleton, double-checked locking để an toàn khi nhiều request
    concurrent cùng gọi rerank() lần đầu tiên (tránh load model N lần)."""
    global _reranker_instance
    if _reranker_instance is not None:
        return _reranker_instance

    with _reranker_lock:
        if _reranker_instance is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise RerankerError(
                    "FlagEmbedding chưa được cài. Chạy `pip install FlagEmbedding`."
                ) from exc

            use_fp16 = _detect_use_fp16()
            logger.info("Đang load reranker '%s' (fp16=%s)...", _MODEL_NAME, use_fp16)
            try:
                _reranker_instance = FlagReranker(_MODEL_NAME, use_fp16=use_fp16)
            except Exception as exc:
                raise RerankerError(f"Không load được reranker '{_MODEL_NAME}': {exc}") from exc
            logger.info("Reranker đã sẵn sàng.")

    return _reranker_instance


def _extract_text(candidate: dict[str, Any]) -> str:
    """Trích text từ 1 candidate — hỗ trợ vài shape phổ biến để linh hoạt
    với output của bước hybrid search (vector + BM25):
        {"text": "..."}                    # phẳng
        {"chunk": {"text": "..."}}          # lồng dict
        {"chunk": Chunk(...)}               # lồng Pydantic Chunk (app.models.schemas.Chunk)
    """
    if isinstance(candidate.get("text"), str) and candidate["text"]:
        return candidate["text"]

    chunk = candidate.get("chunk")
    if chunk is not None:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            if text:
                return text
        else:
            text = getattr(chunk, "text", None)
            if text:
                return text

    raise RerankerError(
        f"Không tìm thấy field text trong candidate — cần key 'text' hoặc "
        f"'chunk.text'. Candidate keys: {list(candidate.keys())}"
    )


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-sort `candidates` (thường là top-20 từ hybrid search) theo độ liên
    quan thật với `query`, dùng cross-encoder bge-reranker-v2-m3, trả về
    top_k candidate liên quan nhất.

    Mỗi candidate trong output được thêm key "rerank_score" (float 0-1 nhờ
    normalize=True) để debug/log — các key gốc được giữ nguyên.

    Args:
        query: câu hỏi (đã qua rewrite_query nếu cần).
        candidates: list dict, mỗi dict chứa ít nhất text của chunk (xem
            `_extract_text` để biết các shape được hỗ trợ).
        top_k: số candidate giữ lại sau rerank.

    Returns:
        list[dict] độ dài <= top_k, đã sort giảm dần theo rerank_score.
        Trả về [] nếu candidates rỗng.

    Raises:
        RerankerError: model không load được, hoặc candidate thiếu text.
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    pairs = [[query, _extract_text(candidate)] for candidate in candidates]

    try:
        scores = reranker.compute_score(pairs, normalize=True)
    except Exception as exc:
        raise RerankerError(f"Lỗi khi compute_score {len(pairs)} cặp: {exc}") from exc

    # compute_score trả về float thô (không phải list) khi chỉ có 1 cặp — chuẩn hoá lại
    if isinstance(scores, (float, int)):
        scores = [scores]

    scored_candidates = []
    for candidate, score in zip(candidates, scores):
        enriched = dict(candidate)
        enriched["rerank_score"] = float(score)
        scored_candidates.append(enriched)

    scored_candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored_candidates[:top_k]