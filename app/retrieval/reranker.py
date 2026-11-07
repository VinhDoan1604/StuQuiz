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
- Đây là bước chạy trên MỌI request QA/Teaching -> trên CPU (fp32) rerank 20
  cặp ~512 token với model 568M là điểm nghẽn LỚN (đo được ~100s/câu trên
  laptop trong eval smoke). Code tự bật fp16 khi có GPU.
- FlagEmbedding 1.4.x "dò batch size" bằng cách forward thử batch đầu rồi
  forward LẠI batch đó trong vòng lặp chính. Với batch mặc định 128 (> số cặp)
  toàn bộ cặp bị tính 2 lần -> đặt _BATCH_SIZE nhỏ để phần lặp chỉ còn vài cặp. Nếu benchmark cho thấy CPU quá chậm, cân
  nhắc: (a) giảm số candidate đưa vào rerank (top-10 thay top-20),
  (b) dùng bản ONNX/quantized của bge-reranker-v2-m3, hoặc (c) tách rerank
  thành service riêng có GPU.
- Các nút vặn đều nằm trong settings (RERANK_CANDIDATES, RERANKER_MODEL,
  RERANKER_MAX_LENGTH, RERANKER_INT8, RERANKER_DEVICE). Đo trước khi đổi:
  `python -m scripts.bench_reranker` (xem EVAL_FOLLOWUP.md).
- RERANKER_INT8=true: torch dynamic quantization (nn.Linear -> qint8) trên CPU.
  Không cần ONNX/optimum; điểm số lệch nhẹ so với fp32 nên PHẢI đo lại
  retrieval (eval --retrieval-only) trước khi bật cho production.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_BATCH_SIZE = 4  # chỉ dùng khi rơi về FlagReranker.compute_score (xem _score_pairs)
_SCORE_BATCH_SIZE = 8  # batch cho vòng chấm điểm tự viết (cặp đã sắp theo độ dài)

_reranker_instance = None
_reranker_lock = threading.Lock()


class RerankerError(RuntimeError):
    """Model không load được, hoặc lỗi khi compute_score."""


def _resolve_device() -> str:
    from app.config import settings

    if settings.reranker_device.strip():
        return settings.reranker_device.strip()
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _quantize_int8(reranker) -> None:
    """Thay nn.Linear bằng bản qint8 (dynamic quantization) — chỉ có ý nghĩa trên CPU."""
    import torch

    reranker.model = torch.quantization.quantize_dynamic(reranker.model, {torch.nn.Linear}, dtype=torch.qint8)


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

            from app.config import settings

            model_name = settings.reranker_model
            device = _resolve_device()
            use_fp16 = device.startswith("cuda")
            int8 = settings.reranker_int8 and device == "cpu"
            logger.info(
                "Đang load reranker '%s' (device=%s, fp16=%s, int8=%s, max_length=%d)...",
                model_name, device, use_fp16, int8, settings.reranker_max_length,
            )
            try:
                instance = FlagReranker(
                    model_name,
                    use_fp16=use_fp16,
                    batch_size=_BATCH_SIZE,
                    max_length=settings.reranker_max_length,
                    devices=[device],
                )
                if int8:
                    _quantize_int8(instance)
            except Exception as exc:
                raise RerankerError(f"Không load được reranker '{model_name}': {exc}") from exc
            _reranker_instance = instance
            logger.info("Reranker đã sẵn sàng.")

    return _reranker_instance


def reset_reranker() -> None:
    """Bỏ model đã load (benchmark đổi cấu hình giữa chừng / test)."""
    global _reranker_instance
    with _reranker_lock:
        _reranker_instance = None


def warmup() -> None:
    """Tải model sẵn lúc khởi động (PRELOAD_MODELS=true)."""
    _get_reranker()


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


def _ensure_torch_threads(torch) -> None:
    """Trong thread phụ (asyncio.to_thread — đường gọi của production và eval), torch chỉ
    dùng ~nửa số luồng của main thread (đo trên máy dev: 4 thay vì 8) => rerank chậm gần
    gấp đôi (37s so với 20s cho 20 ứng viên). Đặt lại số luồng ngay trong thread đang chấm."""
    from app.config import settings

    want = settings.reranker_threads or max(1, (os.cpu_count() or 2) // 2)
    if torch.get_num_threads() != want:
        torch.set_num_threads(want)


def _score_pairs(reranker, pairs: list[list[str]]) -> list[float]:
    """Chấm điểm cặp (query, passage), trả về điểm sigmoid 0-1.

    Tự chạy forward thay vì `FlagReranker.compute_score`: bản 1.4.x forward thử batch
    đầu để "dò" batch size rồi tính LẠI batch đó trong vòng lặp chính => lãng phí
    _BATCH_SIZE cặp mỗi lần (~40% khi chỉ có 10 ứng viên). Mã hoá giống FlagEmbedding:
    cặp câu chuẩn của tokenizer, cắt phía passage ("only_second"), sắp theo độ dài để
    padding ít nhất. Nếu cấu trúc nội bộ của FlagReranker đổi, rơi về compute_score.
    """
    model = getattr(reranker, "model", None)
    tokenizer = getattr(reranker, "tokenizer", None)
    if model is None or tokenizer is None:
        return list(reranker.compute_score(pairs, normalize=True))

    import torch

    _ensure_torch_threads(torch)
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    max_length = getattr(reranker, "max_length", 512)
    order = sorted(range(len(pairs)), key=lambda i: -len(pairs[i][1]))
    scores = [0.0] * len(pairs)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(order), _SCORE_BATCH_SIZE):
            idx = order[start:start + _SCORE_BATCH_SIZE]
            inputs = tokenizer(
                [pairs[i][0] for i in idx],
                [pairs[i][1] for i in idx],
                padding=True,
                truncation="only_second",
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs, return_dict=True).logits.view(-1).float()
            for i, v in zip(idx, torch.sigmoid(logits).cpu().tolist()):
                scores[i] = float(v)
    return scores


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
        scores = _score_pairs(reranker, pairs)
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
