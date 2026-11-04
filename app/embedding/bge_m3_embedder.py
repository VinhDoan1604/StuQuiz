"""
BGE-M3 Embedder — sinh đồng thời Dense vector (1024 chiều)
và Sparse vector (Lexical Weights) trong 1 model duy nhất.

Đặc tính kỹ thuật:
- Sử dụng FlagEmbedding.BGEM3FlagModel.
- Hỗ trợ ngữ cảnh tối đa 8192 tokens, tối ưu xuất sắc cho đa ngôn ngữ & tiếng Việt.
- Tự động phát hiện thiết bị phần cứng: CUDA (GPU) hoặc CPU.
- Cơ chế Batching theo cấu hình (mặc định batch_size=16) để chống OOM.
- Cache LRU theo hash SHA-256 (giới hạn EMBEDDING_CACHE_SIZE) tránh encode lại text lặp.
- Định dạng xuất chuẩn khớp 100% với Qdrant Store:
  {"dense": list[float], "sparse": {"indices": list[int], "values": list[float]}}
"""
import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import Any

from app.config import settings

logger = logging.getLogger("app.embedding.bge_m3")

DENSE_DIM = 1024

_MODEL = None
_MODEL_LOCK = threading.Lock()


class _LRUCache:
    """Cache LRU thread-safe có giới hạn kích thước (tránh phình bộ nhớ theo thời gian)."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._data: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


_EMBEDDING_CACHE = _LRUCache(settings.embedding_cache_size)


def _hash_text(text: str) -> str:
    """Tạo hash SHA-256 duy nhất từ nội dung chuỗi (đã chuẩn hóa strip)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def get_cache_size() -> int:
    """Trả về số lượng embeddings đang được lưu trong cache."""
    return len(_EMBEDDING_CACHE)


def clear_embedding_cache() -> None:
    """Xóa toàn bộ cache in-memory."""
    _EMBEDDING_CACHE.clear()


def get_model():
    """
    Lazy loader cho BGEM3FlagModel.
    Chỉ khởi tạo và tải weights khi có lời gọi embedding đầu tiên.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:  # double-checked locking: nhiều request đồng thời chỉ load 1 lần
        if _MODEL is not None:
            return _MODEL
        logger.info("Đang khởi tạo mô hình BGE-M3 (BAAI/bge-m3)...")
        try:
            import torch
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            logger.error("Thiếu thư viện FlagEmbedding hoặc PyTorch: %s", exc)
            raise ImportError("Vui lòng cài đặt: pip install FlagEmbedding torch") from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_fp16 = device == "cuda"
        if device == "cpu":
            cpu_threads = min(4, os.cpu_count() or 4)
            torch.set_num_threads(cpu_threads)
            logger.info("Thiết lập PyTorch CPU threads: %d", cpu_threads)

        logger.info("BGE-M3 được tải trên thiết bị: %s (fp16=%s)", device, use_fp16)
        _MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16, device=device)
        return _MODEL


def _format_single_embedding(dense_vec: Any, lex_weights: dict) -> dict[str, Any]:
    """Chuyển đổi kết quả encode của BGE-M3 sang định dạng chuẩn Qdrant."""
    # 1. Dense vector
    if hasattr(dense_vec, "tolist"):
        dense_list = dense_vec.tolist()
    else:
        dense_list = list(dense_vec)

    # 2. Sparse vector (lexical weights mapping -> indices & values)
    indices: list[int] = []
    values: list[float] = []
    for k, v in lex_weights.items():
        try:
            indices.append(int(k))
            values.append(float(v))
        except (ValueError, TypeError):
            continue

    return {
        "dense": dense_list,
        "sparse": {
            "indices": indices,
            "values": values,
        },
    }


def embed_chunks(texts: list[str], batch_size: int = 6) -> list[dict[str, Any]]:
    """
    Nhúng một danh sách các chunks văn bản.

    Tối ưu hóa:
    - Hash Caching
    - Nhóm theo độ dài (Length-sorted batching) để triệt tiêu chi phí padding ma trận
    - Giới hạn max_length=512 (khớp độ dài chunk ~200-400 token, giảm 4x chi phí self-attention O(N^2))
    - torch.inference_mode() loại bỏ tracking graph trên CPU
    """
    if not texts:
        return []

    results: list[dict[str, Any] | None] = [None] * len(texts)
    missing_indices: list[int] = []
    missing_texts: list[str] = []

    # 1. Kiểm tra trong Cache
    for i, text in enumerate(texts):
        cached = _EMBEDDING_CACHE.get(_hash_text(text))
        if cached is not None:
            results[i] = cached
        else:
            missing_indices.append(i)
            missing_texts.append(text)

    # 2. Nếu có text chưa được cache, thực hiện encode theo batch tối ưu
    if missing_texts:
        import torch
        model = get_model()
        num_missing = len(missing_texts)
        logger.info(
            "Thực hiện embed %d/%d chunks (đã cache: %d, batch_size: %d, max_len: 512)",
            num_missing,
            len(texts),
            len(texts) - num_missing,
            batch_size,
        )

        # Sắp xếp theo độ dài ký tự để tối ưu padding trong từng batch
        missing_items = list(zip(missing_indices, missing_texts))
        missing_items.sort(key=lambda x: len(x[1]))

        sorted_indices = [x[0] for x in missing_items]
        sorted_texts = [x[1] for x in missing_items]

        with torch.inference_mode():
            for start_idx in range(0, num_missing, batch_size):
                end_idx = min(start_idx + batch_size, num_missing)
                batch = sorted_texts[start_idx:end_idx]

                output = model.encode(
                    batch,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                    max_length=512,
                    batch_size=len(batch),
                )

                dense_vecs = output["dense_vecs"]
                lex_weights_list = output["lexical_weights"]

                for offset, (dense_vec, lex_weights) in enumerate(zip(dense_vecs, lex_weights_list)):
                    original_idx = sorted_indices[start_idx + offset]
                    original_text = batch[offset]

                    formatted = _format_single_embedding(dense_vec, lex_weights)
                    _EMBEDDING_CACHE.put(_hash_text(original_text), formatted)
                    results[original_idx] = formatted

    return [r for r in results if r is not None]


def embed_query(text: str) -> dict[str, Any]:
    """
    Dùng riêng cho việc nhúng truy vấn tại thời điểm retrieval.
    Không xử lý batch, kiểm tra và lưu hash cache đơn lẻ.
    """
    h = _hash_text(text)
    cached = _EMBEDDING_CACHE.get(h)
    if cached is not None:
        return cached

    model = get_model()
    output = model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        max_length=512,
    )

    dense_vec = output["dense_vecs"][0]
    lex_weights = output["lexical_weights"][0]

    formatted = _format_single_embedding(dense_vec, lex_weights)
    _EMBEDDING_CACHE.put(h, formatted)
    return formatted
