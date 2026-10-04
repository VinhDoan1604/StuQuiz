"""
Script kiểm thử độc lập cho các thành phần của Phase 1.
Chạy: python scripts/test_phase1.py
"""
import sys
import os

# Đảm bảo đường dẫn gốc của project có trong sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def test_config():
    print("--- 1. Kiểm tra Cấu hình (app/config.py) ---")
    from app.config import settings
    print(f"Cheap Provider: {settings.cheap_model_provider}")
    print(f"Cheap Model: {settings.cheap_model_name}")
    print(f"Strong Provider: {settings.strong_model_provider}")
    print(f"Strong Model: {settings.strong_model_name}")
    print(f"Effective Google Key: {'[Đã cấu hình]' if settings.effective_google_api_key else '[Chưa có]'}")
    assert settings.cheap_model_provider == "google", "cheap_model_provider phải là google"
    assert settings.strong_model_provider == "google", "strong_model_provider phải là google"
    print("=> Cấu hình hợp lệ!\n")


def test_router():
    print("--- 2. Kiểm tra LLM Router (app/llm/router.py) ---")
    from app.llm.router import get_llm, get_cumulative_usage, reset_usage_metrics

    # Test Cheap Task (rewrite)
    llm_rewrite = get_llm("rewrite")
    print(f"Task 'rewrite' -> Model: {getattr(llm_rewrite, 'model', getattr(llm_rewrite, 'model_name', 'unknown'))}, Temperature: {getattr(llm_rewrite, 'temperature', None)}")
    assert getattr(llm_rewrite, "temperature", None) == 0.1, "Temperature cho rewrite phải là 0.1"

    # Test Strong Task (teaching)
    llm_teaching = get_llm("teaching")
    print(f"Task 'teaching' -> Model: {getattr(llm_teaching, 'model', getattr(llm_teaching, 'model_name', 'unknown'))}, Temperature: {getattr(llm_teaching, 'temperature', None)}")
    assert getattr(llm_teaching, "temperature", None) == 0.7, "Temperature cho teaching phải là 0.7"

    # Test Task (grade)
    llm_grade = get_llm("grade")
    print(f"Task 'grade' -> Model: {getattr(llm_grade, 'model', getattr(llm_grade, 'model_name', 'unknown'))}, Temperature: {getattr(llm_grade, 'temperature', None)}")
    assert getattr(llm_grade, "temperature", None) == 0.0, "Temperature cho grade phải là 0.0"

    usage = get_cumulative_usage()
    print(f"Initial Usage Metrics: {usage}")
    print("=> LLM Router hợp lệ!\n")


def test_embedder_caching_and_formatting():
    print("--- 3. Kiểm tra BGE-M3 Embedder Caching & Format (app/embedding/bge_m3_embedder.py) ---")
    from app.embedding.bge_m3_embedder import (
        _hash_text,
        _format_single_embedding,
        get_cache_size,
        clear_embedding_cache,
    )

    clear_embedding_cache()
    text = "Thuật toán tìm kiếm theo chiều sâu DFS"
    h1 = _hash_text(text)
    h2 = _hash_text("   Thuật toán tìm kiếm theo chiều sâu DFS   ")
    assert h1 == h2, "Hash phải loại bỏ khoảng trắng thừa ở đầu/cuối"

    # Giả lập kết quả encode
    dummy_dense = [0.1] * 1024
    dummy_lex = {101: 0.85, 2024: 0.42}
    formatted = _format_single_embedding(dummy_dense, dummy_lex)

    assert len(formatted["dense"]) == 1024, "Dense vector phải có đúng 1024 chiều"
    assert formatted["sparse"]["indices"] == [101, 2024], "Sparse indices phải đúng"
    assert formatted["sparse"]["values"] == [0.85, 0.42], "Sparse values phải đúng"

    print(f"Dense dimensions: {len(formatted['dense'])}")
    print(f"Sparse keys: {list(formatted['sparse'].keys())}")
    print("=> BGE-M3 Formatting & Cache Logic hợp lệ!\n")


def test_qdrant_store_definitions():
    print("--- 4. Kiểm tra Qdrant Store Definitions (app/indexing/qdrant_store.py) ---")
    import inspect
    from app.indexing.qdrant_store import hybrid_search, upsert_chunks, ensure_collection, COLLECTION_NAME, DENSE_DIM

    assert COLLECTION_NAME == "studyrag_chunks"
    assert DENSE_DIM == 1024

    sig = inspect.signature(hybrid_search)
    params = list(sig.parameters.keys())
    assert params in (
        ["dense_vec", "sparse_vec", "top_k", "metadata_filter"],
        ["dense_vec", "sparse_vec", "top_k", "metadata_filter", "document_ids"]
    ), f"Tham số hybrid_search không đúng: {params}"

    sig_upsert = inspect.signature(upsert_chunks)
    assert list(sig_upsert.parameters.keys()) == ["chunks", "vectors"]

    print("=> Qdrant Store Interface hợp lệ!\n")


if __name__ == "__main__":
    print("========================================")
    print("BẮT ĐẦU KIỂM THỬ GIAI ĐOẠN 1 (PHASE 1)")
    print("========================================\n")
    test_config()
    test_router()
    test_embedder_caching_and_formatting()
    test_qdrant_store_definitions()
    print("========================================")
    print("TẤT CẢ TEST ĐƠN VỊ PHASE 1 ĐỀU THÀNH CÔNG!")
    print("========================================")
