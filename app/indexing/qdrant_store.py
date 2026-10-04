# STATUS: READY (Hoàn thành Phase 1)
"""
Wrapper quanh qdrant-client. BGE-M3 sinh đồng thời dense + sparse vector
trong 1 model (xem app/embedding/bge_m3_embedder.py), collection dùng named
vectors "dense" và "sparse" thay vì phải ghép thêm BM25 riêng.

Hỗ trợ:
- Khởi tạo collection đa vector (ensure_collection)
- Upsert points kèm metadata và ID duy nhất (upsert_chunks)
- Hybrid Search kết hợp Reciprocal Rank Fusion (RRF) và metadata filtering (hybrid_search)
"""
import uuid
import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    Prefetch,
    SparseVector,
    FusionQuery,
    Fusion,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

from app.config import settings

logger = logging.getLogger("app.indexing.qdrant")

COLLECTION_NAME = "studyrag_chunks"
DENSE_DIM = 1024  # BGE-M3 dense dimension

client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def ensure_collection() -> None:
    """Tạo collection với cấu hình Dense + Sparse vector nếu chưa tồn tại."""
    if client.collection_exists(COLLECTION_NAME):
        return
    logger.info("Tạo mới collection Qdrant: %s", COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )


def upsert_chunks(chunks: list, vectors: list) -> None:
    """
    chunks: list[Chunk] (app.models.schemas)
    vectors: list[dict] — mỗi phần tử {"dense": [...], "sparse": {"indices": [...], "values": [...]}}
    """
    ensure_collection()
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        # Tạo UUID mang tính tất định để tránh trùng lặp hoặc xung đột ID khi upload nhiều file
        point_id = str(uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"{chunk.source_file}_{chunk.page}_{chunk.section}_{i}"
        ))

        payload = {
            "text": chunk.text,
            "source_file": chunk.source_file,
            "page": chunk.page,
            "section": chunk.section,
            "content_type": chunk.content_type,
        }
        if hasattr(chunk, "subject") and chunk.subject:
            payload["subject"] = chunk.subject
        if hasattr(chunk, "doc_id") and chunk.doc_id:
            payload["doc_id"] = str(chunk.doc_id)
        elif hasattr(chunk, "metadata") and getattr(chunk.metadata, "doc_id", None):
            payload["doc_id"] = str(chunk.metadata.doc_id)

        points.append(PointStruct(
            id=point_id,
            vector={"dense": vec["dense"], "sparse": vec["sparse"]},
            payload=payload,
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Đã upsert %d points vào collection %s", len(points), COLLECTION_NAME)


def _resolve_document_identifiers(document_ids: list[str]) -> list[str]:
    """
    Nhận danh sách document_ids (có thể là UUID của Document trong Postgres, hoặc filename).
    Truy vấn Postgres để lấy thêm cả filename tương ứng (hoặc ngược lại)
    nhằm đảm bảo tìm kiếm trong Qdrant luôn trúng cả doc_id lẫn source_file.
    """
    if not document_ids:
        return []

    identifiers = set(str(d) for d in document_ids if d)
    try:
        from app.db.session import SessionLocal
        from app.models.db_models import Document
        with SessionLocal() as db:
            docs = db.query(Document).filter(
                (Document.id.in_(list(identifiers))) | (Document.filename.in_(list(identifiers)))
            ).all()
            for doc in docs:
                identifiers.add(str(doc.id))
                if doc.filename:
                    identifiers.add(doc.filename)
    except Exception as exc:
        logger.warning("Không thể resolve document_ids từ DB: %s", exc)

    return [x for x in identifiers if x]


def hybrid_search(
    dense_vec: list[float],
    sparse_vec: dict,
    top_k: int = 20,
    metadata_filter: dict | None = None,
    document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Hybrid search kết hợp Dense (Cosine) và Sparse (Lexical) qua Reciprocal Rank Fusion (RRF).
    
    Hỗ trợ metadata filtering theo các trường trong payload (ví dụ: source_file, subject)
    hoặc lọc theo danh sách document_ids.
    """
    ensure_collection()

    # 1. Xây dựng bộ lọc Metadata nếu có
    filter_conditions = []
    if metadata_filter:
        for key, value in metadata_filter.items():
            if value is not None:
                filter_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
    if document_ids:
        all_ids = _resolve_document_identifiers(document_ids)
        if all_ids:
            filter_conditions.append(
                Filter(
                    should=[
                        FieldCondition(key="doc_id", match=MatchAny(any=all_ids)),
                        FieldCondition(key="source_file", match=MatchAny(any=all_ids)),
                    ]
                )
            )
    qdrant_filter = Filter(must=filter_conditions) if filter_conditions else None

    # 2. Prefetch cho Dense search
    prefetch_dense = Prefetch(
        query=dense_vec,
        using="dense",
        limit=top_k * 2,
        filter=qdrant_filter,
    )

    # 3. Prefetch cho Sparse search (Lexical weights từ BGE-M3)
    indices = sparse_vec.get("indices", [])
    values = sparse_vec.get("values", [])
    prefetch_sparse = Prefetch(
        query=SparseVector(indices=indices, values=values),
        using="sparse",
        limit=top_k * 2,
        filter=qdrant_filter,
    )

    # 4. Hợp nhất bằng Reciprocal Rank Fusion (RRF) qua Query API của Qdrant
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[prefetch_dense, prefetch_sparse],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    )

    # 5. Định dạng kết quả trả về
    results = []
    for pt in response.points:
        payload = pt.payload or {}
        results.append({
            "id": pt.id,
            "score": pt.score,
            "text": payload.get("text", ""),
            "source_file": payload.get("source_file"),
            "page": payload.get("page"),
            "section": payload.get("section"),
            "content_type": payload.get("content_type"),
            "payload": payload,
        })

    return results


def get_all_chunks_for_documents(document_ids: list[str]) -> list[dict[str, Any]]:
    """
    Lấy toàn bộ chunk thuộc danh sách document_ids (dùng scroll API không qua vector search).
    Cần thiết cho Quiz & Summary Agent khi tóm tắt đầy đủ tài liệu.
    """
    ensure_collection()
    if not document_ids:
        return []

    all_ids = _resolve_document_identifiers(document_ids)
    if not all_ids:
        return []

    qdrant_filter = Filter(
        should=[
            FieldCondition(key="doc_id", match=MatchAny(any=all_ids)),
            FieldCondition(key="source_file", match=MatchAny(any=all_ids)),
        ]
    )

    results = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qdrant_filter,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in points:
            payload = pt.payload or {}
            results.append({
                "id": pt.id,
                "score": 1.0,
                "text": payload.get("text", ""),
                "source_file": payload.get("source_file"),
                "page": payload.get("page"),
                "section": payload.get("section"),
                "content_type": payload.get("content_type"),
                "payload": payload,
            })
        if next_offset is None:
            break
        offset = next_offset

    return results
