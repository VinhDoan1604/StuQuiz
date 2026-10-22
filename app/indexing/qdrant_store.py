"""
Wrapper quanh qdrant-client (dense + sparse named vectors của BGE-M3).

Cô lập dữ liệu:
- Point ID = chunk.id (uuid4 duy nhất) => hai tài liệu trùng tên KHÔNG ghi đè nhau.
- Mọi truy vấn BẮT BUỘC kèm danh sách doc_id (đã được tầng API xác minh thuộc user).
  Không có doc_id => trả rỗng, không bao giờ tìm trên toàn collection.
- KHÔNG còn khớp theo filename (trước đây 2 user cùng upload 'slide.pdf' đọc lẫn nhau).
- payload có thêm owner_id (phòng thủ nhiều lớp / phục vụ dọn dẹp, audit).
"""
import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    FusionQuery,
    Fusion,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings

logger = logging.getLogger("app.indexing.qdrant")

COLLECTION_NAME = "studyrag_chunks"
DENSE_DIM = 1024  # BGE-M3
UPSERT_BATCH = 128

client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

_indexes_ready = False


def _ensure_payload_indexes() -> None:
    """Index keyword cho các field dùng để lọc (nhanh + bắt buộc ở Qdrant Cloud)."""
    global _indexes_ready
    if _indexes_ready:
        return
    for field in ("doc_id", "owner_id"):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:  # đã tồn tại / phiên bản không hỗ trợ -> bỏ qua
            logger.debug("create_payload_index(%s): %s", field, exc)
    _indexes_ready = True


def ensure_collection() -> None:
    if not client.collection_exists(COLLECTION_NAME):
        logger.info("Tạo mới collection Qdrant: %s", COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
    _ensure_payload_indexes()


def upsert_chunks(chunks: list, vectors: list, owner_id: str | None = None) -> None:
    """
    chunks: list[Chunk]; vectors: list[{"dense": [...], "sparse": {"indices": [...], "values": [...]}}]
    """
    ensure_collection()
    points = []
    for chunk, vec in zip(chunks, vectors):
        payload = {
            "text": chunk.text,
            "source_file": chunk.source_file,
            "page": chunk.page,
            "section": chunk.section,
            "content_type": chunk.content_type,
            "doc_id": str(chunk.doc_id),
            "parent_id": chunk.parent_id,
        }
        if owner_id:
            payload["owner_id"] = str(owner_id)
        if getattr(chunk, "subject", None):
            payload["subject"] = chunk.subject

        points.append(
            PointStruct(
                id=chunk.id,  # uuid4 duy nhất, KHÔNG suy từ tên file
                vector={"dense": vec["dense"], "sparse": vec["sparse"]},
                payload=payload,
            )
        )

    for i in range(0, len(points), UPSERT_BATCH):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i : i + UPSERT_BATCH])
    logger.info("Đã upsert %d points vào %s", len(points), COLLECTION_NAME)


def delete_document_vectors(document_id: str) -> None:
    """Xoá toàn bộ vector của 1 tài liệu (gọi khi user xoá tài liệu)."""
    ensure_collection()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=str(document_id)))]
            )
        ),
    )
    logger.info("Đã xoá vector của document %s", document_id)


def _doc_filter(document_ids: list[str], metadata_filter: dict | None = None) -> Filter:
    must = [FieldCondition(key="doc_id", match=MatchAny(any=[str(d) for d in document_ids]))]
    for key, value in (metadata_filter or {}).items():
        if value is not None:
            must.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=must)


def _format_point(pt, score: float | None = None) -> dict[str, Any]:
    payload = pt.payload or {}
    return {
        "id": pt.id,
        "score": pt.score if score is None else score,
        "text": payload.get("text", ""),
        "source_file": payload.get("source_file"),
        "page": payload.get("page"),
        "section": payload.get("section"),
        "content_type": payload.get("content_type"),
        "doc_id": payload.get("doc_id"),
        "parent_id": payload.get("parent_id"),
        "payload": payload,
    }


def hybrid_search(
    dense_vec: list[float],
    sparse_vec: dict,
    top_k: int = 20,
    metadata_filter: dict | None = None,
    document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search Dense + Sparse (RRF). `document_ids` là BẮT BUỘC."""
    if not document_ids:
        logger.warning("hybrid_search gọi không kèm document_ids -> trả rỗng (chặn tìm toàn cục).")
        return []

    ensure_collection()
    qdrant_filter = _doc_filter(document_ids, metadata_filter)

    prefetch_dense = Prefetch(query=dense_vec, using="dense", limit=top_k * 2, filter=qdrant_filter)
    prefetch_sparse = Prefetch(
        query=SparseVector(indices=sparse_vec.get("indices", []), values=sparse_vec.get("values", [])),
        using="sparse",
        limit=top_k * 2,
        filter=qdrant_filter,
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[prefetch_dense, prefetch_sparse],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    )
    return [_format_point(pt) for pt in response.points]


def get_all_chunks_for_documents(document_ids: list[str]) -> list[dict[str, Any]]:
    """Scroll toàn bộ chunk của các document (phục vụ tóm tắt đầy đủ)."""
    if not document_ids:
        return []
    ensure_collection()
    qdrant_filter = _doc_filter(document_ids)

    results: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qdrant_filter,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        results.extend(_format_point(pt, score=1.0) for pt in points)
        if offset is None:
            break
    return results