"""
Xoá vector "mồ côi" trong Qdrant: điểm có doc_id không còn tồn tại trong bảng
documents (vd sót lại sau khi DB bị tạo lại, hoặc xoá tài liệu khi Qdrant đang lỗi).

    python scripts/cleanup_orphan_vectors.py --dry-run   # chỉ thống kê
    python scripts/cleanup_orphan_vectors.py             # xoá thật

Nên chụp snapshot Qdrant trước khi xoá (scripts/upgrade_qdrant.py cũng làm việc này).
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchAny  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.indexing import qdrant_store as qs  # noqa: E402
from app.models.db_models import Document  # noqa: E402


def count_points_by_doc() -> Counter:
    counts: Counter = Counter()
    offset = None
    while True:
        points, offset = qs.client.scroll(
            qs.COLLECTION_NAME, limit=500, offset=offset, with_payload=["doc_id"], with_vectors=False
        )
        counts.update(str((p.payload or {}).get("doc_id")) for p in points)
        if offset is None:
            return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not qs.client.collection_exists(qs.COLLECTION_NAME):
        print("Collection chưa tồn tại — không có gì để dọn.")
        return

    with SessionLocal() as db:
        live = {str(r[0]) for r in db.query(Document.id).all()}

    by_doc = count_points_by_doc()
    orphans = {doc: n for doc, n in by_doc.items() if doc not in live}
    print(f"Tổng điểm: {sum(by_doc.values())} | tài liệu trong DB: {len(live)} | "
          f"điểm mồ côi: {sum(orphans.values())} thuộc {len(orphans)} doc_id")
    for doc, n in sorted(orphans.items(), key=lambda x: -x[1]):
        print(f"  - doc_id={doc}: {n} điểm")

    if not orphans or args.dry_run:
        return

    qs.client.delete(
        qs.COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=list(orphans)))])
        ),
        wait=True,
    )
    remaining = sum(count_points_by_doc().values())
    print(f"Đã xoá. Còn lại {remaining} điểm.")


if __name__ == "__main__":
    main()
