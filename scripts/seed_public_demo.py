"""
scripts/seed_public_demo.py

Nạp bộ tài liệu CÔNG KHAI dùng chung cho tài khoản dùng thử (chỉ đọc, xem
app/core/ownership.py). CHỈ dùng tài liệu được phép công khai (Wikipedia, OpenStax...).
Chạy lại an toàn: tài liệu đã có cùng tên thì bỏ qua.

    python scripts/seed_public_demo.py eval/corpus/wiki_dijkstra.md eval/corpus/wiki_bfs.md
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api.routes_documents import _process_document_indexing  # noqa: E402
from app.config import settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.ingestion.parsers import parse_document  # noqa: E402
from app.models.db_models import Document, User  # noqa: E402
from app.utils.ids import new_id  # noqa: E402


def _owner(db) -> User:
    user = db.query(User).filter(User.email == settings.demo_owner_email).first()
    if user is None:
        # password_hash=None: tài khoản này không đăng nhập được.
        user = User(id=new_id(), email=settings.demo_owner_email, full_name="StudyRAG Demo", auth_provider="system", is_verified=True)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    with SessionLocal() as db:
        owner_id = str(_owner(db).id)
        existing = {r[0] for r in db.query(Document.filename).filter(Document.owner_id == owner_id, Document.status == "ready")}

    for path in args.files:
        filename = os.path.basename(path)
        if filename in existing:
            print(f"bỏ qua (đã có): {filename}")
            continue
        with open(path, "rb") as f:
            parsed = parse_document(f.read(), filename)
        doc_id = new_id()
        with SessionLocal() as db:
            db.add(Document(id=doc_id, filename=filename, file_type=parsed.file_type, num_chunks=0, status="processing", owner_id=owner_id))
            db.commit()
        _process_document_indexing(doc_id, owner_id, parsed, filename)
        with SessionLocal() as db:
            doc = db.get(Document, doc_id)
            print(f"{filename}: {doc.status}, {doc.num_chunks} chunk")


if __name__ == "__main__":
    main()
