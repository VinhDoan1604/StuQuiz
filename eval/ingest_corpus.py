"""
eval/ingest_corpus.py

Nạp corpus eval vào Qdrant qua ĐÚNG pipeline production (parse -> clean -> chunk
-> BGE-M3 -> upsert), không cần Postgres/đăng nhập.

    python -m eval.ingest_corpus eval/corpus --out eval/data/corpus_manifest.json

- doc_id = uuid5(sha256 nội dung) => nạp lại cùng file cho cùng doc_id (idempotent);
  vector cũ của doc đó được xoá trước khi nạp lại.
- owner_id="eval": tách biệt hoàn toàn với dữ liệu user thật.
- Manifest ghi lại doc_id, hash, số chunk và tham số chunking để tái lập kết quả.
- Corpus nên là tài liệu CÔNG KHAI (giáo trình mở, slide có giấy phép) để người
  khác chạy lại được benchmark của bạn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

from app.chunking import semantic_chunker as sc
from app.chunking.semantic_chunker import chunk_document
from app.embedding.bge_m3_embedder import embed_chunks
from app.indexing.qdrant_store import delete_document_vectors, upsert_chunks
from app.ingestion.cleaning import clean_document
from app.ingestion.parsers import parse_document

from app.db.session import Base, SessionLocal, engine
from app.models.db_models import ParentChunk

SUPPORTED = {".pdf", ".docx", ".pptx", ".txt", ".md"}
EVAL_OWNER = "eval"


def main() -> None:
    Base.metadata.create_all(bind=engine)
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir")
    ap.add_argument("--out", default="eval/data/corpus_manifest.json")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.corpus_dir).iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        raise SystemExit(f"Không có file {sorted(SUPPORTED)} trong {args.corpus_dir}")

    docs = []
    with SessionLocal() as db:
        for path in files:
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"studyrag-eval:{sha}"))

            parsed = parse_document(raw, path.name)
            chunks = chunk_document(clean_document(parsed), doc_id=doc_id)
            children = [c for c in chunks if c.level == "child"] or chunks
            for c in children:
                c.doc_id = doc_id

            parents = [c for c in chunks if c.level == "parent"] if [c for c in chunks if c.level == "child"] else []
            db.query(ParentChunk).filter(ParentChunk.doc_id == doc_id).delete(synchronize_session=False)
            db.add_all(
                ParentChunk(
                    id=p.id, doc_id=doc_id, text=p.text, section=p.metadata.section,
                    page=p.metadata.page, token_count=p.token_count,
                )
                for p in parents
            )
            db.commit()

            delete_document_vectors(doc_id)
            upsert_chunks(children, embed_chunks([c.text for c in children]), owner_id=EVAL_OWNER)

            docs.append({"filename": path.name, "doc_id": doc_id, "sha256": sha, "num_chunks": len(children)})
            print(f"[ingest] {path.name}: {len(children)} child chunks, {len(parents)} parent chunks (doc_id={doc_id[:8]}…)")

    manifest = {
        "embedding_model": "BAAI/bge-m3",
        "chunking": {
            "child_min_tokens": sc.CHILD_MIN_TOKENS,
            "child_max_tokens": sc.CHILD_MAX_TOKENS,
            "child_overlap_tokens": sc.CHILD_OVERLAP_TOKENS,
            "parent_max_tokens": sc.PARENT_MAX_TOKENS,
        },
        "documents": docs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ingest] Xong {len(docs)} tài liệu -> {args.out}")


if __name__ == "__main__":
    main()