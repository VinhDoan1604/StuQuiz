"""routes_documents v2 (thay thế bản Part 1): hỗ trợ txt/md, lưu parent chunk, dọn parent khi xoá."""
import asyncio
import logging
import os
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.chunking.semantic_chunker import chunk_document
from app.config import settings
from app.core.ownership import get_owned_document
from app.core.security import get_current_user
from app.db.session import SessionLocal, get_db
from app.embedding.bge_m3_embedder import embed_chunks
from app.indexing.qdrant_store import delete_document_vectors, upsert_chunks
from app.ingestion.cleaning import clean_document
from app.ingestion.parsers import ParsingError, UnsupportedFileTypeError, parse_document
from app.models.db_models import Document, ParentChunk, StudySpace, User

router = APIRouter()
logger = logging.getLogger("app.api.documents")

ALLOWED_EXTENSIONS = {"pdf", "docx", "pptx", "txt", "md"}


def _process_document_indexing(doc_id: str, owner_id: str, parsed, filename: str):
    """Làm sạch -> chunking -> lưu parent (Postgres) -> embed child (BGE-M3) -> upsert Qdrant."""
    with SessionLocal() as db:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            logger.error("Không tìm thấy document ID=%s để lập chỉ mục.", doc_id)
            return

        try:
            cleaned = clean_document(parsed)
            all_chunks = chunk_document(cleaned, doc_id=doc_id)

            children = [c for c in all_chunks if c.level == "child"]
            embeddable = children or all_chunks
            for c in embeddable:
                c.doc_id = doc_id
                if c.metadata and not c.metadata.source_file:
                    c.metadata.source_file = filename

            # Parent chỉ có ý nghĩa khi child tồn tại (child trỏ tới parent qua parent_id)
            parents = [c for c in all_chunks if c.level == "parent"] if children else []
            db.query(ParentChunk).filter(ParentChunk.doc_id == doc_id).delete(synchronize_session=False)
            db.add_all(
                ParentChunk(
                    id=p.id, doc_id=doc_id, text=p.text, section=p.metadata.section,
                    page=p.metadata.page, token_count=p.token_count,
                )
                for p in parents
            )

            vectors = embed_chunks([c.text for c in embeddable])
            upsert_chunks(embeddable, vectors, owner_id=owner_id)

            doc.num_chunks = len(embeddable)
            doc.status = "ready"
            doc.error_message = None
            db.commit()  # parent + trạng thái cùng 1 transaction
            logger.info("[Task %s] '%s': %d child, %d parent.", doc_id, filename, len(embeddable), len(parents))
        except Exception as exc:
            logger.error("[Task %s] Lỗi lập chỉ mục '%s': %s", doc_id, filename, exc, exc_info=True)
            db.rollback()  # bỏ parent dở dang
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.status = "failed"
                doc.error_message = str(exc)[:500]
                db.commit()


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sync: bool = Form(False),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    filename = os.path.basename(file.filename or "").strip()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Định dạng '.{ext}' chưa được hỗ trợ. Hỗ trợ: {sorted(ALLOWED_EXTENSIONS)}.",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(status_code=413, detail=f"Tệp vượt quá {settings.max_upload_mb}MB.")
    raw_bytes = await file.read()
    if len(raw_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Tệp vượt quá {settings.max_upload_mb}MB.")
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Tệp rỗng.")

    try:
        parsed = await asyncio.to_thread(parse_document, raw_bytes, filename)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except ParsingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not parsed.blocks:
        raise HTTPException(status_code=422, detail="Không trích xuất được nội dung nào từ tệp này.")

    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id, filename=filename, file_type=parsed.file_type,
        num_chunks=0, status="processing", owner_id=user.id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    if sync:
        await asyncio.to_thread(_process_document_indexing, doc_id, str(user.id), parsed, filename)
        db.refresh(doc)
        return {"document_id": doc.id, "num_chunks": doc.num_chunks, "status": doc.status, "filename": doc.filename}

    background_tasks.add_task(_process_document_indexing, doc_id, str(user.id), parsed, filename)
    return {
        "document_id": doc.id, "num_chunks": 0, "status": "processing", "filename": doc.filename,
        "message": "Tài liệu đã được tải lên và đang được lập chỉ mục trong nền.",
    }


@router.get("/{document_id}/status")
def get_document_status(document_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    doc = get_owned_document(db, user, document_id)
    return {
        "id": str(doc.id), "filename": doc.filename, "status": doc.status or "ready",
        "num_chunks": doc.num_chunks or 0, "error": doc.error_message,
        "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
    }


@router.delete("/{document_id}")
def delete_document(document_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    doc = get_owned_document(db, user, document_id)
    doc_id = str(doc.id)

    for s in db.query(StudySpace).filter(StudySpace.user_id == user.id).all():
        if s.document_ids and doc_id in s.document_ids:
            s.document_ids = [d for d in s.document_ids if d != doc_id]

    db.query(ParentChunk).filter(ParentChunk.doc_id == doc_id).delete(synchronize_session=False)
    db.delete(doc)
    db.commit()

    try:
        delete_document_vectors(doc_id)
    except Exception as exc:
        logger.error("Không xoá được vector của document %s: %s", doc_id, exc)

    return {"message": "Đã xóa tài liệu.", "document_id": doc_id}


@router.get("")
def list_documents(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.owner_id == user.id).order_by(Document.uploaded_at.desc()).all()
    return [
        {
            "id": str(d.id), "filename": d.filename, "file_type": d.file_type, "subject": d.subject,
            "num_chunks": d.num_chunks, "status": d.status or "ready", "error_message": d.error_message,
            "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
        }
        for d in docs
    ]
