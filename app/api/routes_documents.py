from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Header, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
import uuid
import logging

from app.db.session import get_db, SessionLocal
from app.ingestion.parsers import parse_document
from app.ingestion.cleaning import clean_document
from app.chunking.semantic_chunker import chunk_document
from app.embedding.bge_m3_embedder import embed_chunks
from app.indexing.qdrant_store import upsert_chunks
from app.models.db_models import Document, User, StudySpace

router = APIRouter()
logger = logging.getLogger("app.api.documents")


def _resolve_user_id(uid: Optional[str], db: Session) -> Optional[str]:
    if not uid:
        return None
    try:
        val = str(uuid.UUID(str(uid)))
        user = db.query(User).filter(User.id == val).first()
        if user:
            return str(user.id)
    except (ValueError, AttributeError):
        val = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(uid)))
        user = db.query(User).filter(User.id == val).first()
        if user:
            return str(user.id)
    return None


def _process_document_indexing(doc_id: str, parsed, filename: str):
    """
    Xử lý làm sạch, chunking ngữ nghĩa và vector hóa BGE-M3 trong nền.
    Tối ưu hóa:
    - Chỉ vector hóa child chunks (và section độc lập nếu không có con),
      loại bỏ 100% các parent chunk khổng lồ 1500 tokens gây ngốn CPU và nhiễu retrieval.
    - BGE-M3 chạy chế độ inference tối ưu (max_length=512, batch nhỏ, đa luồng CPU tối ưu).
    """
    with SessionLocal() as db:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            logger.error("Không tìm thấy document ID=%s trong database để lập chỉ mục.", doc_id)
            return

        try:
            logger.info(">>> [Task %s] Đang làm sạch tài liệu '%s'...", doc_id, filename)
            cleaned = clean_document(parsed)

            logger.info(">>> [Task %s] Đang phân đoạn cấu trúc ngữ nghĩa (Semantic Chunking)...", doc_id)
            all_chunks = chunk_document(cleaned, doc_id=doc_id)

            # Lọc chỉ lấy child chunks để lập chỉ mục vector (chuẩn xác & nhanh gấp 3-4 lần)
            child_chunks = [c for c in all_chunks if getattr(c, "level", "child") == "child"]
            embeddable_chunks = child_chunks if child_chunks else all_chunks

            for c in embeddable_chunks:
                c.doc_id = doc_id
                if hasattr(c, "metadata") and c.metadata and not c.metadata.source_file:
                    c.metadata.source_file = filename

            logger.info(
                ">>> [Task %s] Chọn %d/%d chunks hiệu dụng để vector hóa (BGE-M3)...",
                doc_id, len(embeddable_chunks), len(all_chunks)
            )

            vectors = embed_chunks([c.text for c in embeddable_chunks])

            logger.info(">>> [Task %s] Upsert %d vectors vào Qdrant Store...", doc_id, len(vectors))
            upsert_chunks(embeddable_chunks, vectors)

            doc.num_chunks = len(embeddable_chunks)
            doc.status = "ready"
            doc.error_message = None
            db.commit()
            logger.info(
                ">>> [Hoàn tất Task %s] Lập chỉ mục thành công tài liệu '%s' với %d chunks!",
                doc_id, filename, len(embeddable_chunks)
            )

        except Exception as exc:
            logger.error(
                ">>> [Task %s] Lỗi khi xử lý lập chỉ mục tài liệu '%s': %s",
                doc_id, filename, exc, exc_info=True
            )
            doc.status = "failed"
            doc.error_message = str(exc)
            db.commit()


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    sync: bool = Form(False),
    db: Session = Depends(get_db),
):
    """
    Tải lên tài liệu.
    - Mặc định: Phân tích nhanh (PyMuPDF < 0.2s), lưu bản ghi ở trạng thái 'processing'
      và trả về NGAY LẬP TỨC (< 1 giây) để giao diện không bị treo / đơ.
      Quá trình nhúng vector BGE-M3 chạy nền mượt mà.
    - sync=true: Xử lý đồng bộ nếu client yêu cầu (chờ xong mới trả về).
    """
    raw_bytes = await file.read()
    effective_user_id = _resolve_user_id(user_id or x_user_id, db)
    logger.info(">>> [Upload] Nhận file: '%s' (%d bytes) từ user=%s. Bắt đầu parse nội dung...", file.filename, len(raw_bytes), effective_user_id)

    # 1. Trích xuất cấu trúc văn bản nhanh (< 0.2s)
    parsed = parse_document(raw_bytes, filename=file.filename)
    doc_id = str(uuid.uuid4())

    # 2. Tạo bản ghi ban đầu trong Postgres
    doc = Document(
        id=doc_id,
        filename=file.filename,
        file_type=parsed.file_type,
        num_chunks=0,
        status="processing",
        owner_id=effective_user_id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    if sync:
        logger.info(">>> [Upload] Yêu cầu xử lý đồng bộ (sync=True)...")
        _process_document_indexing(doc_id, parsed, file.filename)
        db.refresh(doc)
        return {
            "document_id": doc.id,
            "num_chunks": doc.num_chunks,
            "status": doc.status,
            "filename": doc.filename,
        }

    # 3. Đưa công việc nhúng vector nặng vào BackgroundTasks
    background_tasks.add_task(_process_document_indexing, doc_id, parsed, file.filename)

    logger.info(">>> [Upload] Đã tiếp nhận file '%s' (ID=%s). Đang xử lý nền...", file.filename, doc.id)
    return {
        "document_id": doc.id,
        "num_chunks": 0,
        "status": "processing",
        "filename": doc.filename,
        "message": "Tài liệu đã được tải lên thành công và đang được lập chỉ mục trong nền.",
    }


@router.get("/{document_id}/status")
def get_document_status(document_id: str, db: Session = Depends(get_db)):
    """Kiểm tra tiến trình lập chỉ mục của tài liệu."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")

    return {
        "id": str(doc.id),
        "filename": doc.filename,
        "status": getattr(doc, "status", "ready") or "ready",
        "num_chunks": doc.num_chunks or 0,
        "error": getattr(doc, "error_message", None),
        "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
    }


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    user_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    db: Session = Depends(get_db),
):
    """Xóa tài liệu khỏi database và cập nhật các Study Space liên quan."""
    effective_user_id = _resolve_user_id(user_id or x_user_id, db)
    query = db.query(Document).filter(Document.id == document_id)
    if effective_user_id:
        query = query.filter(Document.owner_id == effective_user_id)

    doc = query.first()
    if not doc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu cần xóa.")

    # Xóa tham chiếu document_id trong các StudySpace
    spaces = db.query(StudySpace).all()
    for s in spaces:
        if s.document_ids and document_id in s.document_ids:
            updated_ids = [did for did in s.document_ids if did != document_id]
            s.document_ids = updated_ids

    db.delete(doc)
    db.commit()
    logger.info(">>> Đã xóa thành công tài liệu ID=%s", document_id)
    return {"message": "Đã xóa tài liệu thành công.", "document_id": document_id}


@router.get("")
def list_documents(
    user_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    db: Session = Depends(get_db),
):
    """Trả về danh sách tài liệu riêng của người dùng."""
    effective_user_id = _resolve_user_id(user_id or x_user_id, db)
    query = db.query(Document)
    if effective_user_id:
        query = query.filter(Document.owner_id == effective_user_id)
    else:
        return []

    docs = query.order_by(Document.uploaded_at.desc()).all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "subject": d.subject,
            "num_chunks": d.num_chunks,
            "status": getattr(d, "status", "ready") or "ready",
            "error_message": getattr(d, "error_message", None),
            "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
        }
        for d in docs
    ]
