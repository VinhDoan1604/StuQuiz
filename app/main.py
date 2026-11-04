"""
FastAPI entrypoint: middleware (CORS, request-id + access log), xử lý lỗi
chung, lifecycle (khôi phục tài liệu kẹt "processing", tải sẵn model) và
health/readiness probe.
"""
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api import routes_auth, routes_chat, routes_documents, routes_mastery, routes_quiz, routes_spaces
from app.config import APP_VERSION, settings
from app.core import rate_limit
from app.core.logging_config import request_id_var, setup_logging
from app.db.session import SessionLocal

setup_logging()
logger = logging.getLogger("app.main")


def _preload_models() -> None:
    from app.embedding.bge_m3_embedder import get_model
    from app.retrieval.reranker import warmup as warmup_reranker

    get_model()
    warmup_reranker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    recovered = await asyncio.to_thread(routes_documents.fail_stale_processing_documents)
    if recovered:
        logger.warning("Đánh dấu %d tài liệu kẹt ở trạng thái 'processing' thành 'failed'.", recovered)
    if settings.preload_models:
        logger.info("Đang tải sẵn BGE-M3 + reranker...")
        await asyncio.to_thread(_preload_models)
    logger.info("StudyRAG %s khởi động (APP_ENV=%s).", APP_VERSION, settings.app_env)
    yield


app = FastAPI(
    title="StudyRAG",
    version=APP_VERSION,
    lifespan=lifespan,
    # Ẩn Swagger ngoài môi trường dev
    docs_url="/docs" if settings.is_dev else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.is_dev else None,
)

# Xác thực bằng header Authorization (không dùng cookie) => không cần allow_credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    token = request_id_var.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    if request.url.path not in ("/health", "/ready"):
        logger.info(
            "%s %s -> %d (%.0fms) [%s]",
            request.method, request.url.path, response.status_code,
            (time.perf_counter() - start) * 1000, rid,
        )
    return response


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.exception("Lỗi không xử lý được tại %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Lỗi máy chủ nội bộ. Vui lòng thử lại sau."})


app.include_router(routes_auth.router, prefix="/auth", tags=["auth"])
app.include_router(routes_spaces.router, prefix="/spaces", tags=["spaces"])
app.include_router(routes_documents.router, prefix="/documents", tags=["documents"])
app.include_router(routes_chat.router, prefix="/chat", tags=["chat"])
app.include_router(routes_quiz.router, prefix="/quiz", tags=["quiz"])
app.include_router(routes_mastery.router, prefix="/mastery", tags=["mastery"])


@app.get("/health", tags=["ops"])
def health():
    """Liveness: tiến trình còn sống."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready():
    """Readiness: kiểm tra kết nối DB và Qdrant."""
    from app.indexing.qdrant_store import ping as qdrant_ping

    checks: dict[str, str] = {}
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        logger.warning("Readiness: DB lỗi: %s", exc)
        checks["database"] = "error"
    checks["qdrant"] = "ok" if qdrant_ping() else "error"
    if settings.redis_url:
        checks["redis"] = "ok" if rate_limit.redis_ping() else "error"

    healthy = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "degraded", "version": APP_VERSION, "checks": checks},
    )
