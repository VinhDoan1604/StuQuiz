from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_auth, routes_chat, routes_documents, routes_mastery, routes_quiz, routes_spaces
from app.config import settings
from app.core.logging_config import setup_logging

setup_logging()

app = FastAPI(title="StudyRAG", version="0.2.0")

# Xác thực bằng header Authorization (không dùng cookie) => không cần allow_credentials.
# Origin lấy từ CORS_ORIGINS trong .env thay vì "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(routes_auth.router, prefix="/auth", tags=["auth"])
app.include_router(routes_spaces.router, prefix="/spaces", tags=["spaces"])
app.include_router(routes_documents.router, prefix="/documents", tags=["documents"])
app.include_router(routes_chat.router, prefix="/chat", tags=["chat"])
app.include_router(routes_quiz.router, prefix="/quiz", tags=["quiz"])
app.include_router(routes_mastery.router, prefix="/mastery", tags=["mastery"])


@app.get("/health")
def health():
    return {"status": "ok"}
