from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging_config import setup_logging
from app.api import routes_chat, routes_documents, routes_quiz, routes_mastery, routes_auth, routes_spaces

setup_logging()

app = FastAPI(title="StudyRAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
