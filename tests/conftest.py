"""Fixture dùng chung: app FastAPI + SQLite in-memory + tạo user nhanh."""
import os

# Phải đặt TRƯỚC khi import app: dùng SQLite tạm, không đụng tới DB/Qdrant thật.
os.environ["APP_ENV"] = "dev"
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["JWT_SECRET"] = "test-only-secret-key-at-least-32-bytes-long"
os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:1")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import app.models.db_models as m  # noqa: E402
from app.core.rate_limit import auth_limiter, chat_limiter  # noqa: E402
from app.core.security import create_token, hash_password  # noqa: E402
from app.db.session import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    auth_limiter.reset()
    chat_limiter.reset()
    yield


@pytest.fixture()
def session_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False)
    engine.dispose()


@pytest.fixture()
def db_session(session_factory):
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(session_factory):
    def override():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    c = TestClient(app)
    c.Session = session_factory
    yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def make_user(client):
    def _make(email, verified=True, password="password123"):
        with client.Session() as db:
            u = m.User(email=email, full_name=email, password_hash=hash_password(password), is_verified=verified)
            db.add(u)
            db.commit()
            db.refresh(u)
            return str(u.id), {"Authorization": f"Bearer {create_token(str(u.id))}"}

    return _make
