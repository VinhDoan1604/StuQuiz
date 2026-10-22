"""Fixture dùng chung: app FastAPI + SQLite in-memory + tạo user nhanh."""
import os

os.environ.setdefault("APP_ENV", "dev")
os.environ.setdefault("POSTGRES_PORT", "1")  # fail nhanh -> fallback SQLite khi import app.db.session

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.models.db_models as m
from app.core.security import create_token, hash_password
from app.db.session import Base, get_db
from app.main import app


@pytest.fixture()
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    c = TestClient(app)
    c.Session = Session
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
