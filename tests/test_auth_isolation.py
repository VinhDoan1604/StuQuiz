"""
Test bảo mật: xác thực + cô lập dữ liệu giữa các user.
Chạy: pytest tests/test_auth_isolation.py -v

Lưu ý: import app.db.session sẽ thử kết nối Postgres (timeout 2s) rồi fallback
SQLite file; các test dưới đây override get_db sang SQLite in-memory nên không
đụng tới dữ liệu thật.
"""
import os
import uuid

os.environ.setdefault("APP_ENV", "dev")
os.environ.setdefault("POSTGRES_PORT", "1")  # fail nhanh -> fallback SQLite

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.models.db_models as m  # noqa: F401  (đăng ký bảng vào Base.metadata)
from app.core.security import create_token, hash_password
from app.db.session import Base, get_db
from app.main import app


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
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


def make_user(client, email, verified=True, password="password123"):
    with client.Session() as db:
        user = m.User(
            email=email,
            full_name=email,
            password_hash=hash_password(password),
            is_verified=verified,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return str(user.id), {"Authorization": f"Bearer {create_token(str(user.id))}"}


# ---------------------------------------------------------------- xác thực
def test_requires_token(client):
    assert client.get("/spaces").status_code == 401
    assert client.get("/documents").status_code == 401
    assert client.post("/chat", json={"session_id": "s", "message": "hi"}).status_code == 401


def test_rejects_forged_token(client):
    r = client.get("/spaces", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_google_login_cannot_be_spoofed_with_email_only(client):
    # Lỗ hổng cũ: chỉ cần gửi {"email": "victim@x.com"} là được cấp token
    r = client.post("/auth/google", json={"email": "victim@x.com", "google_id": "1"})
    assert r.status_code in (400, 503)
    assert "token" not in r.json()


def test_login_wrong_password(client):
    make_user(client, "a@x.com")
    r = client.post("/auth/login", json={"email": "a@x.com", "password": "wrong-pass"})
    assert r.status_code == 401


def test_otp_lockout_after_max_attempts(client):
    r = client.post("/auth/register", json={"email": "n@x.com", "password": "password123"})
    assert r.status_code == 200
    real_code = r.json()["dev_code"]
    wrong = "000000" if real_code != "000000" else "111111"

    for _ in range(5):
        assert client.post("/auth/verify-otp", json={"email": "n@x.com", "code": wrong}).status_code == 400
    # Đã khoá: kể cả nhập ĐÚNG mã cũng bị từ chối
    r = client.post("/auth/verify-otp", json={"email": "n@x.com", "code": real_code})
    assert r.status_code == 429


def test_otp_happy_path_returns_token(client):
    r = client.post("/auth/register", json={"email": "ok@x.com", "password": "password123"})
    code = r.json()["dev_code"]
    r = client.post("/auth/verify-otp", json={"email": "ok@x.com", "code": code})
    assert r.status_code == 200 and r.json()["token"]


def test_short_password_rejected(client):
    r = client.post("/auth/register", json={"email": "s@x.com", "password": "123"})
    assert r.status_code == 400


# ------------------------------------------------------------ cô lập dữ liệu
def test_user_cannot_touch_other_users_space(client):
    _, ha = make_user(client, "a@x.com")
    _, hb = make_user(client, "b@x.com")

    sid = client.post("/spaces", json={"title": "Của A"}, headers=ha).json()["id"]

    assert client.get(f"/spaces/{sid}", headers=hb).status_code == 404
    assert client.patch(f"/spaces/{sid}", json={"title": "hack"}, headers=hb).status_code == 404
    assert client.delete(f"/spaces/{sid}", headers=hb).status_code == 404
    assert client.get(f"/spaces/{sid}/chat-history", headers=hb).status_code == 404
    assert client.get(f"/spaces/{sid}/quiz-history", headers=hb).status_code == 404
    assert client.get("/spaces", headers=hb).json() == []
    # A vẫn thấy nguyên vẹn
    assert client.get(f"/spaces/{sid}", headers=ha).status_code == 200


def test_cannot_attach_other_users_document(client):
    a_id, _ = make_user(client, "a@x.com")
    _, hb = make_user(client, "b@x.com")
    with client.Session() as db:
        doc = m.Document(id=str(uuid.uuid4()), owner_id=a_id, filename="a.pdf", status="ready")
        db.add(doc)
        db.commit()
        doc_id = str(doc.id)

    sid = client.post("/spaces", json={"title": "Của B"}, headers=hb).json()["id"]
    r = client.post(f"/spaces/{sid}/documents", json={"document_id": doc_id}, headers=hb)
    assert r.status_code == 404
    assert client.delete(f"/documents/{doc_id}", headers=hb).status_code == 404
    assert client.get(f"/documents/{doc_id}/status", headers=hb).status_code == 404
    assert client.get("/documents", headers=hb).json() == []


def test_quiz_rejects_other_users_documents(client):
    a_id, _ = make_user(client, "a@x.com")
    _, hb = make_user(client, "b@x.com")
    with client.Session() as db:
        doc = m.Document(id=str(uuid.uuid4()), owner_id=a_id, filename="a.pdf", status="ready")
        db.add(doc)
        db.commit()
        doc_id = str(doc.id)

    r = client.post("/quiz/generate", json={"document_ids": [doc_id]}, headers=hb)
    assert r.status_code == 404


def test_cannot_hijack_other_users_chat_session(client):
    a_id, _ = make_user(client, "a@x.com")
    _, hb = make_user(client, "b@x.com")
    sess_id = str(uuid.uuid4())
    with client.Session() as db:
        db.add(m.ChatSession(id=sess_id, user_id=a_id))
        db.commit()

    r = client.post("/chat", json={"session_id": sess_id, "message": "xin chào"}, headers=hb)
    assert r.status_code == 404


def test_vector_search_requires_document_ids():
    from app.indexing.qdrant_store import hybrid_search

    empty = {"indices": [], "values": []}
    assert hybrid_search([0.0] * 1024, empty, document_ids=None) == []
    assert hybrid_search([0.0] * 1024, empty, document_ids=[]) == []