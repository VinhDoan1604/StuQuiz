"""Hành vi production: rate limit, phạm vi tài liệu của chat, giới hạn input, khôi phục tác vụ nền."""
import json
import uuid
from unittest.mock import AsyncMock, patch

import app.models.db_models as m
from app.api import routes_documents
from app.config import settings


def _add_doc(client, owner_id, status="ready"):
    with client.Session() as db:
        doc = m.Document(id=str(uuid.uuid4()), owner_id=owner_id, filename="a.pdf", status=status)
        db.add(doc)
        db.commit()
        return str(doc.id)


# ------------------------------------------------------------ rate limiting
def test_auth_endpoints_are_rate_limited(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_auth_per_minute", 3)
    body = {"email": "x@x.com", "password": "wrong-pass"}
    codes = [client.post("/auth/login", json=body).status_code for _ in range(4)]
    assert codes[:3] == [401, 401, 401]
    assert codes[3] == 429


# ------------------------------------------------------- phạm vi chat/space
def test_chat_in_empty_space_searches_no_documents(client, make_user):
    uid, h = make_user("a@x.com")
    _add_doc(client, uid)  # tài liệu của user nhưng KHÔNG thuộc space
    sid = client.post("/spaces", json={"title": "Trống"}, headers=h).json()["id"]

    fake = AsyncMock(return_value={"intent": "qa", "answer": "ok", "citations": [], "reflection": {}})
    with patch("app.api.routes_chat.run_graph", fake):
        r = client.post("/chat", json={"session_id": str(uuid.uuid4()), "message": "hi", "space_id": sid}, headers=h)
    assert r.status_code == 200
    assert fake.call_args.kwargs["document_ids"] == []


def test_chat_in_space_uses_only_space_documents(client, make_user):
    uid, h = make_user("a@x.com")
    in_space = _add_doc(client, uid)
    _add_doc(client, uid)
    sid = client.post("/spaces", json={"title": "S"}, headers=h).json()["id"]
    client.post(f"/spaces/{sid}/documents", json={"document_id": in_space}, headers=h)

    fake = AsyncMock(return_value={"intent": "qa", "answer": "ok", "citations": [], "reflection": {}})
    with patch("app.api.routes_chat.run_graph", fake):
        client.post("/chat", json={"session_id": str(uuid.uuid4()), "message": "hi", "space_id": sid}, headers=h)
    assert fake.call_args.kwargs["document_ids"] == [in_space]
    assert fake.call_args.kwargs["user_id"] == uid


def test_chat_rejects_overlong_message(client, make_user, monkeypatch):
    monkeypatch.setattr(settings, "max_chat_message_chars", 10)
    _, h = make_user("a@x.com")
    r = client.post("/chat", json={"session_id": str(uuid.uuid4()), "message": "x" * 11}, headers=h)
    assert r.status_code == 413


def test_stream_error_does_not_leak_internal_details(client, make_user):
    _, h = make_user("a@x.com")

    with patch("app.graph.build_graph._prepare_state", side_effect=RuntimeError("postgres://secret-password@db")):
        r = client.post("/chat/stream", json={"session_id": str(uuid.uuid4()), "message": "hi"}, headers=h)
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["event"] == "error"
    assert "secret" not in events[-1]["message"]


# --------------------------------------------------------- tài liệu kẹt nền
def test_stale_processing_documents_marked_failed(client, make_user):
    uid, _ = make_user("a@x.com")
    stuck = _add_doc(client, uid, status="processing")
    ok = _add_doc(client, uid, status="ready")

    with patch.object(routes_documents, "SessionLocal", client.Session):
        assert routes_documents.fail_stale_processing_documents() == 1

    with client.Session() as db:
        assert db.get(m.Document, stuck).status == "failed"
        assert db.get(m.Document, ok).status == "ready"


def test_quiz_generate_validates_num_questions(client, make_user):
    uid, h = make_user("a@x.com")
    doc = _add_doc(client, uid)
    r = client.post("/quiz/generate", json={"document_ids": [doc], "num_questions": 500}, headers=h)
    assert r.status_code == 422
