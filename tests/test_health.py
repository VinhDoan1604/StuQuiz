from unittest.mock import patch


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert resp.headers.get("X-Request-ID")


def test_request_id_is_propagated(client):
    resp = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert resp.headers["X-Request-ID"] == "abc123"


def test_ready_reports_dependency_status(client):
    with patch("app.indexing.qdrant_store.ping", return_value=True):
        r = client.get("/ready")
    assert r.status_code == 200 and r.json()["checks"] == {"database": "ok", "qdrant": "ok"}

    with patch("app.indexing.qdrant_store.ping", return_value=False):
        r = client.get("/ready")
    assert r.status_code == 503 and r.json()["checks"]["qdrant"] == "error"
