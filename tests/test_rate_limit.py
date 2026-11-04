"""Rate limit: bộ đếm Redis dùng chung giữa các instance + lùi về bộ nhớ khi Redis lỗi."""
from types import SimpleNamespace

import fakeredis
import pytest
from fastapi import HTTPException

from app.core import rate_limit
from app.core.rate_limit import RateLimiter, _RedisBackend


def _request(ip="1.2.3.4"):
    return SimpleNamespace(client=SimpleNamespace(host=ip))


@pytest.fixture()
def shared_redis(monkeypatch):
    backend = _RedisBackend.__new__(_RedisBackend)
    backend._client = fakeredis.FakeRedis()
    monkeypatch.setattr(rate_limit, "_get_redis_backend", lambda: backend)
    return backend


def test_redis_counter_is_shared_across_instances(shared_redis):
    # 2 RateLimiter = 2 tiến trình khác nhau (bộ nhớ riêng), cùng 1 Redis
    instance_a = RateLimiter("auth", lambda: 3)
    instance_b = RateLimiter("auth", lambda: 3)
    instance_a(_request())
    instance_b(_request())
    instance_a(_request())
    with pytest.raises(HTTPException) as exc:
        instance_b(_request())
    assert exc.value.status_code == 429
    assert 1 <= int(exc.value.headers["Retry-After"]) <= 60
    instance_b(_request("5.6.7.8"))  # IP khác không bị ảnh hưởng


def test_falls_back_to_memory_when_redis_down(monkeypatch):
    class Broken:
        def hit(self, *_):
            raise ConnectionError("redis down")

    monkeypatch.setattr(rate_limit, "_get_redis_backend", lambda: Broken())
    limiter = RateLimiter("chat", lambda: 2)
    limiter(_request())
    limiter(_request())
    with pytest.raises(HTTPException):
        limiter(_request())


def test_ready_includes_redis_when_configured(client, monkeypatch, shared_redis):
    from unittest.mock import patch

    monkeypatch.setattr(rate_limit.settings, "redis_url", "redis://fake")
    with patch("app.indexing.qdrant_store.ping", return_value=True):
        r = client.get("/ready")
    assert r.status_code == 200 and r.json()["checks"]["redis"] == "ok"
