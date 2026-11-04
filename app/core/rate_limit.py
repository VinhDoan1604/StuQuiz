"""
app/core/rate_limit.py

Giới hạn tần suất theo IP, cửa sổ 60s. Hai backend:

- REDIS_URL được đặt  -> bộ đếm dùng chung giữa mọi instance/worker (fixed window
  INCR + EXPIRE, nguyên tử phía Redis). Bắt buộc khi chạy nhiều container.
- Không có REDIS_URL -> đếm trong bộ nhớ tiến trình (sliding window), đủ cho 1 instance.

Redis tạm thời lỗi thì tự lùi về bộ đếm trong bộ nhớ (không chặn người dùng vì
sự cố hạ tầng) và ghi log cảnh báo.

Dùng: `dependencies=[Depends(auth_limiter)]`
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Callable

from fastapi import HTTPException, Request

from app.config import settings

logger = logging.getLogger("app.rate_limit")

_WINDOW_SECONDS = 60


class _MemoryBackend:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int) -> int | None:
        """Ghi nhận 1 request; trả về số giây cần chờ nếu vượt giới hạn, ngược lại None."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > _WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= limit:
                return int(_WINDOW_SECONDS - (now - hits[0])) + 1
            hits.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


class _RedisBackend:
    def __init__(self, url: str) -> None:
        import redis  # import trễ: chỉ cần khi cấu hình REDIS_URL

        self._client = redis.Redis.from_url(url, socket_timeout=1, socket_connect_timeout=1)

    def hit(self, key: str, limit: int) -> int | None:
        window = int(time.time()) // _WINDOW_SECONDS
        redis_key = f"studyrag:rl:{key}:{window}"
        pipe = self._client.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, _WINDOW_SECONDS + 1)
        count, _ = pipe.execute()
        if int(count) > limit:
            return _WINDOW_SECONDS - int(time.time()) % _WINDOW_SECONDS
        return None

    def reset(self) -> None:
        for k in self._client.scan_iter("studyrag:rl:*"):
            self._client.delete(k)


_redis_backend: _RedisBackend | None = None
_redis_lock = threading.Lock()


def _get_redis_backend() -> _RedisBackend | None:
    global _redis_backend
    if not settings.redis_url:
        return None
    if _redis_backend is None:
        with _redis_lock:
            if _redis_backend is None:
                _redis_backend = _RedisBackend(settings.redis_url)
    return _redis_backend


def redis_ping() -> bool:
    """Readiness probe cho Redis (chỉ gọi khi REDIS_URL được đặt)."""
    try:
        backend = _get_redis_backend()
        return bool(backend and backend._client.ping())
    except Exception as exc:
        logger.warning("Redis không phản hồi: %s", exc)
        return False


class RateLimiter:
    def __init__(self, scope: str, limit: Callable[[], int]):
        self.scope = scope
        self._limit = limit
        self._memory = _MemoryBackend()

    def reset(self) -> None:
        self._memory.reset()
        backend = _get_redis_backend()
        if backend is not None:
            backend.reset()

    def _retry_after(self, ip: str, limit: int) -> int | None:
        key = f"{self.scope}:{ip}"
        backend = _get_redis_backend()
        if backend is not None:
            try:
                return backend.hit(key, limit)
            except Exception as exc:
                logger.warning("Redis rate limit lỗi (%s) — tạm dùng bộ đếm trong bộ nhớ.", exc)
        return self._memory.hit(key, limit)

    def __call__(self, request: Request) -> None:
        if not settings.rate_limit_enabled:
            return
        ip = request.client.host if request.client else "unknown"
        retry_after = self._retry_after(ip, self._limit())
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Bạn thao tác quá nhanh, vui lòng thử lại sau ít phút.",
                headers={"Retry-After": str(retry_after)},
            )


auth_limiter = RateLimiter("auth", lambda: settings.rate_limit_auth_per_minute)
chat_limiter = RateLimiter("chat", lambda: settings.rate_limit_chat_per_minute)
