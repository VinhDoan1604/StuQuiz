"""
app/utils/time.py

Mọi timestamp trong DB là UTC dạng naive (cột DateTime không timezone). Dùng
hàm này thay cho `datetime.utcnow()` (deprecated từ Python 3.12).
"""
from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
