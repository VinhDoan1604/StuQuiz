"""
app/utils/ids.py

Chuẩn hoá định danh UUID dùng chung (trước đây mỗi module tự viết 1 bản
_safe_uuid/_normalize_uuid/_valid_uuid với namespace khác nhau).
"""
from __future__ import annotations

import uuid


def new_id() -> str:
    return str(uuid.uuid4())


def parse_uuid(value: object) -> str | None:
    """Trả về chuỗi UUID chuẩn hoá, hoặc None nếu `value` không phải UUID hợp lệ."""
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return None
