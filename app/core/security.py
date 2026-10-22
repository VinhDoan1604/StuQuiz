"""
app/core/security.py

Hạ tầng xác thực dùng chung:
- Password: argon2id (thay SHA-256 + salt cố định).
- Token: JWT HS256 có `exp`, được KIỂM TRA ở mọi route qua `get_current_user`.
- OTP: sinh bằng `secrets`, chỉ lưu HMAC (không lưu mã gốc), so sánh hằng thời gian.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.models.db_models import User

_ph = PasswordHasher()
_bearer = HTTPBearer(auto_error=False)  # tự raise 401 (HTTPBearer mặc định trả 403)

JWT_ALGORITHM = "HS256"


# --------------------------------------------------------------------------
# Password
# --------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    try:
        return _ph.verify(stored_hash, password)
    except Exception:  # sai mật khẩu, hash cũ (SHA-256) không đúng định dạng, ...
        return False


# --------------------------------------------------------------------------
# OTP
# --------------------------------------------------------------------------
def generate_otp() -> str:
    return f"{secrets.randbelow(10**6):06d}"


def hash_otp(email: str, code: str) -> str:
    msg = f"{email.strip().lower()}:{code.strip()}".encode("utf-8")
    return hmac.new(settings.jwt_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def verify_otp_hash(stored_hash: str | None, email: str, code: str) -> bool:
    if not stored_hash:
        return False
    return hmac.compare_digest(stored_hash, hash_otp(email, code))


# --------------------------------------------------------------------------
# JWT
# --------------------------------------------------------------------------
def create_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(days=settings.jwt_expire_days),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGORITHM)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Chưa đăng nhập hoặc phiên đã hết hạn.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None or not creds.credentials:
        raise unauthorized
    try:
        payload = jwt.decode(
            creds.credentials,
            settings.jwt_secret,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError:
        raise unauthorized

    user = db.get(User, payload["sub"])
    if user is None or not user.is_verified:
        raise unauthorized
    return user