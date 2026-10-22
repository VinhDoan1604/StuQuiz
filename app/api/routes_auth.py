"""
app/api/routes_auth.py

Nguyên tắc: server CHỈ tin danh tính đã được xác minh bằng mật khẩu (argon2),
OTP (hash + giới hạn số lần thử + hết hạn) hoặc token do Google ký/xác nhận.
Không bao giờ tin email/google_id do client tự khai trong body.
"""
import json
import logging
import re
import secrets
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.core.security import (
    create_token,
    generate_otp,
    get_current_user,
    hash_otp,
    hash_password,
    verify_otp_hash,
    verify_password,
)
from app.db.session import get_db
from app.models.db_models import User
from app.utils.mailer import send_verification_email

router = APIRouter()
logger = logging.getLogger("app.api.auth")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MIN_PASSWORD_LEN = 8
# Hash giả để login tốn thời gian tương đương dù email không tồn tại (chống dò email qua timing)
_DUMMY_HASH = hash_password("dummy-password-for-timing")


# ===========================================================================
# Schemas
# ===========================================================================
class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class VerifyOtpRequest(BaseModel):
    email: str
    code: str


class ResendOtpRequest(BaseModel):
    email: str


class GoogleAuthRequest(BaseModel):
    credential: Optional[str] = None     # Google ID token (JWT)
    access_token: Optional[str] = None   # Google OAuth2 access token
    code: Optional[str] = None           # Authorization code
    redirect_uri: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    auth_provider: Optional[str] = "local"
    is_verified: bool = False
    created_at: Optional[str] = None


class AuthResultResponse(BaseModel):
    status: str = "success"  # "success" | "needs_verification"
    message: str = ""
    email: Optional[str] = None
    user: Optional[UserResponse] = None
    token: Optional[str] = None
    dev_code: Optional[str] = None  # CHỈ có giá trị khi APP_ENV=dev


# ===========================================================================
# Helpers
# ===========================================================================
def _clean_email(raw: str) -> str:
    email = (raw or "").strip().lower()
    if not _EMAIL_RE.match(email) or len(email) > 254:
        raise HTTPException(status_code=400, detail="Địa chỉ email không hợp lệ.")
    return email


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        auth_provider=user.auth_provider,
        is_verified=bool(user.is_verified),
        created_at=user.created_at.isoformat() if user.created_at else None,
    )


def _auth_success(user: User, message: str) -> AuthResultResponse:
    return AuthResultResponse(
        status="success",
        message=message,
        email=user.email,
        user=_user_response(user),
        token=create_token(str(user.id)),
    )


def _otp_response(email: str, code: Optional[str], message: str) -> AuthResultResponse:
    return AuthResultResponse(
        status="needs_verification",
        message=message,
        email=email,
        dev_code=code if settings.app_env == "dev" else None,
    )


def _seconds_since_otp_sent(user: User) -> Optional[float]:
    """Suy ra thời điểm gửi từ hạn dùng (expires_at - TTL) để khỏi thêm cột DB."""
    if not user.verification_code_expires_at:
        return None
    sent_at = user.verification_code_expires_at - timedelta(minutes=settings.otp_ttl_minutes)
    return (datetime.utcnow() - sent_at).total_seconds()


def _enforce_resend_cooldown(user: User) -> None:
    elapsed = _seconds_since_otp_sent(user)
    if elapsed is not None and elapsed < settings.otp_resend_cooldown_seconds:
        wait = int(settings.otp_resend_cooldown_seconds - elapsed) + 1
        raise HTTPException(
            status_code=429, detail=f"Vui lòng đợi {wait}s trước khi yêu cầu mã mới."
        )


def _issue_otp(db: Session, user: User) -> str:
    code = generate_otp()
    user.verification_code = hash_otp(user.email, code)  # chỉ lưu HMAC
    user.verification_code_expires_at = datetime.utcnow() + timedelta(minutes=settings.otp_ttl_minutes)
    user.verification_attempts = 0
    db.commit()
    return code


# ===========================================================================
# Local auth
# ===========================================================================
@router.post("/register", response_model=AuthResultResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    email = _clean_email(req.email)
    if len(req.password) < _MIN_PASSWORD_LEN:
        raise HTTPException(
            status_code=400, detail=f"Mật khẩu tối thiểu {_MIN_PASSWORD_LEN} ký tự."
        )

    full_name = (req.full_name or "").strip()[:100] or email.split("@")[0]
    user = db.query(User).filter(User.email == email).first()

    if user:
        if user.is_verified:
            raise HTTPException(
                status_code=409,
                detail="Email này đã được đăng ký. Vui lòng đăng nhập.",
            )
        _enforce_resend_cooldown(user)
        user.password_hash = hash_password(req.password)
        user.full_name = full_name
    else:
        user = User(
            email=email,
            full_name=full_name,
            password_hash=hash_password(req.password),
            auth_provider="local",
            is_verified=False,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    code = _issue_otp(db, user)
    send_verification_email(email, code)
    return _otp_response(email, code, "Vui lòng nhập mã xác thực 6 số đã gửi tới email của bạn.")


@router.post("/verify-otp", response_model=AuthResultResponse)
def verify_otp(req: VerifyOtpRequest, db: Session = Depends(get_db)):
    email = _clean_email(req.email)
    invalid = HTTPException(status_code=400, detail="Mã xác thực không hợp lệ hoặc đã hết hạn.")

    user = db.query(User).filter(User.email == email).first()
    if not user or not user.verification_code:
        raise invalid

    if (user.verification_attempts or 0) >= settings.otp_max_attempts:
        raise HTTPException(
            status_code=429, detail="Nhập sai quá nhiều lần. Vui lòng yêu cầu mã mới."
        )
    if user.verification_code_expires_at and datetime.utcnow() > user.verification_code_expires_at:
        raise invalid

    if not verify_otp_hash(user.verification_code, email, req.code):
        user.verification_attempts = (user.verification_attempts or 0) + 1
        db.commit()
        raise invalid

    user.is_verified = True
    user.verification_code = None
    user.verification_code_expires_at = None
    user.verification_attempts = 0
    db.commit()
    db.refresh(user)
    return _auth_success(user, "Xác thực tài khoản thành công!")


@router.post("/resend-otp", response_model=AuthResultResponse)
def resend_otp(req: ResendOtpRequest, db: Session = Depends(get_db)):
    email = _clean_email(req.email)
    generic = "Nếu tài khoản tồn tại và chưa xác thực, mã mới đã được gửi."
    user = db.query(User).filter(User.email == email).first()
    if not user or user.is_verified:
        return _otp_response(email, None, generic)  # không lộ email có tồn tại hay không

    _enforce_resend_cooldown(user)
    code = _issue_otp(db, user)
    send_verification_email(email, code)
    return _otp_response(email, code, generic)


@router.post("/login", response_model=AuthResultResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    email = (req.email or "").strip().lower()
    user = db.query(User).filter(User.email == email).first()

    has_password = bool(user and user.password_hash)
    password_ok = verify_password(user.password_hash if has_password else _DUMMY_HASH, req.password)
    if not (user and has_password and password_ok):
        raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không chính xác.")

    if not user.is_verified:
        elapsed = _seconds_since_otp_sent(user)
        if elapsed is None or elapsed >= settings.otp_resend_cooldown_seconds:
            code = _issue_otp(db, user)
            send_verification_email(email, code)
            return _otp_response(email, code, "Tài khoản chưa xác thực. Vui lòng nhập mã OTP.")
        return _otp_response(email, None, "Mã OTP đã được gửi gần đây, vui lòng kiểm tra email.")

    return _auth_success(user, "Đăng nhập thành công!")


# ===========================================================================
# Demo (mỗi lần bấm = 1 user tạm thời riêng, KHÔNG dùng chung dữ liệu)
# ===========================================================================
@router.post("/demo", response_model=AuthResultResponse)
def demo_login(db: Session = Depends(get_db)):
    if not settings.demo_enabled:
        raise HTTPException(status_code=404, detail="Not found")
    suffix = secrets.token_hex(4)
    user = User(
        email=f"demo-{suffix}@demo.studyrag.local",
        full_name="Học viên dùng thử",
        auth_provider="demo",
        is_verified=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _auth_success(user, "Đã tạo phiên dùng thử.")


# ===========================================================================
# Google OAuth — chỉ tin thông tin do Google xác nhận
# ===========================================================================
def _http_json(url: str, data: Optional[bytes] = None, headers: Optional[dict] = None) -> dict[str, Any]:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("Gọi Google thất bại (%s): %s", url.split("?")[0], exc)
        raise HTTPException(status_code=401, detail="Không xác thực được với Google.")


def _verify_google_id_token(token: str, client_id: str) -> tuple[str, str, Optional[str]]:
    from google.auth.transport import requests as g_requests
    from google.oauth2 import id_token as g_id_token

    try:
        info = g_id_token.verify_oauth2_token(token, g_requests.Request(), client_id)
    except Exception as exc:
        logger.warning("ID token Google không hợp lệ: %s", exc)
        raise HTTPException(status_code=401, detail="Token Google không hợp lệ.")
    if not info.get("email_verified"):
        raise HTTPException(status_code=401, detail="Email Google chưa được xác minh.")
    return info["email"], info["sub"], info.get("name")


def _verify_google_access_token(access_token: str, client_id: str) -> tuple[str, str, Optional[str]]:
    # Phải kiểm tra `aud`: token do app KHÁC xin cho cùng user sẽ không được chấp nhận
    tokeninfo = _http_json(
        "https://oauth2.googleapis.com/tokeninfo?" + urllib.parse.urlencode({"access_token": access_token})
    )
    if client_id not in (tokeninfo.get("aud"), tokeninfo.get("azp")):
        raise HTTPException(status_code=401, detail="Token Google không dành cho ứng dụng này.")

    info = _http_json(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if not info.get("email") or not info.get("email_verified"):
        raise HTTPException(status_code=401, detail="Email Google chưa được xác minh.")
    return info["email"], info["sub"], info.get("name")


def _exchange_google_code(code: str, redirect_uri: Optional[str]) -> str:
    resp = _http_json(
        "https://oauth2.googleapis.com/token",
        data=urllib.parse.urlencode(
            {
                "code": code,
                "client_id": settings.google_oauth_client_id,
                "client_secret": settings.google_oauth_client_secret,
                "redirect_uri": redirect_uri or "http://localhost:5173",
                "grant_type": "authorization_code",
            }
        ).encode("utf-8"),
    )
    id_token = resp.get("id_token")
    if not id_token:
        raise HTTPException(status_code=401, detail="Google không trả về ID token.")
    return id_token


@router.get("/google/config")
def get_google_config():
    return {"client_id": settings.google_oauth_client_id}


@router.post("/google", response_model=AuthResultResponse)
def google_auth(req: GoogleAuthRequest, db: Session = Depends(get_db)):
    client_id = settings.google_oauth_client_id
    if not client_id:
        raise HTTPException(status_code=503, detail="Google OAuth chưa được cấu hình trên máy chủ.")

    if req.code:
        email, google_id, name = _verify_google_id_token(
            _exchange_google_code(req.code, req.redirect_uri), client_id
        )
    elif req.credential:
        email, google_id, name = _verify_google_id_token(req.credential, client_id)
    elif req.access_token:
        email, google_id, name = _verify_google_access_token(req.access_token, client_id)
    else:
        raise HTTPException(status_code=400, detail="Thiếu thông tin xác thực Google.")

    email = email.strip().lower()
    user = db.query(User).filter(User.email == email).first()

    if user:
        if not user.is_verified:
            # Chống pre-hijack: kẻ tấn công đăng ký sẵn email nạn nhân kèm mật khẩu của hắn
            user.password_hash = None
        user.google_id = google_id
        user.is_verified = True
        user.verification_code = None
        user.verification_code_expires_at = None
        user.verification_attempts = 0
        if not user.full_name and name:
            user.full_name = name
    else:
        user = User(
            email=email,
            full_name=name or email.split("@")[0],
            auth_provider="google",
            google_id=google_id,
            is_verified=True,
        )
        db.add(user)

    db.commit()
    db.refresh(user)
    return _auth_success(user, "Đăng nhập bằng Google thành công!")


# ===========================================================================
# Session
# ===========================================================================
@router.post("/logout")
def logout():
    # JWT stateless: client tự xoá token. (Nếu cần thu hồi phía server -> thêm denylist/refresh token.)
    return {"status": "ok", "message": "Đã đăng xuất."}


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user)):
    return _user_response(user)
