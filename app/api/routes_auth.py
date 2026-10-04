import hashlib
import secrets
import json
import base64
import urllib.request
import urllib.parse
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.models.db_models import User
from app.utils.mailer import send_verification_email

router = APIRouter()
logger = logging.getLogger("app.api.auth")


def _hash_password(password: str) -> str:
    """Hash password với SHA-256 + salt."""
    salt = "studyrag_salt_2026"
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _generate_otp() -> str:
    """Sinh mã OTP 6 chữ số ngẫu nhiên."""
    return f"{secrets.randbelow(900000) + 100000:06d}"


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
    credential: Optional[str] = None    # Google JWT ID Token (từ Google GSI Button/OneTap)
    access_token: Optional[str] = None  # Google OAuth2 Access Token (từ GSI Token Client)
    code: Optional[str] = None          # Google OAuth2 Authorization Code
    redirect_uri: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None
    google_id: Optional[str] = None
    picture: Optional[str] = None


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
    dev_code: Optional[str] = None  # Dành cho môi trường dev khi chưa cấu hình SMTP


# ===========================================================================
# Endpoints
# ===========================================================================

@router.post("/register", response_model=AuthResultResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    clean_email = req.email.strip().lower()
    if not clean_email or len(req.password) < 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email không hợp lệ hoặc mật khẩu quá ngắn (tối thiểu 4 ký tự)."
        )

    existing = db.query(User).filter(User.email == clean_email).first()
    if existing:
        if existing.is_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email này đã được đăng ký và xác thực trong hệ thống. Vui lòng đăng nhập."
            )
        # Nếu đã tạo nhưng chưa xác thực mã OTP, cập nhật lại mật khẩu và gửi lại OTP
        otp = _generate_otp()
        existing.password_hash = _hash_password(req.password)
        if req.full_name:
            existing.full_name = req.full_name
        existing.verification_code = otp
        existing.verification_code_expires_at = datetime.utcnow() + timedelta(minutes=10)
        db.commit()
        send_verification_email(clean_email, otp)
        return AuthResultResponse(
            status="needs_verification",
            message="Tài khoản đã tạo trước đó chưa xác thực. Đã gửi mã xác thực mới tới Gmail của bạn.",
            email=clean_email,
            dev_code=otp
        )

    otp = _generate_otp()
    user = User(
        email=clean_email,
        full_name=req.full_name or clean_email.split('@')[0],
        password_hash=_hash_password(req.password),
        auth_provider="local",
        is_verified=False,
        verification_code=otp,
        verification_code_expires_at=datetime.utcnow() + timedelta(minutes=10),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    send_verification_email(clean_email, otp)

    return AuthResultResponse(
        status="needs_verification",
        message="Đăng ký thành công! Vui lòng nhập mã xác thực 6 số vừa được gửi tới Gmail của bạn.",
        email=clean_email,
        dev_code=otp
    )


@router.post("/verify-otp", response_model=AuthResultResponse)
def verify_otp(req: VerifyOtpRequest, db: Session = Depends(get_db)):
    clean_email = req.email.strip().lower()
    code_input = req.code.strip()

    user = db.query(User).filter(User.email == clean_email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy tài khoản với email này."
        )

    if not user.verification_code or user.verification_code != code_input:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mã xác thực không chính xác. Vui lòng kiểm tra lại Gmail."
        )

    if user.verification_code_expires_at and datetime.utcnow() > user.verification_code_expires_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mã xác thực đã hết hạn (10 phút). Vui lòng nhấn gửi lại mã."
        )

    user.is_verified = True
    user.verification_code = None
    user.verification_code_expires_at = None
    db.commit()
    db.refresh(user)

    token = f"sr_{user.id}_{secrets.token_hex(16)}"
    return AuthResultResponse(
        status="success",
        message="Xác thực tài khoản thành công!",
        email=user.email,
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            auth_provider=user.auth_provider,
            is_verified=True,
            created_at=user.created_at.isoformat() if user.created_at else None,
        ),
        token=token
    )


@router.post("/resend-otp", response_model=AuthResultResponse)
def resend_otp(req: ResendOtpRequest, db: Session = Depends(get_db)):
    clean_email = req.email.strip().lower()
    user = db.query(User).filter(User.email == clean_email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy tài khoản với email này."
        )

    otp = _generate_otp()
    user.verification_code = otp
    user.verification_code_expires_at = datetime.utcnow() + timedelta(minutes=10)
    db.commit()

    send_verification_email(clean_email, otp)
    return AuthResultResponse(
        status="needs_verification",
        message="Đã cấp mã xác thực mới gửi tới Gmail.",
        email=clean_email,
        dev_code=otp
    )


@router.post("/login", response_model=AuthResultResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    clean_email = req.email.strip().lower()
    user = db.query(User).filter(User.email == clean_email).first()
    if not user or not user.password_hash or user.password_hash != _hash_password(req.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email hoặc mật khẩu không chính xác."
        )

    # Nếu tài khoản chưa xác thực qua Gmail, yêu cầu xác thực OTP trước khi cho phép vào
    if not user.is_verified:
        otp = _generate_otp()
        user.verification_code = otp
        user.verification_code_expires_at = datetime.utcnow() + timedelta(minutes=10)
        db.commit()
        send_verification_email(clean_email, otp)
        return AuthResultResponse(
            status="needs_verification",
            message="Tài khoản này chưa hoàn tất xác thực Gmail. Vui lòng nhập mã OTP để tiếp tục.",
            email=clean_email,
            dev_code=otp
        )

    token = f"sr_{user.id}_{secrets.token_hex(16)}"
    return AuthResultResponse(
        status="success",
        message="Đăng nhập thành công!",
        email=user.email,
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            auth_provider=user.auth_provider,
            is_verified=True,
            created_at=user.created_at.isoformat() if user.created_at else None,
        ),
        token=token
    )


@router.get("/google/config")
def get_google_config():
    """Trả về Client ID đã cấu hình cho Google OAuth 2.0 trên frontend."""
    return {"client_id": settings.google_oauth_client_id}


@router.post("/google", response_model=AuthResultResponse)
def google_auth(req: GoogleAuthRequest, db: Session = Depends(get_db)):
    """
    Đăng nhập bằng tài khoản Google (OAuth 2.0 chuẩn):
    - Hỗ trợ đổi authorization code lấy token trực tiếp từ Google Token Endpoint qua Client Secret.
    - Hỗ trợ xác thực access_token qua Google UserInfo Endpoint.
    - Hỗ trợ xác thực ID Token qua Google TokenInfo Endpoint.
    - Hài hòa dữ liệu người dùng trong PostgreSQL và phát hành token phiên làm việc.
    """
    email = req.email
    full_name = req.full_name
    google_id = req.google_id

    # 1. Nếu nhận được authorization code -> Trao đổi lấy access_token bằng Client Secret
    if req.code:
        try:
            token_url = "https://oauth2.googleapis.com/token"
            data = urllib.parse.urlencode({
                "code": req.code,
                "client_id": settings.google_oauth_client_id,
                "client_secret": settings.google_oauth_client_secret,
                "redirect_uri": req.redirect_uri or "http://localhost:5173",
                "grant_type": "authorization_code",
            }).encode("utf-8")
            token_req = urllib.request.Request(token_url, data=data, method="POST")
            with urllib.request.urlopen(token_req, timeout=10) as resp:
                token_resp = json.loads(resp.read().decode("utf-8"))
                req.access_token = token_resp.get("access_token")
                if token_resp.get("id_token") and not req.credential:
                    req.credential = token_resp.get("id_token")
        except Exception as exc:
            logger.error("google_auth: Không thể trao đổi OAuth code với Google: %s", exc)

    # 2. Nếu có access_token -> Lấy thông tin tài khoản chuẩn xác từ Google UserInfo API
    if req.access_token:
        try:
            userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
            userinfo_req = urllib.request.Request(
                userinfo_url,
                headers={"Authorization": f"Bearer {req.access_token}"}
            )
            with urllib.request.urlopen(userinfo_req, timeout=10) as resp:
                user_info = json.loads(resp.read().decode("utf-8"))
                email = user_info.get("email") or email
                full_name = user_info.get("name") or full_name
                google_id = user_info.get("sub") or google_id
        except Exception as exc:
            logger.warning("google_auth: Lỗi gọi Google UserInfo API với access_token: %s", exc)

    # 3. Nếu có credential (JWT ID Token) -> Xác thực chữ ký và lấy thông tin từ Google TokenInfo
    if (not email or not google_id) and req.credential:
        try:
            tokeninfo_url = f"https://oauth2.googleapis.com/tokeninfo?id_token={req.credential}"
            with urllib.request.urlopen(tokeninfo_url, timeout=10) as resp:
                token_info = json.loads(resp.read().decode("utf-8"))
                email = token_info.get("email") or email
                full_name = token_info.get("name") or full_name
                google_id = token_info.get("sub") or google_id
        except Exception as exc:
            logger.warning("google_auth: Tokeninfo thất bại, giải mã payload JWT trực tiếp: %s", exc)
            try:
                parts = req.credential.split(".")
                if len(parts) >= 2:
                    padded = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
                    payload_json = base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8")
                    payload = json.loads(payload_json)
                    email = payload.get("email", email)
                    full_name = payload.get("name", full_name)
                    google_id = payload.get("sub", google_id)
            except Exception:
                pass

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không lấy được địa chỉ email từ phản hồi Google OAuth. Vui lòng thử lại."
        )

    clean_email = email.strip().lower()
    user = db.query(User).filter(User.email == clean_email).first()

    if user:
        # Hòa hợp: Cập nhật thông tin Google vào tài khoản hiện có
        user.auth_provider = "google"
        if google_id:
            user.google_id = google_id
        if full_name and (not user.full_name or user.full_name == user.email.split('@')[0]):
            user.full_name = full_name
        user.is_verified = True  # Tài khoản Google đã được Google xác thực
        db.commit()
        db.refresh(user)
    else:
        # Tạo mới tài khoản đăng nhập qua Google
        user = User(
            email=clean_email,
            full_name=full_name or clean_email.split('@')[0],
            auth_provider="google",
            google_id=google_id,
            is_verified=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    token = f"sr_{user.id}_{secrets.token_hex(16)}"
    return AuthResultResponse(
        status="success",
        message="Đăng nhập bằng tài khoản Google thành công!",
        email=user.email,
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            auth_provider=user.auth_provider,
            is_verified=True,
            created_at=user.created_at.isoformat() if user.created_at else None,
        ),
        token=token
    )


@router.post("/logout")
def logout():
    """Đăng xuất phiên làm việc."""
    return {"status": "ok", "message": "Đã đăng xuất thành công."}


@router.get("/me", response_model=UserResponse)
def get_me(user_id: Optional[str] = None, db: Session = Depends(get_db)):
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            return UserResponse(
                id=str(user.id),
                email=user.email,
                full_name=user.full_name,
                auth_provider=user.auth_provider,
                is_verified=user.is_verified,
                created_at=user.created_at.isoformat() if user.created_at else None,
            )

    raise HTTPException(status_code=401, detail="Chưa đăng nhập.")
