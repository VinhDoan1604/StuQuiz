"""
Luồng đăng ký / đăng nhập / OTP end-to-end (SQLite in-memory, không cần SMTP/Google thật).
Chạy: pytest tests/test_auth_flows.py -v
"""
from datetime import timedelta

import pytest

import app.models.db_models as m
from app.api import routes_auth
from app.config import settings
from app.utils.time import utcnow

PW = "password123"


def _register(client, email="u@x.com", password=PW, name="U"):
    return client.post("/auth/register", json={"email": email, "password": password, "full_name": name})


def _expire_otp(client, email):
    with client.Session() as db:
        u = db.query(m.User).filter(m.User.email == email).first()
        u.verification_code_expires_at = utcnow() - timedelta(minutes=1)
        db.commit()


def _clear_cooldown(client, email):
    """Đẩy thời điểm gửi OTP về quá khứ (hạn dùng vẫn còn)."""
    with client.Session() as db:
        u = db.query(m.User).filter(m.User.email == email).first()
        u.verification_code_expires_at = utcnow() + timedelta(seconds=30)
        db.commit()


def test_register_verify_then_login(client):
    r = _register(client)
    assert r.status_code == 200 and r.json()["status"] == "needs_verification"
    assert r.json()["token"] is None  # chưa xác thực thì không được cấp token

    r = client.post("/auth/verify-otp", json={"email": "u@x.com", "code": r.json()["dev_code"]})
    assert r.status_code == 200 and r.json()["token"]

    r = client.post("/auth/login", json={"email": "u@x.com", "password": PW})
    assert r.status_code == 200 and r.json()["status"] == "success" and r.json()["token"]
    token = r.json()["token"]
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["email"] == "u@x.com"


def test_email_is_case_and_whitespace_insensitive(client):
    r = _register(client, email="  Mixed@X.com ")
    code = r.json()["dev_code"]
    assert r.json()["email"] == "mixed@x.com"
    assert client.post("/auth/verify-otp", json={"email": "MIXED@x.com ", "code": code}).status_code == 200
    assert client.post("/auth/login", json={"email": " MIXED@X.COM", "password": PW}).json()["token"]


def test_duplicate_register_of_verified_email_is_409(client, make_user):
    make_user("dup@x.com")
    assert _register(client, email="dup@x.com").status_code == 409


def test_unverified_user_gets_no_token_and_no_access(client):
    _register(client)
    r = client.post("/auth/login", json={"email": "u@x.com", "password": PW})
    assert r.status_code == 200 and r.json()["status"] == "needs_verification"
    assert r.json()["token"] is None


def test_resend_respects_cooldown_and_then_works(client):
    _register(client)
    assert client.post("/auth/resend-otp", json={"email": "u@x.com"}).status_code == 429

    with client.Session() as db:  # coi như đã qua thời gian chờ
        u = db.query(m.User).first()
        u.verification_code_expires_at = utcnow() + timedelta(minutes=settings.otp_ttl_minutes) - timedelta(
            seconds=settings.otp_resend_cooldown_seconds + 5
        )
        db.commit()
    r = client.post("/auth/resend-otp", json={"email": "u@x.com"})
    assert r.status_code == 200 and r.json()["dev_code"]
    assert client.post("/auth/verify-otp", json={"email": "u@x.com", "code": r.json()["dev_code"]}).status_code == 200


def test_resend_does_not_reveal_whether_email_exists(client):
    a = client.post("/auth/resend-otp", json={"email": "nobody@x.com"})
    assert a.status_code == 200 and a.json()["dev_code"] is None


def test_expired_otp_is_rejected_and_login_issues_a_new_one(client):
    code = _register(client).json()["dev_code"]
    _expire_otp(client, "u@x.com")
    assert client.post("/auth/verify-otp", json={"email": "u@x.com", "code": code}).status_code == 400

    # Người dùng quay lại đăng nhập: hệ thống phải cấp mã mới (không bị kẹt)
    r = client.post("/auth/login", json={"email": "u@x.com", "password": PW})
    assert r.json()["status"] == "needs_verification" and r.json()["dev_code"]
    r = client.post("/auth/verify-otp", json={"email": "u@x.com", "code": r.json()["dev_code"]})
    assert r.status_code == 200 and r.json()["token"]


def test_otp_lockout_is_cleared_by_new_code(client):
    code = _register(client).json()["dev_code"]
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(settings.otp_max_attempts):
        client.post("/auth/verify-otp", json={"email": "u@x.com", "code": wrong})
    assert client.post("/auth/verify-otp", json={"email": "u@x.com", "code": code}).status_code == 429

    _expire_otp(client, "u@x.com")  # qua thời gian chờ gửi lại
    r = client.post("/auth/login", json={"email": "u@x.com", "password": PW})
    assert r.json()["dev_code"]
    assert client.post("/auth/verify-otp", json={"email": "u@x.com", "code": r.json()["dev_code"]}).status_code == 200


def test_register_again_while_unverified_replaces_password(client):
    _register(client, password="first-password")
    _clear_cooldown(client, "u@x.com")
    with client.Session() as db:
        u = db.query(m.User).first()
        u.verification_code_expires_at = utcnow() - timedelta(minutes=1)
        db.commit()
    r = _register(client, password="second-password")
    assert r.status_code == 200
    client.post("/auth/verify-otp", json={"email": "u@x.com", "code": r.json()["dev_code"]})
    assert client.post("/auth/login", json={"email": "u@x.com", "password": "first-password"}).status_code == 401
    assert client.post("/auth/login", json={"email": "u@x.com", "password": "second-password"}).json()["token"]


def test_invalid_email_and_oversized_input(client):
    assert _register(client, email="not-an-email").status_code == 400
    assert client.post("/auth/resend-otp", json={"email": "a" * 400 + "@x.com"}).status_code in (400, 422)
    assert client.post("/auth/login", json={"email": "x@x.com", "password": "p" * 500}).status_code == 422


def test_google_login_on_existing_local_account_keeps_password_login(client, make_user, monkeypatch):
    make_user("g@x.com", password=PW)
    monkeypatch.setattr(settings, "google_oauth_client_id", "cid")
    monkeypatch.setattr(routes_auth, "_verify_google_access_token", lambda t, c: ("g@x.com", "gid-1", "G"))
    r = client.post("/auth/google", json={"access_token": "t"})
    assert r.status_code == 200 and r.json()["token"]
    assert client.post("/auth/login", json={"email": "g@x.com", "password": PW}).json()["token"]


def test_google_login_cannot_be_pre_hijacked_by_unverified_signup(client, monkeypatch):
    _register(client, email="v@x.com", password="attacker-password")  # kẻ tấn công đăng ký trước
    monkeypatch.setattr(settings, "google_oauth_client_id", "cid")
    monkeypatch.setattr(routes_auth, "_verify_google_access_token", lambda t, c: ("v@x.com", "gid-2", "V"))
    assert client.post("/auth/google", json={"access_token": "t"}).status_code == 200
    assert client.post("/auth/login", json={"email": "v@x.com", "password": "attacker-password"}).status_code == 401


def test_google_only_account_cannot_login_with_password(client, monkeypatch):
    monkeypatch.setattr(settings, "google_oauth_client_id", "cid")
    monkeypatch.setattr(routes_auth, "_verify_google_access_token", lambda t, c: ("only@x.com", "gid-3", "O"))
    client.post("/auth/google", json={"access_token": "t"})
    assert client.post("/auth/login", json={"email": "only@x.com", "password": ""}).status_code == 401


def test_demo_login_gives_a_working_session(client):
    r = client.post("/auth/demo")
    assert r.status_code == 200
    h = {"Authorization": f"Bearer {r.json()['token']}"}
    assert client.get("/auth/me", headers=h).status_code == 200


# ----------------------------------------------------------------- production
@pytest.fixture()
def prod_no_mail(monkeypatch):
    """Môi trường prod mà email gửi thất bại (SMTP chưa cấu hình hoặc lỗi mạng)."""
    monkeypatch.setattr(settings, "app_env", "prod")
    monkeypatch.setattr(routes_auth, "send_verification_email", lambda to, code: False)


def test_prod_mail_failure_returns_503_and_does_not_lock_the_user_out(client, prod_no_mail, monkeypatch):
    r = _register(client)
    assert r.status_code == 503

    # SMTP được sửa xong: người dùng thử lại ngay phải được, không bị 429 vì một mã chưa từng tới tay họ
    sent = []
    monkeypatch.setattr(routes_auth, "send_verification_email", lambda to, code: sent.append(code) or True)
    r = _register(client)
    assert r.status_code == 200, r.json()
    assert sent
    assert client.post("/auth/verify-otp", json={"email": "u@x.com", "code": sent[-1]}).status_code == 200


def test_prod_never_returns_otp_in_response(client, monkeypatch):
    monkeypatch.setattr(settings, "app_env", "prod")
    monkeypatch.setattr(routes_auth, "send_verification_email", lambda to, code: True)
    r = _register(client)
    assert r.status_code == 200 and r.json()["dev_code"] is None
