"""Đăng nhập Google bằng access token (luồng Google Identity Services của frontend)."""
from unittest.mock import patch

from app.config import settings

CLIENT_ID = "123-test.apps.googleusercontent.com"


def _fake_google(aud: str, email_verified: bool = True):
    def fake(url, data=None, headers=None):
        if "tokeninfo" in url:
            return {"aud": aud, "azp": aud}
        return {"email": "Learner@Gmail.com", "email_verified": email_verified, "sub": "g-1", "name": "Learner"}
    return fake


def test_google_access_token_login_creates_user(client, monkeypatch):
    monkeypatch.setattr(settings, "google_oauth_client_id", CLIENT_ID)
    with patch("app.api.routes_auth._http_json", side_effect=_fake_google(CLIENT_ID)):
        r = client.post("/auth/google", json={"access_token": "tok"})
    assert r.status_code == 200
    body = r.json()
    assert body["token"] and body["user"]["email"] == "learner@gmail.com"
    assert body["user"]["auth_provider"] == "google"


def test_google_token_for_another_app_is_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "google_oauth_client_id", CLIENT_ID)
    with patch("app.api.routes_auth._http_json", side_effect=_fake_google("other-app.apps.googleusercontent.com")):
        r = client.post("/auth/google", json={"access_token": "tok"})
    assert r.status_code == 401 and "token" not in r.json()


def test_google_unverified_email_is_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "google_oauth_client_id", CLIENT_ID)
    with patch("app.api.routes_auth._http_json", side_effect=_fake_google(CLIENT_ID, email_verified=False)):
        assert client.post("/auth/google", json={"access_token": "tok"}).status_code == 401


def test_google_config_exposes_client_id_only(client, monkeypatch):
    monkeypatch.setattr(settings, "google_oauth_client_id", CLIENT_ID)
    assert client.get("/auth/google/config").json() == {"client_id": CLIENT_ID}
    monkeypatch.setattr(settings, "google_oauth_client_id", "")
    assert client.get("/auth/google/config").json() == {"client_id": None}
