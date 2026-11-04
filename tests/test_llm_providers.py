"""Registry provider + chọn model theo request (không gọi mạng)."""
import pytest

from app.config import settings
from app.llm import router as llm_router
from app.llm.providers import build_chat_model, get_provider, parse_model_options


@pytest.fixture()
def shopaikey(monkeypatch):
    monkeypatch.setattr(settings, "shopaikey_api_key", "sk-test")
    monkeypatch.setattr(
        settings,
        "llm_chat_models",
        "shopaikey_google:gemini-2.5-flash|Gemini 2.5 Flash,shopaikey_openai:gpt-5-mini,"
        "shopaikey_anthropic:claude-sonnet-4-20250514",
    )
    llm_router._build_llm.cache_clear()
    yield
    llm_router._build_llm.cache_clear()


def test_parse_model_options_skips_invalid():
    opts = parse_model_options("shopaikey_google:g1|Nhãn, bad, unknown:x, openai:, gemini:g2, shopaikey_google:g1")
    assert [(o.id, o.label) for o in opts] == [("shopaikey_google:g1", "Nhãn"), ("google:g2", "g2")]


def test_unknown_provider():
    with pytest.raises(ValueError):
        get_provider("nope")


def test_missing_key_is_clear_error(monkeypatch):
    monkeypatch.setattr(settings, "shopaikey_api_key", None)
    with pytest.raises(ValueError, match="SHOPAIKEY_API_KEY"):
        build_chat_model(get_provider("shopaikey_openai"), "gpt-5-mini", {"model": "gpt-5-mini"})


def test_shopaikey_clients_use_gateway(shopaikey):
    g = build_chat_model(get_provider("shopaikey_google"), "gemini-2.5-flash", {"model": "gemini-2.5-flash"})
    assert g.client._api_client._http_options.base_url.rstrip("/") == "https://api.shopaikey.com"
    o = build_chat_model(get_provider("shopaikey_openai"), "gpt-5-mini", {"model": "gpt-5-mini"})
    assert str(o.root_client.base_url).rstrip("/") == "https://api.shopaikey.com/v1"
    a = build_chat_model(get_provider("shopaikey_anthropic"), "claude-x", {"model": "claude-x"})
    assert str(a._client.base_url).rstrip("/") == "https://api.shopaikey.com"
    assert a.default_headers["Authorization"] == "Bearer sk-test"


def test_override_applies_only_to_answer_tasks(shopaikey, monkeypatch):
    monkeypatch.setattr(settings, "cheap_model_provider", "shopaikey_google")
    monkeypatch.setattr(settings, "cheap_model_name", "gemini-2.5-flash")
    token = llm_router.set_model_override("shopaikey_openai", "gpt-5-mini")
    try:
        assert llm_router.get_llm("teaching").model_name == "gpt-5-mini"
        assert llm_router.get_llm("qa").model_name == "gpt-5-mini"
        assert llm_router.get_llm("grade").model == "gemini-2.5-flash"  # task nội bộ giữ mặc định
    finally:
        llm_router.reset_model_override(token)
    assert llm_router.get_llm("qa").model == "gemini-2.5-flash"


def test_models_endpoint_and_validation(client, make_user, shopaikey):
    _, headers = make_user("p@x.com")

    r = client.get("/chat/models", headers=headers)
    assert r.status_code == 200
    ids = [x["id"] for x in r.json()["models"]]
    assert ids[0] == "shopaikey_google:gemini-2.5-flash" and len(ids) == 3

    r = client.post("/chat", headers=headers, json={"session_id": "s1", "message": "hi", "model": "evil:gpt-9"})
    assert r.status_code == 400
