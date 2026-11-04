"""
Registry các LLM provider — nơi DUY NHẤT cần sửa khi thêm/bớt provider.

Mỗi provider khai báo:
- `protocol`: định dạng API ("openai" | "anthropic" | "google") => quyết định
  class LangChain dùng để gọi (ChatOpenAI / ChatAnthropic / ChatGoogleGenerativeAI).
- `api_key_setting`: tên field trong app/config.py chứa API key.
- `base_url_setting`: tên field chứa base URL (None = endpoint gốc của hãng).

Thêm provider mới tương thích OpenAI (OpenRouter, DeepSeek, Groq, ...):
  1. Thêm 2 field vào Settings (vd `openrouter_api_key`, `openrouter_base_url`).
  2. Thêm 1 dòng vào PROVIDERS bên dưới.
  3. Khai báo model trong .env: LLM_CHAT_MODELS=openrouter:<model>,...

Model user được chọn trên UI: biến LLM_CHAT_MODELS, dạng
  "<provider>:<model>[|Nhãn hiển thị], ..."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings

Protocol = Literal["openai", "anthropic", "google"]


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    protocol: Protocol
    api_key_setting: str
    base_url_setting: str | None = None
    # Gửi kèm "Authorization: Bearer <key>" (gateway kiểu ShopAIKey xác thực bằng header này)
    bearer_auth: bool = False

    def api_key(self) -> str | None:
        return getattr(settings, self.api_key_setting, None) or None

    def base_url(self) -> str | None:
        if not self.base_url_setting:
            return None
        return getattr(settings, self.base_url_setting, None) or None


PROVIDERS: dict[str, ProviderSpec] = {
    # --- Gọi thẳng API của hãng ---
    "google": ProviderSpec("google", "Google", "google", "effective_google_api_key"),
    "openai": ProviderSpec("openai", "OpenAI", "openai", "openai_api_key"),
    "anthropic": ProviderSpec("anthropic", "Anthropic", "anthropic", "anthropic_api_key"),
    # --- ShopAIKey (1 key dùng chung cho cả 3 định dạng) ---
    "shopaikey_openai": ProviderSpec(
        "shopaikey_openai", "ShopAIKey", "openai", "shopaikey_api_key", "shopaikey_openai_base_url"
    ),
    "shopaikey_anthropic": ProviderSpec(
        "shopaikey_anthropic", "ShopAIKey", "anthropic", "shopaikey_api_key", "shopaikey_base_url",
        bearer_auth=True,
    ),
    "shopaikey_google": ProviderSpec(
        "shopaikey_google", "ShopAIKey", "google", "shopaikey_api_key", "shopaikey_base_url"
    ),
}

# Tên cũ vẫn dùng được trong .env
_ALIASES = {"gemini": "google"}


def get_provider(name: str) -> ProviderSpec:
    key = _ALIASES.get(name.strip().lower(), name.strip().lower())
    spec = PROVIDERS.get(key)
    if spec is None:
        raise ValueError(f"Provider không được hỗ trợ: '{name}'. Hỗ trợ: {', '.join(sorted(PROVIDERS))}.")
    return spec


def build_chat_model(spec: ProviderSpec, model: str, common: dict[str, Any]) -> BaseChatModel:
    """Khởi tạo chat model LangChain theo `spec.protocol`."""
    api_key = spec.api_key()
    if not api_key:
        raise ValueError(
            f"Thiếu API key cho provider '{spec.name}' (đặt {spec.api_key_setting.upper()} trong .env)."
        )
    base_url = spec.base_url()
    timeout = settings.llm_timeout_seconds

    if spec.protocol == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        extra: dict[str, Any] = {"base_url": base_url} if base_url else {}
        return ChatGoogleGenerativeAI(google_api_key=api_key, timeout=timeout, **extra, **common)

    if spec.protocol == "openai":
        from langchain_openai import ChatOpenAI

        extra = {"base_url": base_url} if base_url else {}
        return ChatOpenAI(api_key=api_key, timeout=timeout, **extra, **common)

    if spec.protocol == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError("Để dùng Anthropic: pip install langchain-anthropic") from exc
        extra = {"base_url": base_url} if base_url else {}
        if spec.bearer_auth:
            extra["default_headers"] = {"Authorization": f"Bearer {api_key}"}
        return ChatAnthropic(api_key=api_key, default_request_timeout=timeout, **extra, **common)

    raise ValueError(f"Protocol không hỗ trợ: '{spec.protocol}'")


# ---------------------------------------------------------------------------
# Danh sách model cho user chọn trên UI
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatModelOption:
    id: str          # "<provider>:<model>" — giá trị client gửi lên trong ChatRequest.model
    provider: str
    model: str
    label: str


def parse_model_options(raw: str | None) -> list[ChatModelOption]:
    """Parse LLM_CHAT_MODELS. Bỏ qua mục sai định dạng / provider lạ (không làm sập app)."""
    options: list[ChatModelOption] = []
    seen: set[str] = set()
    for item in (raw or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        ref, _, label = item.partition("|")
        provider, _, model = ref.strip().partition(":")
        provider, model = provider.strip().lower(), model.strip()
        try:
            spec = get_provider(provider)
        except ValueError:
            continue
        if not model:
            continue
        option_id = f"{spec.name}:{model}"
        if option_id in seen:
            continue
        seen.add(option_id)
        options.append(ChatModelOption(option_id, spec.name, model, label.strip() or model))
    return options


def chat_model_options() -> list[ChatModelOption]:
    return parse_model_options(settings.llm_chat_models)


def find_model_option(option_id: str) -> ChatModelOption | None:
    return next((o for o in chat_model_options() if o.id == option_id), None)
