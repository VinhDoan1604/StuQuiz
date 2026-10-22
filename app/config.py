# app/config.py
"""
Cấu hình tập trung cho toàn app, đọc từ .env qua pydantic-settings.
Mọi module khác import `settings` từ đây, KHÔNG đọc os.environ trực tiếp.
"""
from urllib.parse import quote_plus

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEV_JWT_SECRET = "dev-insecure-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres
    postgres_user: str = "studyrag"
    postgres_password: str = "changeme"
    postgres_db: str = "studyrag"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # LLM routing (xem app/llm/router.py) — sẽ được chuẩn hoá ở phần sau
    cheap_model_provider: str = "google"
    cheap_model_name: str = "gemini-3.8-flash"
    strong_model_provider: str = "google"
    strong_model_name: str = "gemini-3.8-flash"

    # API Keys
    google_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Google OAuth 2.0
    google_oauth_client_id: str | None = None
    google_oauth_client_secret: str | None = None

    # --- Security ---
    jwt_secret: str = _DEV_JWT_SECRET
    jwt_expire_days: int = 7
    # Danh sách origin được phép, phân tách bằng dấu phẩy
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # None => tự động: chỉ bật ở APP_ENV=dev
    allow_demo_login: bool | None = None
    otp_ttl_minutes: int = 10
    otp_max_attempts: int = 5
    otp_resend_cooldown_seconds: int = 60
    max_upload_mb: int = 50

    app_env: str = "dev"
    log_level: str = "INFO"
    max_reflection_loops: int = 2

    enable_parent_expansion: bool = True
    parent_context_max_tokens: int = 3500

    @model_validator(mode="after")
    def _validate_non_dev(self):
        if self.app_env != "dev":
            if self.jwt_secret == _DEV_JWT_SECRET or len(self.jwt_secret) < 32:
                raise ValueError(
                    "JWT_SECRET phải được đặt (>= 32 ký tự) khi APP_ENV != 'dev'. "
                    'Sinh bằng: python -c "import secrets; print(secrets.token_urlsafe(48))"'
                )
        return self

    @property
    def demo_enabled(self) -> bool:
        if self.allow_demo_login is not None:
            return self.allow_demo_login
        return self.app_env == "dev"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def postgres_dsn(self) -> str:
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql+psycopg2://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def effective_google_api_key(self) -> str | None:
        return self.google_api_key or self.gemini_api_key


settings = Settings()
