# STATUS: READY
# app/config.py
"""
Cấu hình tập trung cho toàn app, đọc từ .env qua pydantic-settings.
Mọi module khác import `settings` từ đây, KHÔNG đọc os.environ trực tiếp
rải rác trong code.
"""
from urllib.parse import quote_plus
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # LLM routing (xem app/llm/router.py)
    # Mặc định dùng Gemini 3.8 Flash cho tốc độ siêu tốc và chất lượng cao nhất
    cheap_model_provider: str = "google"       # "openai"
    cheap_model_name: str = "gemini-3.8-flash" # "gpt-4o-mini"
    strong_model_provider: str = "google"      # "anthropic"
    strong_model_name: str = "gemini-3.8-flash"# "gemini-3.8-flash"

    # API Keys
    google_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None          # "sk-..."
    anthropic_api_key: str | None = None       # "sk-ant-..."

    # Google OAuth 2.0 Credentials
    google_oauth_client_id: str | None = None
    google_oauth_client_secret: str | None = None

    app_env: str = "dev"
    log_level: str = "INFO"
    max_reflection_loops: int = 2

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
