# app/config.py
"""
Cấu hình tập trung cho toàn app, đọc từ .env qua pydantic-settings.
Mọi module khác import `settings` từ đây, KHÔNG đọc os.environ trực tiếp.
"""
from urllib.parse import quote_plus

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_VERSION = "1.0.0"

_DEV_JWT_SECRET = "dev-insecure-change-me"
_SQLITE_FALLBACK_URL = "sqlite:///./studyrag.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    app_env: str = "dev"                   # dev | staging | prod
    log_level: str = "INFO"
    log_format: str = "text"               # text | json
    # URL frontend (dùng làm redirect_uri mặc định cho Google OAuth)
    public_app_url: str = "http://localhost:5173"

    # --- Database ---
    # Nếu đặt DATABASE_URL thì dùng nguyên chuỗi này; ngược lại ghép từ POSTGRES_*.
    database_url: str | None = None
    postgres_user: str = "studyrag"
    postgres_password: str = "changeme"
    postgres_db: str = "studyrag"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_timeout_seconds: int = 30

    # --- LLM routing (xem app/llm/router.py) ---
    # Provider: xem PROVIDERS trong app/llm/providers.py
    cheap_model_provider: str = "shopaikey_google"
    cheap_model_name: str = "gemini-2.5-flash"
    strong_model_provider: str = "shopaikey_google"
    strong_model_name: str = "gemini-2.5-flash"
    # Danh sách model user chọn trên UI: "<provider>:<model>[|Nhãn]", phân tách bằng dấu phẩy
    llm_chat_models: str = ""
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 2
    # Judge cho eval/ (eval/judge.py). Biến môi trường cùng tên ghi đè giá trị trong .env.
    # Để trống -> dùng STRONG model (và báo cáo sẽ cảnh báo nếu trùng model sinh).
    eval_judge_provider: str = ""
    eval_judge_model: str = ""

    # API Keys
    google_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    # ShopAIKey: 1 key dùng cho cả định dạng OpenAI / Anthropic / Google GenAI
    shopaikey_api_key: str | None = None
    shopaikey_base_url: str = "https://api.shopaikey.com"            # Anthropic & Google GenAI
    shopaikey_openai_base_url: str = "https://api.shopaikey.com/v1"  # OpenAI

    # Google OAuth 2.0
    google_oauth_client_id: str | None = None
    google_oauth_client_secret: str | None = None

    # --- SMTP (gửi OTP) ---
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None           # mặc định = smtp_user

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
    # Giới hạn tần suất theo IP: số request / cửa sổ 60s. Có REDIS_URL => đếm chung
    # giữa mọi instance; không có => đếm riêng trong từng tiến trình.
    rate_limit_enabled: bool = True
    redis_url: str | None = None
    rate_limit_auth_per_minute: int = 10
    rate_limit_chat_per_minute: int = 30

    # --- RAG ---
    max_reflection_loops: int = 2
    max_chat_message_chars: int = 4000
    enable_parent_expansion: bool = True
    parent_context_max_tokens: int = 3500
    embedding_cache_size: int = 5000
    # Tải sẵn BGE-M3 + reranker lúc khởi động (tránh request đầu tiên chậm vài chục giây)
    preload_models: bool = False

    # --- Ingestion ---
    # Ngôn ngữ Tesseract cho OCR (PDF scan, hình trong slide). "vie+eng": tài liệu học song ngữ;
    # đo trên tài liệu thật: tiếng Việt gần như không đổi so với "vie", chữ tiếng Anh đọc chắc hơn.
    ocr_lang: str = "vie+eng"
    # OCR chữ trong hình của slide (sơ đồ, bảng chụp). Tắt nếu cần upload nhanh.
    ocr_slide_images: bool = True

    @model_validator(mode="after")
    def _validate_non_dev(self):
        if not self.jwt_secret:  # "JWT_SECRET=" để trống trong .env
            self.jwt_secret = _DEV_JWT_SECRET
        if self.app_env != "dev":
            if self.jwt_secret == _DEV_JWT_SECRET or len(self.jwt_secret) < 32:
                raise ValueError(
                    "JWT_SECRET phải được đặt (>= 32 ký tự) khi APP_ENV != 'dev'. "
                    'Sinh bằng: python -c "import secrets; print(secrets.token_urlsafe(48))"'
                )
        return self

    @property
    def is_dev(self) -> bool:
        return self.app_env == "dev"

    @property
    def demo_enabled(self) -> bool:
        if self.allow_demo_login is not None:
            return self.allow_demo_login
        return self.is_dev

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
    def sqlalchemy_url(self) -> str:
        return self.database_url or self.postgres_dsn

    @property
    def sqlite_fallback_url(self) -> str:
        return _SQLITE_FALLBACK_URL

    @property
    def effective_google_api_key(self) -> str | None:
        return self.google_api_key or self.gemini_api_key

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_user and self.smtp_password)


settings = Settings()
