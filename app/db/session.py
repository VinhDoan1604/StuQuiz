"""
Engine + session factory dùng chung cho toàn app (dependency `get_db`).

- URL lấy từ DATABASE_URL, hoặc ghép từ POSTGRES_* nếu không đặt.
- CHỈ ở APP_ENV=dev: nếu Postgres không kết nối được thì tự fallback SQLite
  cục bộ (sqlite:///./studyrag.db) để chạy thử không cần Docker. Ở môi trường
  khác, lỗi kết nối làm app dừng ngay lúc khởi động (fail fast) thay vì âm
  thầm ghi dữ liệu thật vào 1 file SQLite.
"""
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


def _make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        connect_args={"connect_timeout": 5} if url.startswith("postgresql") else {},
    )


def _init_engine() -> Engine:
    url = settings.sqlalchemy_url
    if url.startswith("sqlite"):
        return _make_engine(url)

    eng = _make_engine(url)
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Đã kết nối cơ sở dữ liệu %s.", eng.url.render_as_string(hide_password=True))
        return eng
    except Exception as exc:
        if not settings.is_dev:
            raise RuntimeError(f"Không kết nối được cơ sở dữ liệu: {exc}") from exc
        logger.warning(
            "Không kết nối được Postgres (%s) — APP_ENV=dev nên fallback SQLite %s.",
            exc, settings.sqlite_fallback_url,
        )
        eng.dispose()
        return _make_engine(settings.sqlite_fallback_url)


engine = _init_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
