# STATUS: READY
"""
Engine + session factory cho Postgres với cơ chế tự động fallback SQLite cục bộ
nếu Docker / Postgres chưa khởi chạy. Dùng chung cho toàn app qua dependency `get_db`.
"""
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


def _init_engine():
    try:
        eng = create_engine(
            settings.postgres_dsn,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 2}
        )
        with eng.connect():
            logger.info("Kết nối thành công cơ sở dữ liệu PostgreSQL.")
            return eng
    except Exception as e:
        logger.warning(
            f"Không thể kết nối PostgreSQL ({e}). "
            "Kích hoạt chế độ fallback SQLite cục bộ (sqlite:///./studyrag.db) để đảm bảo hệ thống luôn sẵn sàng 100%."
        )
        eng = create_engine("sqlite:///./studyrag.db", connect_args={"check_same_thread": False})
        return eng


engine = _init_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
