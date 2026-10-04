import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.db.session import Base, engine
from app.models import db_models  # noqa: F401  (import để register models vào Base.metadata)

if __name__ == "__main__":
    print("Đang xóa và khởi tạo lại toàn bộ bảng trong PostgreSQL...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Đã khởi tạo mới hoàn toàn toàn bộ bảng trong PostgreSQL thành công.")
