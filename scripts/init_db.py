"""
Khởi tạo / nâng cấp schema bằng Alembic (`alembic upgrade head`).

    python scripts/init_db.py            # tạo mới hoặc nâng cấp lên bản mới nhất
    python scripts/init_db.py --reset    # XOÁ TOÀN BỘ dữ liệu rồi tạo lại (chỉ APP_ENV=dev)

DB cũ tạo bằng create_all (trước khi có Alembic): chạy 1 lần
`alembic stamp 0001_baseline` rồi chạy lại script này.
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

from app.config import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="xoá toàn bộ bảng rồi tạo lại (chỉ dev)")
    args = parser.parse_args()

    cfg = Config(os.path.join(ROOT, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(ROOT, "migrations"))

    if args.reset:
        if not settings.is_dev:
            sys.exit("--reset chỉ được phép khi APP_ENV=dev.")
        print("Đang xoá toàn bộ schema...")
        command.downgrade(cfg, "base")

    command.upgrade(cfg, "head")
    print("Schema đã ở phiên bản mới nhất.")


if __name__ == "__main__":
    main()
