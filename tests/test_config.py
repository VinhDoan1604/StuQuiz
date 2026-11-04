import pytest
from pydantic import ValidationError

from app.config import Settings


def test_non_dev_requires_strong_jwt_secret():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="prod", jwt_secret="short")
    ok = Settings(_env_file=None, app_env="prod", jwt_secret="x" * 40)
    assert not ok.demo_enabled and not ok.is_dev


def test_empty_jwt_secret_never_used():
    s = Settings(_env_file=None, app_env="dev", jwt_secret="")
    assert s.jwt_secret  # không ký JWT bằng khoá rỗng
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="prod", jwt_secret="")


def test_database_url_overrides_postgres_parts():
    s = Settings(_env_file=None, database_url="sqlite:///x.db", postgres_host="db")
    assert s.sqlalchemy_url == "sqlite:///x.db"
    assert "@db:" in Settings(_env_file=None, database_url=None, postgres_host="db").sqlalchemy_url
