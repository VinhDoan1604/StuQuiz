-- Phần 3: thêm cột cho quiz_attempts. (Bảng mới parent_chunks do create_all tự tạo:
-- chạy lại `python scripts/init_db.py`.) Postgres & SQLite đều chạy được.
ALTER TABLE quiz_attempts ADD COLUMN explanation TEXT;
ALTER TABLE quiz_attempts ADD COLUMN difficulty VARCHAR;
ALTER TABLE quiz_attempts ADD COLUMN answered_at TIMESTAMP;
