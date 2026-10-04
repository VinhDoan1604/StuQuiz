import os
import sys
import uuid
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, MagicMock

try:
    import pytest
    fixture_dec = pytest.fixture
except ImportError:
    pytest = None
    def fixture_dec(fn):
        return fn

from app.db.session import Base
from app.models.db_models import User, ChatSession, ChatTurn, MasteryRecord
from app.models.schemas import MasteryTopic
from app.memory.mastery_tracker import (
    record_quiz_result,
    get_weak_topics,
    _next_interval_days,
    _get_or_create_record,
    _SPACED_REPETITION_SCHEDULE_DAYS,
)
from app.memory.context_window import (
    build_context,
    maybe_summarize,
    _covered_turn_count,
    SLIDING_WINDOW_TURNS,
    SUMMARIZE_TRIGGER_TURNS,
)


@fixture_dec
def db_session():
    """Tạo session SQLite in-memory để test hoàn toàn độc lập, không phụ thuộc vào Postgres bên ngoài."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_next_interval_days():
    assert _next_interval_days(1) == 1
    assert _next_interval_days(2) == 3
    assert _next_interval_days(3) == 7
    assert _next_interval_days(4) == 14
    assert _next_interval_days(10) == 14  # Max clamp


def test_mastery_tracker_streak_and_spaced_repetition(db_session):
    user_id = str(uuid.uuid4())
    topic = "Thuật toán Dijkstra"

    # Lần 1: Đúng -> streak=1, interval=1 ngày, score=10
    record_quiz_result(db_session, user_id, topic, is_correct=True)
    rec = _get_or_create_record(db_session, user_id, topic)
    assert rec.streak == 1
    assert rec.mastery_score == 10
    assert rec.correct_count == 1
    assert rec.incorrect_count == 0
    assert rec.next_review_due_at is not None
    # Khoảng cách khoảng 1 ngày
    delta = (rec.next_review_due_at - rec.last_reviewed_at).total_seconds()
    assert abs(delta - 86400) < 60

    # Lần 2: Đúng tiếp -> streak=2, interval=3 ngày, score=20
    record_quiz_result(db_session, user_id, topic, is_correct=True)
    rec = _get_or_create_record(db_session, user_id, topic)
    assert rec.streak == 2
    assert rec.mastery_score == 20
    assert rec.correct_count == 2
    delta = (rec.next_review_due_at - rec.last_reviewed_at).total_seconds()
    assert abs(delta - 3 * 86400) < 60

    # Lần 3: Đúng tiếp -> streak=3, interval=7 ngày, score=30
    record_quiz_result(db_session, user_id, topic, is_correct=True)
    rec = _get_or_create_record(db_session, user_id, topic)
    assert rec.streak == 3
    assert rec.mastery_score == 30
    delta = (rec.next_review_due_at - rec.last_reviewed_at).total_seconds()
    assert abs(delta - 7 * 86400) < 60

    # Lần 4: Đúng tiếp -> streak=4, interval=14 ngày, score=40
    record_quiz_result(db_session, user_id, topic, is_correct=True)
    rec = _get_or_create_record(db_session, user_id, topic)
    assert rec.streak == 4
    assert rec.mastery_score == 40
    delta = (rec.next_review_due_at - rec.last_reviewed_at).total_seconds()
    assert abs(delta - 14 * 86400) < 60

    # Lần 5: SAI -> streak reset về 0, interval=1 ngày, score bị phạt -15 -> 25
    record_quiz_result(db_session, user_id, topic, is_correct=False)
    rec = _get_or_create_record(db_session, user_id, topic)
    assert rec.streak == 0
    assert rec.mastery_score == 25
    assert rec.incorrect_count == 1
    delta = (rec.next_review_due_at - rec.last_reviewed_at).total_seconds()
    assert abs(delta - 86400) < 60


def test_mastery_tracker_clamping(db_session):
    user_id = str(uuid.uuid4())
    topic = "Quy hoạch động"

    # Test clamp MIN 0
    record_quiz_result(db_session, user_id, topic, is_correct=False)
    rec = _get_or_create_record(db_session, user_id, topic)
    assert rec.mastery_score == 0

    # Test clamp MAX 100
    for _ in range(12):
        record_quiz_result(db_session, user_id, topic, is_correct=True)
    rec = _get_or_create_record(db_session, user_id, topic)
    assert rec.mastery_score == 100
    assert rec.streak == 12


def test_get_weak_topics(db_session):
    user_id = str(uuid.uuid4())
    record_quiz_result(db_session, user_id, "Topic_Weak_1", is_correct=False)
    record_quiz_result(db_session, user_id, "Topic_Weak_2", is_correct=True)  # score=10

    weak = get_weak_topics(db_session, user_id, limit=5)
    assert len(weak) == 2
    assert all(isinstance(w, MasteryTopic) for w in weak)
    assert hasattr(weak[0], "streak")


def test_context_window_incremental(db_session):
    session_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    chat_session = ChatSession(id=session_id, user_id=user_id, summarized_turn_count=0)
    db_session.add(chat_session)
    db_session.commit()

    # Thêm 12 turns (đủ điều kiện trigger vì 12 - 0 >= SUMMARIZE_TRIGGER_TURNS=10)
    for i in range(12):
        turn = ChatTurn(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"Message {i}",
            created_at=datetime.utcnow() + timedelta(seconds=i),
        )
        db_session.add(turn)
    db_session.commit()

    assert _covered_turn_count(chat_session) == 0

    # Mock LLM call trong maybe_summarize
    with patch("app.llm.router.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Bản tóm tắt 6 turns đầu tiên."
        mock_llm.invoke.return_value = mock_resp
        mock_get_llm.return_value = mock_llm

        maybe_summarize(db_session, session_id)

    db_session.refresh(chat_session)
    # Total=12, SLIDING_WINDOW=6, offset=0 -> gộp 12 - 6 - 0 = 6 turns
    assert chat_session.summarized_turn_count == 6
    assert chat_session.rolling_summary == "Bản tóm tắt 6 turns đầu tiên."
    assert _covered_turn_count(chat_session) == 6

    # Test build_context
    ctx = build_context(db_session, session_id, token_budget=1000)
    assert "[Tóm tắt hội thoại trước đó]" in ctx
    assert "Bản tóm tắt 6 turns đầu tiên." in ctx
    assert "[Lịch sử gần đây]" in ctx


if __name__ == "__main__":
    print("--- Running tests for Mastery Tracker & Context Window ---")
    test_next_interval_days()
    print("[PASS] test_next_interval_days")

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = TestingSessionLocal()
    test_mastery_tracker_streak_and_spaced_repetition(session)
    print("[PASS] test_mastery_tracker_streak_and_spaced_repetition")
    session.close()

    session = TestingSessionLocal()
    test_mastery_tracker_clamping(session)
    print("[PASS] test_mastery_tracker_clamping")
    session.close()

    session = TestingSessionLocal()
    test_get_weak_topics(session)
    print("[PASS] test_get_weak_topics")
    session.close()

    session = TestingSessionLocal()
    test_context_window_incremental(session)
    print("[PASS] test_context_window_incremental")
    session.close()

    print("\n=> ALL MEMORY TESTS PASSED 100%!")
