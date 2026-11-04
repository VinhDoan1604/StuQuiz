"""Chủ đề quiz theo mục tài liệu + ưu tiên spaced repetition."""
import asyncio
import uuid
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import app.models.db_models as m
from app.agents.quiz_summary_agent import generate_quiz
from app.agents.quiz_topics import build_section_topics, clean_heading, is_meaningful_heading, prioritize_topics
from app.models.schemas import QuizQuestion
from app.utils.time import utcnow

LONG = "Nội dung giảng giải chi tiết về chủ đề này. " * 20   # > MIN_TOPIC_TOKENS
SHORT = "Bảng số liệu."


def src(doc, section, text=LONG):
    return {"doc_id": doc, "section": section, "text": text, "page": 1}


# ------------------------------------------------------------- heading filter
def test_clean_heading_strips_numbering():
    assert clean_heading("2.2 Domain Adaptation") == "Domain Adaptation"
    assert clean_heading("Chương 3: Đường đi ngắn nhất") == "Đường đi ngắn nhất"
    assert clean_heading("IV. Kết luận") == "Kết luận"


def test_noise_headings_rejected_real_ones_kept():
    for junk in ["T", "O(", "AUC (%)", "EAPIV (%)", "MMD", "CTIV (", "3.0 and then to 12.0. The peak of scores"]:
        assert not is_meaningful_heading(junk), junk
    for good in ["Methods", "3.1 Thuật toán Dijkstra", "GIỚI THIỆU CHUNG", "Infection-aware DRR generator"]:
        assert is_meaningful_heading(good), good


# ------------------------------------------------------------ topic building
def test_noise_subheading_merges_into_parent():
    topics = build_section_topics(
        [src("d1", "4. Results > 4.1 Setup"), src("d1", "4. Results > AUC (%)"), src("d1", "4. Results")],
        {"d1": "paper.pdf"},
    )
    labels = [t.label for t in topics]
    assert labels == ["Setup — paper", "Results — paper"]
    assert len(topics[1].chunks) == 2   # nội dung của "AUC (%)" thuộc về "Results"


def test_small_section_merges_up_and_no_heading_uses_doc_title():
    topics = build_section_topics(
        [src("d1", "(không có heading)"), src("d1", "1. Intro > 1.1 Tiny", SHORT), src("d1", "1. Intro")],
        {"d1": "notes.md"},
    )
    assert [t.label for t in topics] == ["notes", "Intro — notes"]


def test_duplicate_uploads_collapse_to_one_topic():
    topics = build_section_topics(
        [src("d1", "1. Intro"), src("d2", "1. Intro")], {"d1": "paper.pdf", "d2": "paper.pdf"}
    )
    assert len(topics) == 1 and len(topics[0].chunks) == 1


def test_prioritize_overdue_then_unseen_then_weak():
    now = utcnow()
    topics = build_section_topics(
        [src("d", f"{i}. Topic {name}") for i, name in enumerate(["Alpha", "Beta", "Gamma", "Delta"], 1)],
        {"d": "doc.pdf"},
    )
    rec = lambda score, due: SimpleNamespace(mastery_score=score, next_review_due_at=due)  # noqa: E731
    mastery = {
        "Topic Alpha — doc": rec(90, now + timedelta(days=7)),   # đã vững, chưa tới hạn
        "Topic Beta — doc": rec(30, now + timedelta(days=1)),    # yếu, chưa tới hạn
        "Topic Delta — doc": rec(80, now - timedelta(days=2)),   # quá hạn
    }
    order = [t.label for t in prioritize_topics(topics, mastery, now, 60)]
    assert order == ["Topic Delta — doc", "Topic Gamma — doc", "Topic Beta — doc", "Topic Alpha — doc"]


# -------------------------------------------------------------- generate_quiz
def _fake_questions(chunks, topic, n, existing):
    return [
        QuizQuestion(question=f"{topic} Q{i}", choices=["A", "B", "C", "D"], correct_choice="A",
                     explanation="e", topic=topic, difficulty="easy")
        for i in range(n)
    ]


def test_generate_quiz_uses_document_sections(db_session):
    user, doc = str(uuid.uuid4()), str(uuid.uuid4())
    db_session.add(m.Document(id=doc, owner_id=user, filename="graph.md", status="ready"))
    for i, sec in enumerate(["1. Dijkstra", "2. Bellman-Ford", "3. Ứng dụng"]):
        db_session.add(m.ParentChunk(id=f"p{i}", doc_id=doc, text=LONG, section=sec, page=None))
    # Bellman-Ford đã quá hạn ôn => phải được ra đề trước
    db_session.add(m.MasteryRecord(user_id=user, topic="Bellman-Ford — graph", mastery_score=70, streak=1,
                                   next_review_due_at=utcnow() - timedelta(days=1)))
    db_session.commit()

    async def fake(*args):
        return _fake_questions(*args)

    with patch("app.agents.quiz_summary_agent._generate_quiz_questions", side_effect=fake):
        out = asyncio.run(generate_quiz(db_session, user, [doc], num_questions=2))

    assert [q.topic for q in out] == ["Bellman-Ford — graph", "Dijkstra — graph"]
    stored = {a.topic for a in db_session.query(m.QuizAttempt).filter(m.QuizAttempt.user_id == user)}
    assert stored == {"Bellman-Ford — graph", "Dijkstra — graph"}


def test_generate_quiz_without_content_returns_empty(db_session):
    user, doc = str(uuid.uuid4()), str(uuid.uuid4())
    db_session.add(m.Document(id=doc, owner_id=user, filename="empty.pdf", status="ready"))
    db_session.commit()

    async def no_chunks(_ids):
        return []

    with patch("app.agents.quiz_summary_agent.aget_all_chunks_for_documents", side_effect=no_chunks):
        assert asyncio.run(generate_quiz(db_session, user, [doc], num_questions=3)) == []
