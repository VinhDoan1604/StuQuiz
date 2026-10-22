"""Luồng quiz: chấm điểm phía server, mastery, lịch sử. pytest tests/test_quiz_flow.py -v"""
import uuid

import app.models.db_models as m
from app.models.schemas import QuizQuestionOut


def make_attempt(client, user_id, space_id=None, choices=None, correct="B", topic="Dijkstra"):
    with client.Session() as db:
        a = m.QuizAttempt(
            id=str(uuid.uuid4()), user_id=user_id, space_id=space_id, topic=topic, question="Q?",
            choices=choices or ["A", "B", "C", "D"], correct_choice=correct,
            explanation="vì B đúng", difficulty="easy",
        )
        db.add(a)
        db.commit()
        return str(a.id)


def mastery(client, user_id):
    with client.Session() as db:
        return db.query(m.MasteryRecord).filter(m.MasteryRecord.user_id == user_id).all()


def test_generate_schema_never_leaks_answer():
    assert "correct_choice" not in QuizQuestionOut.model_fields
    assert "explanation" not in QuizQuestionOut.model_fields
    assert "attempt_id" in QuizQuestionOut.model_fields


def test_correct_answer_graded_by_server_and_updates_mastery(client, make_user):
    uid, h = make_user("a@x.com")
    aid = make_attempt(client, uid)
    r = client.post("/quiz/answer", json={"attempt_id": aid, "choice": "B"}, headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["is_correct"] is True and body["correct_choice"] == "B" and body["explanation"] == "vì B đúng"
    rec = mastery(client, uid)
    assert len(rec) == 1 and rec[0].correct_count == 1 and rec[0].mastery_score == 10 and rec[0].streak == 1


def test_wrong_answer_recorded(client, make_user):
    uid, h = make_user("a@x.com")
    aid = make_attempt(client, uid)
    r = client.post("/quiz/answer", json={"attempt_id": aid, "choice": "C"}, headers=h)
    assert r.json()["is_correct"] is False and r.json()["correct_choice"] == "B"
    rec = mastery(client, uid)[0]
    assert rec.incorrect_count == 1 and rec.streak == 0


def test_resubmit_is_idempotent_and_cannot_change_answer(client, make_user):
    uid, h = make_user("a@x.com")
    aid = make_attempt(client, uid)
    client.post("/quiz/answer", json={"attempt_id": aid, "choice": "C"}, headers=h)  # sai
    r = client.post("/quiz/answer", json={"attempt_id": aid, "choice": "B"}, headers=h)  # thử đổi sang đúng
    assert r.status_code == 200 and r.json()["is_correct"] is False
    rec = mastery(client, uid)[0]
    assert rec.incorrect_count == 1 and rec.correct_count == 0  # không cộng/trừ lần 2


def test_choice_must_belong_to_question(client, make_user):
    uid, h = make_user("a@x.com")
    aid = make_attempt(client, uid)
    r = client.post("/quiz/answer", json={"attempt_id": aid, "choice": "Z"}, headers=h)
    assert r.status_code == 400
    assert mastery(client, uid) == []


def test_cannot_answer_other_users_attempt(client, make_user):
    a_id, _ = make_user("a@x.com")
    _, hb = make_user("b@x.com")
    aid = make_attempt(client, a_id)
    assert client.post("/quiz/answer", json={"attempt_id": aid, "choice": "B"}, headers=hb).status_code == 404
    assert client.post("/quiz/answer", json={"attempt_id": "not-a-uuid", "choice": "B"}, headers=hb).status_code == 404


def test_client_reported_mastery_endpoint_removed(client, make_user):
    _, h = make_user("a@x.com")
    r = client.post("/mastery/record", json={"topic": "x", "is_correct": True}, headers=h)
    assert r.status_code in (404, 405)


def test_history_shows_only_answered_attempts(client, make_user):
    uid, h = make_user("a@x.com")
    sid = client.post("/spaces", json={"title": "S"}, headers=h).json()["id"]
    aid = make_attempt(client, uid, space_id=sid)
    make_attempt(client, uid, space_id=sid)  # câu chưa trả lời
    assert client.get(f"/spaces/{sid}/quiz-history", headers=h).json() == []

    client.post("/quiz/answer", json={"attempt_id": aid, "choice": "B"}, headers=h)
    hist = client.get(f"/spaces/{sid}/quiz-history", headers=h).json()
    assert len(hist) == 1 and hist[0]["is_correct"] is True
