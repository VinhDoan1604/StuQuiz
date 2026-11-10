"""Các thay đổi rút ra từ eval full-v1 (xem eval/FOLLOWUP.md): bật/tắt CRAG + critic,
critic chấm theo từng khẳng định, model riêng cho critic, retrieval dense/không rerank,
prompt từ chối không gắn tiếng Việt. Không gọi mạng / model thật."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from app.agents import generation_agent as gen
from app.agents import reflection_agent as ra
from app.config import Settings, settings
from app.graph.build_graph import build_study_graph, route_after_generation, route_after_retrieval
from app.llm import router as llm_router
from app.retrieval import helpers


# --- Bật/tắt CRAG + critic -------------------------------------------------------
def test_crag_toggle(monkeypatch):
    monkeypatch.setattr(settings, "enable_crag", False)
    assert route_after_retrieval({"intent": "qa"}) == "generation"
    assert route_after_retrieval({"intent": "quiz"}) == "quiz_summary"
    monkeypatch.setattr(settings, "enable_crag", True)
    assert route_after_retrieval({"intent": "qa"}) == "grade_context"


def test_critic_toggle(monkeypatch):
    monkeypatch.setattr(settings, "enable_answer_critic", False)
    assert route_after_generation({}) == "accept_answer"
    monkeypatch.setattr(settings, "enable_answer_critic", True)
    assert route_after_generation({}) == "grade_answer"


def test_graph_with_everything_off_makes_one_llm_call(monkeypatch):
    monkeypatch.setattr(settings, "enable_crag", False)
    monkeypatch.setattr(settings, "enable_answer_critic", False)
    graph = build_study_graph()
    state = {"user_query": "Dijkstra là gì?", "intent": "qa", "num_reflection_loops": 0,
             "reflection_history": [], "rewritten_queries": [], "document_ids": ["d"]}
    with patch("app.graph.build_graph.run_orchestrator", new=AsyncMock(return_value={"intent": "qa"})), \
         patch("app.graph.build_graph.run_retrieval", new=AsyncMock(return_value={"retrieved_chunks": [{"text": "x"}]})), \
         patch("app.graph.build_graph.run_generation", new=AsyncMock(return_value={"answer_draft": "A [1]", "citations": []})) as g, \
         patch("app.graph.build_graph.grade_context", new=AsyncMock()) as gc, \
         patch("app.graph.build_graph.grade_answer", new=AsyncMock()) as ga:
        out = asyncio.run(graph.ainvoke(state))
    assert out["final_answer"] == "A [1]"
    assert g.await_count == 1 and gc.await_count == 0 and ga.await_count == 0


# --- Critic theo từng khẳng định ---------------------------------------------------
def _verdict(n_claims: int, n_unsupported: int, completeness: float = 1.0) -> ra.AnswerGradeOutput:
    claims = [ra.CriticClaim(claim=f"ý {i}", supported=i >= n_unsupported) for i in range(n_claims)]
    return ra.AnswerGradeOutput(abstained=False, claims=claims, completeness=completeness, feedback="")


def _grade(verdict, loops=0):
    state = {"user_query": "q", "answer_draft": "trả lời", "retrieved_chunks": [{"text": "ctx"}],
             "num_reflection_loops": loops}
    with patch("app.agents.reflection_agent.acall_structured", new=AsyncMock(return_value=verdict)) as m:
        res = asyncio.run(ra.grade_answer(state))
    assert m.await_args.args[0] == "critic"
    return res


def test_critic_rejects_two_unsupported_claims_and_lists_them():
    res = _grade(_verdict(10, 2))
    assert res["answer_faithful"] is False
    assert '"ý 0"' in res["feedback"] and '"ý 1"' in res["feedback"]


def test_critic_accepts_one_stray_claim_in_ten():
    assert _grade(_verdict(10, 1))["answer_faithful"] is True


def test_critic_abstention_without_claims_is_faithful():
    v = ra.AnswerGradeOutput(abstained=True, claims=[], completeness=1.0, feedback="")
    assert _grade(v)["answer_faithful"] is True


def test_critic_forced_accept_when_loops_exhausted():
    res = _grade(_verdict(4, 3), loops=settings.max_reflection_loops)
    assert res["answer_faithful"] is True and res["forced"] is True


def test_critic_model_override(monkeypatch):
    monkeypatch.setattr(settings, "shopaikey_api_key", "sk-test")
    monkeypatch.setattr(settings, "cheap_model_provider", "shopaikey_google")
    monkeypatch.setattr(settings, "cheap_model_name", "gemini-2.5-flash")
    llm_router._build_llm.cache_clear()
    try:
        assert llm_router.get_llm("critic").model == "gemini-2.5-flash"  # chưa cấu hình => cheap
        monkeypatch.setattr(settings, "critic_model_provider", "shopaikey_openai")
        monkeypatch.setattr(settings, "critic_model_name", "gpt-5-mini")
        assert llm_router.get_llm("critic").model_name == "gpt-5-mini"
        assert llm_router.get_llm("grade").model == "gemini-2.5-flash"  # chấm context không đổi
    finally:
        llm_router._build_llm.cache_clear()


# --- Prompt ---------------------------------------------------------------------------
def test_refusal_rule_not_tied_to_vietnamese():
    # Câu mẫu tiếng Việt cũ bị chép nguyên văn vào 3 câu trả lời cho câu hỏi tiếng Anh (full-v1).
    assert "Tài liệu chưa cung cấp dữ kiện về" not in gen._QA_SYSTEM_PROMPT
    assert "do not cover" in gen._EN_ANSWER_DIRECTIVE


def test_notes_section_restricted_to_context():
    line = next(x for x in gen._QA_SYSTEM_PROMPT.splitlines() if "💡" in x and "Điểm cần lưu ý" in x)
    assert "CONTEXT" in line and "BỎ HẲN" in line


# --- Retrieval mode / rerank -----------------------------------------------------------
def test_retrieval_mode_validation():
    assert Settings(_env_file=None, retrieval_mode=" Dense ").retrieval_mode == "dense"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieval_mode="bm25")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rerank_candidates=0)


def _run_retrieve(monkeypatch, mode, rerank_enabled, candidates=10):
    monkeypatch.setattr(settings, "retrieval_mode", mode)
    monkeypatch.setattr(settings, "rerank_enabled", rerank_enabled)
    monkeypatch.setattr(settings, "rerank_candidates", candidates)
    cands = [{"text": f"c{i}"} for i in range(candidates)]
    with patch.object(helpers, "aembed_query", new=AsyncMock(return_value={"dense": [0.1], "sparse": {}})), \
         patch.object(helpers, "adense_search", new=AsyncMock(return_value=cands)) as dense, \
         patch.object(helpers, "ahybrid_search", new=AsyncMock(return_value=cands)) as hybrid, \
         patch.object(helpers, "arerank", new=AsyncMock(return_value=cands[::-1][:5])) as rr:
        out = asyncio.run(helpers.retrieve_top_k("q", top_k=5, document_ids=["d"]))
    return out, dense, hybrid, rr


def test_retrieve_dense_mode_with_rerank(monkeypatch):
    out, dense, hybrid, rr = _run_retrieve(monkeypatch, "dense", True)
    assert dense.await_count == 1 and hybrid.await_count == 0 and rr.await_count == 1
    assert dense.await_args.kwargs["top_k"] == 10  # RERANK_CANDIDATES
    assert out[0]["text"] == "c9"


def test_retrieve_without_rerank_only_fetches_top_k(monkeypatch):
    out, dense, hybrid, rr = _run_retrieve(monkeypatch, "hybrid", False)
    assert hybrid.await_count == 1 and rr.await_count == 0
    assert hybrid.await_args.kwargs["top_k"] == 5
    assert [c["text"] for c in out] == ["c0", "c1", "c2", "c3", "c4"]
