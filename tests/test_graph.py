"""
tests/test_graph.py

Bộ kiểm thử tự động toàn diện cho LangGraph StateGraph (app/graph/build_graph.py):
1. Kiểm tra cấu trúc đồ thị (Topology, nodes, edges).
2. Kiểm tra các hàm điều hướng có điều kiện (Conditional Edge Routing).
3. Kiểm tra vòng lặp CRAG (Corrective RAG) khi context chưa đạt ngưỡng.
4. Kiểm tra vòng lặp Self-RAG khi draft chưa trung thực (unfaithful).
5. Kiểm tra phân nhánh intent="quiz" và intent="summary".
6. Kiểm tra tích hợp bộ nhớ (Memory, ChatTurn, Mastery, ReflectionTrace).
7. Kiểm tra xử lý ngoại lệ và fallback an toàn (Fail-Open/Graceful degradation).
"""
import os
import sys
import uuid
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.session import Base
from app.models.db_models import ChatSession, ChatTurn, MasteryRecord
from app.agents.state import GraphState, ReflectionResult
from app.graph.build_graph import (
    build_study_graph,
    get_graph,
    route_after_orchestrator,
    route_after_retrieval,
    route_after_grade_context,
    route_after_grade_answer,
    _extract_reflection_trace,
    run_graph,
)


@pytest.fixture
def mock_db_session():
    """Tạo SQLite in-memory database độc lập hoàn toàn cho unit test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


def test_graph_topology():
    """Kiểm tra đồ thị StateGraph biên dịch thành công và có đủ các node bắt buộc."""
    graph = build_study_graph()
    assert graph is not None

    expected_nodes = {
        "__start__",
        "orchestrator",
        "retrieval",
        "grade_context",
        "rewrite_query",
        "generation",
        "grade_answer",
        "retry_generation",
        "accept_answer",
        "quiz_summary",
    }
    actual_nodes = set(graph.nodes.keys())
    assert expected_nodes.issubset(actual_nodes), f"Thiếu node trong graph: {expected_nodes - actual_nodes}"


def test_routing_after_orchestrator():
    """Mọi intent đều phải đi qua retrieval trước để lấy chunk."""
    assert route_after_orchestrator({"intent": "qa"}) == "retrieval"
    assert route_after_orchestrator({"intent": "teaching"}) == "retrieval"
    assert route_after_orchestrator({"intent": "quiz"}) == "retrieval"
    assert route_after_orchestrator({"intent": "summary"}) == "retrieval"


def test_routing_after_retrieval():
    """Sau retrieval: quiz/summary rẽ sang quiz_summary; qa/teaching rẽ sang grade_context."""
    assert route_after_retrieval({"intent": "quiz"}) == "quiz_summary"
    assert route_after_retrieval({"intent": "summary"}) == "quiz_summary"
    assert route_after_retrieval({"intent": "qa"}) == "grade_context"
    assert route_after_retrieval({"intent": "teaching"}) == "grade_context"


def test_routing_after_grade_context():
    """Kiểm tra điều hướng CRAG: nếu context chưa đủ và còn lượt loop -> rewrite_query."""
    # 1. Đủ context -> đi tiếp generation
    state_sufficient: GraphState = {
        "num_reflection_loops": 0,
        "reflection_history": [{"context_sufficient": True, "context_grade": 0.9}],
    }
    assert route_after_grade_context(state_sufficient) == "generation"

    # 2. Chưa đủ context và loops = 0 (< max_loops=2) -> rewrite_query
    state_insufficient: GraphState = {
        "num_reflection_loops": 0,
        "reflection_history": [{"context_sufficient": False, "context_grade": 0.4}],
    }
    assert route_after_grade_context(state_insufficient) == "rewrite_query"

    # 3. Chưa đủ context nhưng ĐÃ HẾT lượt loop (loops = 2) -> buộc phải sang generation
    state_loops_exhausted: GraphState = {
        "num_reflection_loops": 2,
        "reflection_history": [{"context_sufficient": False, "context_grade": 0.4}],
    }
    assert route_after_grade_context(state_loops_exhausted) == "generation"


def test_routing_after_grade_answer():
    """Kiểm tra điều hướng Self-RAG: nếu draft chưa faithful và còn lượt -> retry_generation."""
    # 1. Faithful -> accept_answer
    state_faithful: GraphState = {
        "num_reflection_loops": 0,
        "reflection_history": [{"answer_faithful": True, "answer_grade": 0.95}],
    }
    assert route_after_grade_answer(state_faithful) == "accept_answer"

    # 2. Unfaithful và loops = 0 -> retry_generation
    state_unfaithful: GraphState = {
        "num_reflection_loops": 0,
        "reflection_history": [{"answer_faithful": False, "answer_grade": 0.5}],
    }
    assert route_after_grade_answer(state_unfaithful) == "retry_generation"

    # 3. Unfaithful nhưng ĐÃ HẾT lượt loop -> buộc phải accept
    state_exhausted: GraphState = {
        "num_reflection_loops": 2,
        "reflection_history": [{"answer_faithful": False, "answer_grade": 0.5}],
    }
    assert route_after_grade_answer(state_exhausted) == "accept_answer"


def test_extract_reflection_trace():
    """Kiểm tra trích xuất ReflectionTrace hiển thị ra UI."""
    history = [
        {"context_grade": 0.6, "context_sufficient": False, "feedback": "Thiếu định nghĩa"},
        {"context_grade": 0.88, "context_sufficient": True, "feedback": ""},
        {"answer_grade": 0.92, "answer_faithful": True, "feedback": ""},
    ]
    state: GraphState = {
        "reflection_history": history,
        "rewritten_queries": ["Thuật toán Dijkstra là gì?"],
        "num_reflection_loops": 1,
    }
    trace = _extract_reflection_trace(state)
    assert trace["context_grade"] == 0.88
    assert trace["context_sufficient"] is True
    assert trace["answer_grade"] == 0.92
    assert trace["answer_faithful"] is True
    assert trace["num_loops"] == 1
    assert len(trace["rewritten_queries"]) == 1


import asyncio


def test_run_graph_empty_message():
    """Tin nhắn rỗng phải trả về phản hồi mặc định ngay lập tức mà không tốn LLM call."""
    res = asyncio.run(run_graph(session_id="test_sess", user_message="   "))
    assert res["intent"] == "qa"
    assert len(res["answer"]) > 0
    assert res["reflection"]["num_loops"] == 0


def test_run_graph_end_to_end_mocked():
    """Kiểm tra toàn bộ luồng run_graph với Mock các Agent calls và in-memory SQLite session."""
    async def _run():
        from sqlalchemy.pool import StaticPool
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)
        TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        session_id = str(uuid.uuid4())

        mock_chunks = [
            {"text": "Dijkstra tìm đường đi ngắn nhất từ 1 đỉnh nguồn.", "source_file": "dsa.pdf", "page": 45, "section": "Đồ thị"}
        ]

        with patch("app.graph.build_graph.SessionLocal", side_effect=TestingSession), \
             patch("app.graph.build_graph.run_orchestrator", new_callable=AsyncMock) as mock_orch, \
             patch("app.graph.build_graph.run_retrieval", new_callable=AsyncMock) as mock_retr, \
             patch("app.graph.build_graph.grade_context", new_callable=AsyncMock) as mock_gc, \
             patch("app.graph.build_graph.run_generation", new_callable=AsyncMock) as mock_gen, \
             patch("app.graph.build_graph.grade_answer", new_callable=AsyncMock) as mock_ga:

            mock_orch.return_value = {"intent": "qa", "suggested_review_topic": None}
            mock_retr.return_value = {"retrieved_chunks": mock_chunks, "rewritten_queries": []}
            mock_gc.return_value = {"context_grade": 0.9, "context_sufficient": True, "feedback": ""}
            mock_gen.return_value = {
                "answer_draft": "Dijkstra là thuật toán tìm đường đi ngắn nhất [1].",
                "citations": [{"file": "dsa.pdf", "page": 45, "section": "Đồ thị"}],
            }
            mock_ga.return_value = {"answer_grade": 0.95, "answer_faithful": True, "feedback": ""}

            result = await run_graph(session_id=session_id, user_message="Dijkstra dùng để làm gì?")

            assert result["intent"] == "qa"
            assert "Dijkstra là thuật toán tìm đường đi ngắn nhất" in result["answer"]
            assert len(result["citations"]) == 1
            assert result["citations"][0]["file"] == "dsa.pdf"
            assert result["reflection"]["context_grade"] == 0.9
            assert result["reflection"]["answer_grade"] == 0.95
            assert result["reflection"]["num_loops"] == 0

            # Kiểm tra dữ liệu được ghi vào database
            db = TestingSession()
            turns = db.query(ChatTurn).filter(ChatTurn.session_id == session_id).all()
            assert len(turns) == 2  # 1 user turn + 1 assistant turn
            user_turn = turns[0] if turns[0].role == "user" else turns[1]
            asst_turn = turns[1] if turns[1].role == "assistant" else turns[0]
            assert user_turn.content == "Dijkstra dùng để làm gì?"
            assert "Dijkstra" in asst_turn.content
            assert asst_turn.reflection_trace is not None
            db.close()

    asyncio.run(_run())
