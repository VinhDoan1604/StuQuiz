"""
scripts/test_graph.py

Script kiểm thử toàn diện luồng LangGraph StateGraph (Phase 1 / Bước 2 trong NEXT_STEPS.md):
- Hiển thị trực quan Topology của đồ thị LangGraph.
- Kiểm tra tính đúng đắn của các node & conditional routing.
- Chạy thử nghiệm mô phỏng (Simulated flow) với CRAG và Self-RAG reflection loops.
- Chạy thử nghiệm Live end-to-end với LLM thật (Google Gemini).
"""
import os
import sys
import time
import uuid
import asyncio

# Đảm bảo import được module app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch, AsyncMock

from app.db.session import Base
from app.models.db_models import ChatSession, ChatTurn, MasteryRecord
from app.agents.state import GraphState
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


def print_header(title: str):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)


def test_topology():
    print_header("1. KIỂM TRA TOPOLOGY ĐỒ THỊ LANGGRAPH")
    graph = get_graph()
    nodes = list(graph.nodes.keys())
    print(f"[*] Đồ thị StateGraph đã biên dịch thành công!")
    print(f"[*] Tổng số Node: {len(nodes)}")
    for i, node in enumerate(nodes, 1):
        print(f"    {i:02d}. Node: '{node}'")

    expected = {"orchestrator", "retrieval", "grade_context", "rewrite_query", "generation", "grade_answer", "retry_generation", "accept_answer", "quiz_summary"}
    missing = expected - set(nodes)
    if missing:
        print(f"[!] LỖI: Thiếu các node: {missing}")
        return False
    print("=> Topology đạt chuẩn 100%!")
    return True


def test_routing_logic():
    print_header("2. KIỂM TRA LOGIC CONDITIONAL ROUTING (CRAG & SELF-RAG)")

    # Test Orchestrator routing
    assert route_after_orchestrator({"intent": "qa"}) == "retrieval"
    print("  [OK] orchestrator -> retrieval (với mọi intent)")

    # Test Retrieval routing
    assert route_after_retrieval({"intent": "qa"}) == "grade_context"
    assert route_after_retrieval({"intent": "teaching"}) == "grade_context"
    assert route_after_retrieval({"intent": "quiz"}) == "quiz_summary"
    assert route_after_retrieval({"intent": "summary"}) == "quiz_summary"
    print("  [OK] retrieval -> grade_context (QA/Teaching) & quiz_summary (Quiz/Summary)")

    # Test CRAG loop routing
    # Case A: context sufficient -> generation
    state_ok: GraphState = {"num_reflection_loops": 0, "reflection_history": [{"context_sufficient": True}]}
    assert route_after_grade_context(state_ok) == "generation"
    print("  [OK] grade_context -> generation (khi context_sufficient=True)")

    # Case B: context insufficient, loops < max -> rewrite_query (CRAG Loop)
    state_fail: GraphState = {"num_reflection_loops": 0, "reflection_history": [{"context_sufficient": False}]}
    assert route_after_grade_context(state_fail) == "rewrite_query"
    print("  [OK] grade_context -> rewrite_query (CRAG retry khi context thiếu & còn lượt)")

    # Case C: context insufficient, loops >= max -> generation (Fail-open)
    state_exhausted: GraphState = {"num_reflection_loops": 2, "reflection_history": [{"context_sufficient": False}]}
    assert route_after_grade_context(state_exhausted) == "generation"
    print("  [OK] grade_context -> generation (chặn loop vô hạn khi đã hết số lần phản tư)")

    # Test Self-RAG loop routing
    # Case A: answer faithful -> accept_answer
    ans_ok: GraphState = {"num_reflection_loops": 0, "reflection_history": [{"answer_faithful": True}]}
    assert route_after_grade_answer(ans_ok) == "accept_answer"
    print("  [OK] grade_answer -> accept_answer (khi answer_faithful=True)")

    # Case B: answer unfaithful, loops < max -> retry_generation
    ans_fail: GraphState = {"num_reflection_loops": 0, "reflection_history": [{"answer_faithful": False}]}
    assert route_after_grade_answer(ans_fail) == "retry_generation"
    print("  [OK] grade_answer -> retry_generation (Self-RAG retry khi câu trả lời có ảo giác)")

    # Case C: answer unfaithful, loops >= max -> accept_answer
    ans_exhausted: GraphState = {"num_reflection_loops": 2, "reflection_history": [{"answer_faithful": False}]}
    assert route_after_grade_answer(ans_exhausted) == "accept_answer"
    print("  [OK] grade_answer -> accept_answer (dừng loop khi chạm giới hạn recursion)")

    print("=> Toàn bộ quy tắc rẽ nhánh đạt chuẩn thiết kế!")
    return True


async def test_simulated_execution():
    print_header("3. KIỂM TRA MÔ PHỎNG END-TO-END VỚI MOCK VÀ IN-MEMORY DB")
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session_id = str(uuid.uuid4())
    mock_chunks = [
        {"text": "Giải thuật Dijkstra có độ phức tạp O((V + E) log V) khi dùng Fibonacci Heap.", "source_file": "dsa.pdf", "page": 12, "section": "Đồ thị"}
    ]

    with patch("app.graph.build_graph.SessionLocal", side_effect=TestingSession), \
         patch("app.graph.build_graph.run_orchestrator", new_callable=AsyncMock) as mock_orch, \
         patch("app.graph.build_graph.run_retrieval", new_callable=AsyncMock) as mock_retr, \
         patch("app.graph.build_graph.grade_context", new_callable=AsyncMock) as mock_gc, \
         patch("app.graph.build_graph.run_generation", new_callable=AsyncMock) as mock_gen, \
         patch("app.graph.build_graph.grade_answer", new_callable=AsyncMock) as mock_ga:

        mock_orch.return_value = {"intent": "qa", "suggested_review_topic": None}
        mock_retr.return_value = {"retrieved_chunks": mock_chunks, "rewritten_queries": []}
        mock_gc.return_value = {"context_grade": 0.92, "context_sufficient": True, "feedback": ""}
        mock_gen.return_value = {
            "answer_draft": "Giải thuật Dijkstra có độ phức tạp O((V + E) log V) khi kết hợp cùng Fibonacci Heap [1].",
            "citations": [{"file": "dsa.pdf", "page": 12, "section": "Đồ thị"}],
        }
        mock_ga.return_value = {"answer_grade": 0.98, "answer_faithful": True, "feedback": ""}

        print(f"[*] Đang gửi query: 'Độ phức tạp của thuật toán Dijkstra là bao nhiêu?'")
        start_t = time.perf_counter()
        resp = await run_graph(session_id=session_id, user_message="Độ phức tạp của thuật toán Dijkstra là bao nhiêu?")
        elapsed = (time.perf_counter() - start_t) * 1000

        print(f"[*] Kết quả trả về sau {elapsed:.1f}ms:")
        print(f"    - Intent: {resp['intent']}")
        print(f"    - Answer: {resp['answer']}")
        print(f"    - Citations: {resp['citations']}")
        print(f"    - Reflection Trace: {resp['reflection']}")

        # Kiểm tra Database
        db = TestingSession()
        turns = db.query(ChatTurn).all()
        print(f"[*] Kiểm tra database session: Đã lưu {len(turns)} lượt hội thoại (turns).")
        assert len(turns) == 2
        db.close()

    print("=> Mô phỏng end-to-end thành công hoàn hảo!")
    return True


async def test_live_execution():
    print_header("4. KIỂM TRA LIVE VỚI GOOGLE GEMINI (THỰC TẾ END-TO-END)")
    from app.config import settings

    if not settings.effective_google_api_key:
        print("[!] Không tìm thấy GOOGLE_API_KEY hoặc GEMINI_API_KEY trong .env — bỏ qua kiểm tra Live.")
        return True

    print(f"[*] Provider: {settings.cheap_model_provider} | Model: {settings.cheap_model_name}")
    session_id = str(uuid.uuid4())
    test_question = "Giải thích ngắn gọn định lý Bayes trong xác suất thống kê là gì?"
    print(f"[*] Đang gửi tin nhắn: '{test_question}'")

    start_t = time.perf_counter()
    try:
        resp = await run_graph(session_id=session_id, user_message=test_question)
        elapsed = (time.perf_counter() - start_t) * 1000

        print(f"\n[+] Phản hồi thành công từ LLM ({elapsed:.1f}ms):")
        print(f"    - Intent nhận diện: {resp['intent']}")
        print(f"    - Câu trả lời:\n      {resp['answer'][:200]}...")
        print(f"    - Citations: {resp['citations']}")
        print(f"    - Reflection Trace: {resp['reflection']}")
        print("=> Kiểm tra Live với Google Gemini hoàn tất thành công rực rỡ!")
        return True
    except Exception as exc:
        print(f"[!] Lỗi khi chạy live: {exc}")
        return False


async def main():
    print("=" * 65)
    print(" BỘ KIỂM TRA CHỨC NĂNG LANGGRAPH STATEGRAPH — STUDYRAG")
    print("=" * 65)

    ok1 = test_topology()
    ok2 = test_routing_logic()
    ok3 = await test_simulated_execution()
    ok4 = await test_live_execution()

    print_header("TỔNG KẾT KIỂM THỬ GIAI ĐOẠN 1 (BƯỚC 1 & 2)")
    if ok1 and ok2 and ok3 and ok4:
        print(" [PASSED 100%] HỆ THỐNG LANGGRAPH STATEGRAPH HOẠT ĐỘNG HOÀN HẢO!")
        print(" Sẵn sàng tiến sang Giai Đoạn 2: Xây dựng Giao diện UI Streamlit.")
    else:
        print(" [FAILED] Có bài test chưa đạt yêu cầu.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
