"""
State schema dùng chung cho LangGraph StateGraph — "hợp đồng dữ liệu" giữa
các node/agent trong app/agents/* và app/graph/build_graph.py.
"""
from typing import Any, Literal, TypedDict


class ReflectionResult(TypedDict, total=False):
    context_grade: float
    context_sufficient: bool
    answer_grade: float
    answer_faithful: bool
    feedback: str
    graded: bool    # False = bộ chấm lỗi (fail-open), điểm là None
    forced: bool    # True = chưa đạt nhưng bị ép chấp nhận do hết lượt thử


class GraphState(TypedDict, total=False):
    # Input
    session_id: str
    user_query: str
    document_ids: list[str]   # phạm vi tài liệu đã xác thực (rỗng = không có tài liệu)

    # Orchestrator output
    intent: Literal["qa", "teaching", "quiz", "summary"]
    suggested_review_topic: str | None
    standalone_query: str     # câu hỏi đã tự đủ nghĩa (chỉ có khi phiên đã có lịch sử)

    # Retrieval
    rewritten_queries: list[str]
    retrieved_chunks: list[dict[str, Any]]

    # Reflection (hiển thị ra UI — xem ReflectionTrace trong schemas.py)
    reflection_history: list[ReflectionResult]
    num_reflection_loops: int

    # Generation
    answer_draft: str
    final_answer: str
    citations: list[dict[str, Any]]
    generation_failed: bool   # LLM lỗi: answer_draft là câu báo lỗi, không lưu vào lịch sử
    stream_tokens: bool       # True: phát từng token qua stream "custom" của LangGraph

    # Memory
    chat_history_summary: str
    mastery_context: dict[str, Any]
