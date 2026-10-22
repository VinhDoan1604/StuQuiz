# STATUS: READY
"""
State schema dùng chung cho toàn bộ LangGraph StateGraph (mục 5.12 tài
liệu gốc). Đây là "hợp đồng dữ liệu" giữa các node/agent — nên định nghĩa
rõ ràng trước, còn logic từng agent implement dần trong app/agents/*.
"""
from typing import TypedDict, Literal, Any


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
    document_ids: list[str] | None

    # Orchestrator output
    intent: Literal["qa", "teaching", "quiz", "summary"]
    suggested_review_topic: str | None

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

    # Memory
    chat_history_summary: str
    mastery_context: dict[str, Any]
