"""
app/agents/__init__.py

Multi-Agent layer cho StudyRAG:
- Orchestrator Agent (phân loại intent, gợi ý ôn tập)
- Retrieval Agent (CRAG retrieval loop, query rewrite/decompose)
- Generation Agent (QA mode có citation, Teaching mode scaffolding)
- Reflection Agent (chấm context và faithfulness)
- Quiz & Summary Agent (sinh trắc nghiệm & tóm tắt)
- State & Constants
"""
from app.agents.state import GraphState, ReflectionResult
from app.agents.constants import CONTEXT_SUFFICIENT_THRESHOLD, ANSWER_ACCEPT_THRESHOLD

__all__ = [
    "GraphState",
    "ReflectionResult",
    "CONTEXT_SUFFICIENT_THRESHOLD",
    "ANSWER_ACCEPT_THRESHOLD",
]
