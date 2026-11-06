"""
eval/pipelines.py

Các cấu hình ablation, xếp từ đơn giản đến đầy đủ. Mỗi cấu hình chỉ thêm ĐÚNG MỘT
thành phần so với cấu hình liền trước:

  dense         Naive RAG = dense top-5 -> 1 lần sinh.            (BASELINE)
  hybrid        + sparse lexical, hợp nhất RRF (top-5).
  hybrid_rerank + cross-encoder reranker (hybrid top-20 -> rerank -> top-5).
  crag          + Retrieval Agent (decompose) + grade_context + viết lại truy vấn (CRAG).
  full          + Critic chấm faithfulness + sinh lại (Self-RAG)  = đồ thị đầy đủ.

Quy ước công bằng:
- Mọi cấu hình dùng CÙNG prompt sinh (generation_agent, intent="qa"), cùng top-5.
- Bỏ qua node Orchestrator ở mọi cấu hình (intent cố định "qa") để chỉ đo chất lượng RAG.
- Mỗi lần chạy: xoá cache embedding + reset bộ đếm LLM => latency/token không bị
  cấu hình trước "ăn sẵn". Chạy TUẦN TỰ (bộ đếm usage là global).
- crag/full tái dùng NGUYÊN các node và hàm routing của app/graph/build_graph.py
  (không viết lại logic) => đo đúng thứ đang chạy ở production.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from app.agents.generation_agent import _ERROR_FALLBACK_ANSWER, run_generation
from app.embedding.bge_m3_embedder import clear_embedding_cache
from app.llm.router import get_cumulative_usage, reset_usage_metrics
from app.retrieval.helpers import aembed_query, ahybrid_search, arerank, normalize_chunk

FINAL_TOP_K = 5
HYBRID_TOP_K = 20

RETRIEVAL_CONFIGS = ("dense", "hybrid", "hybrid_rerank", "parent")
GRAPH_CONFIGS = ("crag", "full")
ALL_CONFIGS = RETRIEVAL_CONFIGS + GRAPH_CONFIGS


@dataclass
class PipelineResult:
    answer: str = ""
    chunks: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    llm_calls: int = 0
    tokens: int = 0
    loops: int = 0
    rewritten_queries: list[str] = field(default_factory=list)
    error: str | None = None
    # Chunk con TRƯỚC khi mở rộng parent. Retrieval được chấm trên danh sách này để
    # mọi cấu hình so sánh cùng đơn vị (chunk 200-400 token); `chunks` là context
    # thật gửi LLM (có thể là parent tới 1500 token) và chỉ dùng cho chỉ số context.
    child_chunks: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval-only configs
# ---------------------------------------------------------------------------
def _dense_search(dense_vec: list[float], top_k: int, document_ids: list[str]) -> list[dict]:
    from app.indexing import qdrant_store as qs

    qs.ensure_collection()
    resp = qs.client.query_points(
        collection_name=qs.COLLECTION_NAME,
        query=dense_vec,
        using="dense",
        limit=top_k,
        query_filter=qs._doc_filter(document_ids),
        with_payload=True,
    )
    return [qs._format_point(p) for p in resp.points]


async def retrieve(config: str, question: str, document_ids: list[str]) -> tuple[list[dict], list[dict]]:
    """-> (context gửi LLM, chunk con trước khi mở rộng parent)."""
    vec = await aembed_query(question)
    if config == "dense":
        hits = await asyncio.to_thread(_dense_search, vec["dense"], FINAL_TOP_K, document_ids)
        chunks = [normalize_chunk(h) for h in hits]
        return chunks, chunks
    elif config == "hybrid":
        hits = await ahybrid_search(vec["dense"], vec["sparse"], top_k=FINAL_TOP_K, document_ids=document_ids)
        chunks = [normalize_chunk(h) for h in hits]
        return chunks, chunks
    elif config in ("hybrid_rerank", "parent"):
        cands = await ahybrid_search(vec["dense"], vec["sparse"], top_k=HYBRID_TOP_K, document_ids=document_ids)
        hits = await arerank(question, cands, top_k=FINAL_TOP_K) if cands else []
        children = [normalize_chunk(h) for h in hits]
        chunks = children
        if config == "parent":
            from app.config import settings
            from app.retrieval.expansion import aexpand_with_parents
            chunks = await aexpand_with_parents(children, settings.parent_context_max_tokens)
        return chunks, children
    else:
        raise ValueError(f"Cấu hình retrieval không hợp lệ: {config}")


# ---------------------------------------------------------------------------
# Graph configs (tái dùng node của build_graph.py)
# ---------------------------------------------------------------------------
def build_variant_graph(use_critic: bool):
    from langgraph.graph import END, START, StateGraph

    import app.graph.build_graph as bg
    from app.agents.state import GraphState

    wf = StateGraph(GraphState)
    wf.add_node("retrieval", bg.retrieval_node)
    wf.add_node("grade_context", bg.grade_context_node)
    wf.add_node("rewrite_query", bg.rewrite_query_node)
    wf.add_node("generation", bg.generation_node)
    wf.add_node("accept_answer", bg.accept_answer_node)

    wf.add_edge(START, "retrieval")
    wf.add_edge("retrieval", "grade_context")
    wf.add_conditional_edges(
        "grade_context",
        bg.route_after_grade_context,
        {"rewrite_query": "rewrite_query", "generation": "generation"},
    )
    wf.add_edge("rewrite_query", "retrieval")

    if use_critic:
        wf.add_node("grade_answer", bg.grade_answer_node)
        wf.add_node("retry_generation", bg.retry_generation_node)
        wf.add_edge("generation", "grade_answer")
        wf.add_conditional_edges(
            "grade_answer",
            bg.route_after_grade_answer,
            {"retry_generation": "retry_generation", "accept_answer": "accept_answer"},
        )
        wf.add_edge("retry_generation", "generation")
    else:
        wf.add_edge("generation", "accept_answer")

    wf.add_edge("accept_answer", END)
    return wf.compile()


_pre_expansion: list[dict] | None = None


def _install_expansion_probe() -> None:
    """Bọc aexpand_with_parents trong retrieval_agent để ghi lại chunk con (input) của
    lần retrieval CUỐI trong đồ thị — không sửa logic production."""
    import app.agents.retrieval_agent as ra

    if getattr(ra.aexpand_with_parents, "_eval_probe", False):
        return
    original = ra.aexpand_with_parents

    async def probe(chunks, *args, **kwargs):
        global _pre_expansion
        _pre_expansion = list(chunks)
        return await original(chunks, *args, **kwargs)

    probe._eval_probe = True  # type: ignore[attr-defined]
    ra.aexpand_with_parents = probe


def build_graphs(configs: list[str]) -> dict[str, Any]:
    if any(c in GRAPH_CONFIGS for c in configs):
        _install_expansion_probe()
    graphs: dict[str, Any] = {}
    if "crag" in configs:
        graphs["crag"] = build_variant_graph(use_critic=False)
    if "full" in configs:
        graphs["full"] = build_variant_graph(use_critic=True)
    return graphs


# ---------------------------------------------------------------------------
async def warmup(document_ids: list[str], with_llm: bool = False) -> None:
    """Nạp model embedding + reranker (và mở kết nối LLM) trước để cold-start
    không tính vào latency của câu đầu tiên."""
    vec = await aembed_query("khởi động hệ thống")
    await ahybrid_search(vec["dense"], vec["sparse"], top_k=2, document_ids=document_ids)
    await arerank("khởi động", [{"text": "khởi động hệ thống"}], top_k=1)
    if with_llm:
        from app.llm.utils import LLMCallError, acall_llm_text

        for task in ("qa", "grade"):
            try:
                await acall_llm_text(task, "Chỉ trả lời: ok", "ping")
            except LLMCallError as exc:
                print(f"[warmup] LLM task '{task}' lỗi: {exc}")


async def run_pipeline(
    config: str, question: str, document_ids: list[str], graphs: dict[str, Any], retrieval_only: bool = False
) -> PipelineResult:
    global _pre_expansion
    clear_embedding_cache()
    reset_usage_metrics()
    t0 = time.perf_counter()

    try:
        if config in RETRIEVAL_CONFIGS:
            chunks, children = await retrieve(config, question, document_ids)
            if retrieval_only:
                return PipelineResult(chunks=chunks, child_chunks=children, latency_s=time.perf_counter() - t0)
            out = await run_generation(
                {
                    "session_id": "eval",
                    "user_query": question,
                    "intent": "qa",
                    "retrieved_chunks": chunks,
                    "rewritten_queries": [],
                    "reflection_history": [],
                }
            )
            answer, loops, rewritten = out.get("answer_draft", ""), 0, []
        else:
            state = {
                "session_id": "eval",
                "user_query": question,
                "intent": "qa",
                "document_ids": document_ids,
                "chat_history_summary": "",
                "mastery_context": {},
                "num_reflection_loops": 0,
                "reflection_history": [],
                "rewritten_queries": [],
                "retrieved_chunks": [],
                "citations": [],
                "answer_draft": "",
                "final_answer": "",
            }
            _pre_expansion = None
            final = await graphs[config].ainvoke(state, config={"recursion_limit": 25})
            chunks = final.get("retrieved_chunks") or []
            children = _pre_expansion if _pre_expansion is not None else chunks
            answer = final.get("final_answer") or final.get("answer_draft") or ""
            loops = final.get("num_reflection_loops", 0)
            rewritten = final.get("rewritten_queries") or []

        if not answer.strip() or answer == _ERROR_FALLBACK_ANSWER:
            raise RuntimeError("LLM sinh câu trả lời thất bại (rỗng hoặc fallback lỗi).")
    except Exception as exc:
        return PipelineResult(latency_s=time.perf_counter() - t0, error=f"{type(exc).__name__}: {exc}")

    latency = time.perf_counter() - t0
    usage = get_cumulative_usage()
    return PipelineResult(
        answer=answer,
        chunks=chunks,
        latency_s=latency,
        llm_calls=usage["total_calls"],
        tokens=usage["total_tokens"],
        loops=loops,
        rewritten_queries=rewritten,
        child_chunks=children,
    )
