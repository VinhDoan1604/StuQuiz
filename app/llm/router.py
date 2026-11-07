"""
Model routing theo độ phức tạp task — tối ưu chi phí và độ trễ.

Quy tắc phân bổ:
- CHEAP model (CHEAP_MODEL_PROVIDER / CHEAP_MODEL_NAME):
  + rewrite: viết lại truy vấn dựa vào lịch sử
  + decompose: phân rã câu hỏi phức hợp thành sub-queries
  + grade: chấm độ đủ của context (CRAG)
  + critic: chấm faithfulness của answer theo từng khẳng định (Self-RAG). Đặt
    CRITIC_MODEL_PROVIDER / CRITIC_MODEL_NAME để dùng model KHÁC họ với model sinh.
  + qa: trả lời câu hỏi trực tiếp, ngắn gọn
- STRONG model (STRONG_MODEL_PROVIDER / STRONG_MODEL_NAME):
  + teaching: giảng bài theo phương pháp giàn giáo (scaffolding)
  + quiz: sinh câu hỏi trắc nghiệm kèm giải thích
  + summary: tóm tắt phân cấp tài liệu lớn

Cung cấp:
- `get_llm(task, streaming=False) -> BaseChatModel` (đã cache theo task)
- `set_model_override()` — model user chọn trên UI cho request hiện tại
  (provider khai báo ở app/llm/providers.py)
- `UsageTrackingCallbackHandler` đo token usage & latency từng lời gọi
- `get_cumulative_usage()` / `reset_usage_metrics()` (phục vụ eval/)
"""
import logging
import threading
import time
from collections import deque
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import LLMResult

from app.config import settings
from app.llm.providers import build_chat_model, get_provider

logger = logging.getLogger("app.llm.router")

TaskType = Literal["rewrite", "decompose", "grade", "critic", "qa", "teaching", "quiz", "summary"]

_USAGE_HISTORY_LIMIT = 1000


# --- Tracking Callbacks & Metrics ---

@dataclass
class UsageMetrics:
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=_USAGE_HISTORY_LIMIT))


_GLOBAL_USAGE = UsageMetrics()
_USAGE_LOCK = threading.Lock()


def get_cumulative_usage() -> dict[str, Any]:
    """Tổng token và độ trễ tích luỹ (phục vụ eval/)."""
    with _USAGE_LOCK:
        return {
            "total_calls": _GLOBAL_USAGE.total_calls,
            "total_prompt_tokens": _GLOBAL_USAGE.total_prompt_tokens,
            "total_completion_tokens": _GLOBAL_USAGE.total_completion_tokens,
            "total_tokens": _GLOBAL_USAGE.total_tokens,
            "total_latency_ms": round(_GLOBAL_USAGE.total_latency_ms, 2),
            "history_count": len(_GLOBAL_USAGE.history),
        }


def reset_usage_metrics() -> None:
    """Reset bộ đếm token usage (dùng khi bắt đầu benchmark mới)."""
    global _GLOBAL_USAGE
    with _USAGE_LOCK:
        _GLOBAL_USAGE = UsageMetrics()


def _extract_usage(response: LLMResult) -> tuple[int, int, int]:
    if response.llm_output and "token_usage" in response.llm_output:
        usage = response.llm_output["token_usage"] or {}
        prompt = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        completion = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        return prompt, completion, usage.get("total_tokens", prompt + completion)

    prompt = completion = total = 0
    for gen_list in response.generations:
        for gen in gen_list:
            message = getattr(gen, "message", None)
            meta = getattr(message, "usage_metadata", None) or (
                (getattr(gen, "generation_info", None) or {}).get("usage_metadata")
            )
            if meta:
                prompt += meta.get("input_tokens", 0)
                completion += meta.get("output_tokens", 0)
                total += meta.get("total_tokens", 0)
    return prompt, completion, total


class UsageTrackingCallbackHandler(BaseCallbackHandler):
    """Ghi nhận latency và token của từng lời gọi LLM.

    1 handler được dùng chung bởi model đã cache => thời điểm bắt đầu lưu theo
    `run_id` để các lời gọi song song không ghi đè nhau.
    """

    def __init__(self, task: str, model_name: str):
        super().__init__()
        self.task = task
        self.model_name = model_name
        self._starts: dict[UUID, float] = {}

    def _on_start(self, run_id: UUID) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs: Any) -> None:
        self._on_start(run_id)

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list, *, run_id: UUID, **kwargs: Any) -> None:
        self._on_start(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._starts.pop(run_id, None)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        start = self._starts.pop(run_id, None)
        latency_ms = (time.perf_counter() - start) * 1000.0 if start is not None else 0.0
        prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)

        with _USAGE_LOCK:
            _GLOBAL_USAGE.total_calls += 1
            _GLOBAL_USAGE.total_prompt_tokens += prompt_tokens
            _GLOBAL_USAGE.total_completion_tokens += completion_tokens
            _GLOBAL_USAGE.total_tokens += total_tokens
            _GLOBAL_USAGE.total_latency_ms += latency_ms
            _GLOBAL_USAGE.history.append({
                "task": self.task,
                "model": self.model_name,
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            })

        logger.info(
            "LLM call: task=%s | model=%s | latency=%.0fms | tokens in=%d out=%d total=%d",
            self.task, self.model_name, latency_ms, prompt_tokens, completion_tokens, total_tokens,
        )


# --- Mapping nhiệm vụ & nhiệt độ ---

TASK_TEMPERATURE_MAP: dict[TaskType, float] = {
    "grade": 0.0,       # chấm điểm khách quan, ổn định
    "critic": 0.0,
    "rewrite": 0.1,     # viết lại truy vấn bám sát ý gốc
    "decompose": 0.1,   # phân rã câu hỏi theo logic chặt
    "qa": 0.5,
    "summary": 0.3,
    "quiz": 0.6,        # câu hỏi đa dạng
    "teaching": 0.7,    # ví dụ minh hoạ phong phú
}

CHEAP_TASKS: set[TaskType] = {"rewrite", "decompose", "grade", "critic", "qa"}
STRONG_TASKS: set[TaskType] = {"teaching", "quiz", "summary"}


# Model user chọn trên UI chỉ áp cho các task SINH CÂU TRẢ LỜI. Các task nội bộ
# (rewrite/decompose/grade) giữ model cấu hình sẵn để ổn định structured output & chi phí.
USER_SELECTABLE_TASKS: set[TaskType] = {"qa", "teaching", "quiz", "summary"}

# (provider, model) user chọn cho request hiện tại. ContextVar => mỗi request độc lập,
# tự lan xuống các node LangGraph (asyncio copy context khi tạo task).
_MODEL_OVERRIDE: ContextVar[tuple[str, str] | None] = ContextVar("llm_model_override", default=None)


def set_model_override(provider: str | None, model: str | None) -> Token:
    """Đặt model cho request hiện tại (None = dùng cấu hình mặc định)."""
    value = (provider, model) if provider and model else None
    return _MODEL_OVERRIDE.set(value)


def reset_model_override(token: Token) -> None:
    _MODEL_OVERRIDE.reset(token)


def get_llm(task: TaskType, streaming: bool = False) -> BaseChatModel:
    """Trả về chat model cho `task` (cache theo (task, streaming, provider, model))."""
    override = _MODEL_OVERRIDE.get()
    if override and task in USER_SELECTABLE_TASKS:
        provider, model_name = override
    elif task == "critic" and settings.critic_model_provider.strip() and settings.critic_model_name.strip():
        provider, model_name = settings.critic_model_provider, settings.critic_model_name.strip()
    elif task in CHEAP_TASKS:
        provider, model_name = settings.cheap_model_provider, settings.cheap_model_name
    else:
        provider, model_name = settings.strong_model_provider, settings.strong_model_name
    return _build_llm(task, streaming, provider.strip().lower(), model_name)


@lru_cache(maxsize=64)
def _build_llm(task: TaskType, streaming: bool, provider: str, model_name: str) -> BaseChatModel:
    spec = get_provider(provider)
    common: dict[str, Any] = {
        "model": model_name,
        "temperature": TASK_TEMPERATURE_MAP.get(task, 0.5),
        "streaming": streaming,
        "max_retries": settings.llm_max_retries,
        "callbacks": [UsageTrackingCallbackHandler(task=task, model_name=f"{spec.name}:{model_name}")],
    }
    if task == "critic" and spec.protocol == "anthropic":
        common["max_tokens"] = 4096  # danh sách claim dài; mặc định langchain-anthropic dễ cắt cụt tool call
    return build_chat_model(spec, model_name, common)
