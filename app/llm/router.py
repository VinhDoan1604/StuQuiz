# STATUS: READY (Hoàn thành Phase 1)
"""
Model routing theo độ phức tạp task (mục 5.9 ROADMAP) — tối ưu chi phí và độ trễ.

Quy tắc phân bổ:
- CHEAP model (mặc định Gemini Flash / OpenAI gpt-4o-mini):
  + rewrite: Viết lại truy vấn dựa vào lịch sử
  + decompose: Phân rã câu hỏi phức hợp thành các sub-queries
  + grade: Chấm điểm liên quan của context và tính trung thực của answer
  + qa: Trả lời câu hỏi trực tiếp, ngắn gọn
- STRONG model (mặc định Gemini Pro / Claude Sonnet / GPT-4o):
  + teaching: Giảng bài theo phương pháp giàn giáo (Scaffolding), giải thích bản chất
  + quiz: Sinh câu hỏi trắc nghiệm kèm giải thích logic
  + summary: Tóm tắt phân cấp tài liệu lớn

Cung cấp:
- Hàm `get_llm(task: TaskType, streaming: bool = False) -> BaseChatModel`
- Callback handler `UsageTrackingCallbackHandler` đo token usage & latency
- Hàm truy xuất metrics `get_cumulative_usage()`
"""
import time
import logging
from typing import Literal, Any
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.config import settings

logger = logging.getLogger("app.llm.router")

TaskType = Literal["rewrite", "decompose", "grade", "qa", "teaching", "quiz", "summary"]


# --- Tracking Callbacks & Metrics ---

@dataclass
class UsageMetrics:
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


_GLOBAL_USAGE = UsageMetrics()


def get_cumulative_usage() -> dict[str, Any]:
    """Trả về tổng lượng token và độ trễ tích lũy phục vụ phần evaluation (mục 5.13)."""
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
    _GLOBAL_USAGE = UsageMetrics()


class UsageTrackingCallbackHandler(BaseCallbackHandler):
    """Callback ghi nhận latency và lượng token tiêu thụ của từng lời gọi LLM."""

    def __init__(self, task: str, model_name: str):
        super().__init__()
        self.task = task
        self.model_name = model_name
        self.start_time: float = 0.0

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self.start_time = time.perf_counter()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        latency_ms = (time.perf_counter() - self.start_time) * 1000.0

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        # Trích xuất token usage từ LLMResult (LangChain output metadata)
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        else:
            # Đối với Google GenAI / LangChain mới hơn
            for gen_list in response.generations:
                for gen in gen_list:
                    gen_info = getattr(gen, "generation_info", {}) or {}
                    usage_meta = gen_info.get("usage_metadata")
                    if usage_meta:
                        prompt_tokens += usage_meta.get("input_tokens", 0)
                        completion_tokens += usage_meta.get("output_tokens", 0)
                        total_tokens += usage_meta.get("total_tokens", 0)

        # Cập nhật global metrics
        _GLOBAL_USAGE.total_calls += 1
        _GLOBAL_USAGE.total_prompt_tokens += prompt_tokens
        _GLOBAL_USAGE.total_completion_tokens += completion_tokens
        _GLOBAL_USAGE.total_tokens += total_tokens
        _GLOBAL_USAGE.total_latency_ms += latency_ms

        call_record = {
            "task": self.task,
            "model": self.model_name,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        _GLOBAL_USAGE.history.append(call_record)

        logger.info(
            "LLM Call: task=%s | model=%s | latency=%.1fms | tokens: in=%d, out=%d, total=%d",
            self.task,
            self.model_name,
            latency_ms,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )


# --- Mapping Nhiệm vụ & Nhiệt độ (Temperature) ---

TASK_TEMPERATURE_MAP: dict[TaskType, float] = {
    "grade": 0.0,       # Tuyệt đối khách quan khi chấm context/answer
    "rewrite": 0.1,     # Viết lại truy vấn chính xác, tránh biến tướng
    "decompose": 0.1,   # Phân rã câu hỏi multi-hop theo logic chặt
    "qa": 0.5,          # Trả lời hỏi đáp thông thường
    "summary": 0.3,     # Tóm tắt cô đọng trung thực
    "quiz": 0.6,        # Sinh câu hỏi trắc nghiệm phong phú, sáng tạo
    "teaching": 0.7,    # Giảng giải sư phạm, ví dụ trực quan phong phú
}

CHEAP_TASKS: set[TaskType] = {"rewrite", "decompose", "grade", "qa"}
STRONG_TASKS: set[TaskType] = {"teaching", "quiz", "summary"}


def get_llm(task: TaskType, streaming: bool = False) -> BaseChatModel:
    """
    Khởi tạo và trả về LangChain Chat Model phù hợp cho task cụ thể.
    
    Hỗ trợ provider:
    - "google" (hoặc "gemini"): Sử dụng Gemini models (gemini-2.0-flash, gemini-2.5-pro, ...)
    - "openai": Sử dụng OpenAI models (gpt-4o-mini, gpt-4o, ...)
    - "anthropic": Sử dụng Anthropic models (claude-sonnet-4-6, ...)
    """
    # 1. Phân loại Cheap vs Strong
    if task in CHEAP_TASKS:
        provider = settings.cheap_model_provider.lower()
        model_name = settings.cheap_model_name
    else:
        provider = settings.strong_model_provider.lower()
        model_name = settings.strong_model_name

    temperature = TASK_TEMPERATURE_MAP.get(task, 0.5)
    tracking_callback = UsageTrackingCallbackHandler(task=task, model_name=model_name)

    # 2. Khởi tạo Chat Model theo Provider
    if provider in ("google", "gemini"):
        api_key = settings.effective_google_api_key
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                streaming=streaming,
                callbacks=[tracking_callback],
            )
        except ImportError:
            # Fallback: Nếu môi trường chưa có langchain-google-genai, dùng endpoint OpenAI tương thích của Google
            from langchain_openai import ChatOpenAI
            logger.warning(
                "langchain-google-genai chưa cài đặt, fallback sang ChatOpenAI với Google API Endpoint"
            )
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                temperature=temperature,
                streaming=streaming,
                callbacks=[tracking_callback],
            )

    elif provider == "openai":
        # Sẵn sàng kích hoạt khi chuyển đổi sang OpenAI
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            temperature=temperature,
            streaming=streaming,
            callbacks=[tracking_callback],
        )

    elif provider == "anthropic":
        # Sẵn sàng kích hoạt khi chuyển đổi sang Claude
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model_name,
                api_key=settings.anthropic_api_key,
                temperature=temperature,
                streaming=streaming,
                callbacks=[tracking_callback],
            )
        except ImportError as exc:
            raise ImportError(
                "Để sử dụng Anthropic, vui lòng cài đặt: pip install langchain-anthropic"
            ) from exc

    else:
        raise ValueError(
            f"Provider không được hỗ trợ: '{provider}'. Hỗ trợ: 'google', 'openai', 'anthropic'."
        )
