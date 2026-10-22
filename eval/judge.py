"""
eval/judge.py

LLM-as-judge. Khác bản cũ trong ragas_eval.py:
- KHÔNG BAO GIỜ trả hằng số khi lỗi: thử lại rồi `raise JudgeError`. Câu nào
  judge lỗi sẽ bị đánh dấu và đếm riêng trong báo cáo, không "đẹp hoá" điểm.
- Faithfulness theo từng khẳng định (claim-level): tách câu trả lời thành các
  khẳng định, mỗi cái được hỗ trợ bởi context hay không -> điểm = tỉ lệ được hỗ trợ.
- Judge nên là model KHÁC model sinh. Đặt EVAL_JUDGE_PROVIDER / EVAL_JUDGE_MODEL;
  nếu không đặt, dùng strong model trong settings và báo cáo sẽ ghi cảnh báo
  khi trùng với model sinh (cheap model).
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field

from app.config import settings
from app.retrieval.helpers import format_chunks_for_prompt

logger = logging.getLogger("eval.judge")
T = TypeVar("T", bound=BaseModel)


class JudgeError(RuntimeError):
    pass


class Claim(BaseModel):
    claim: str
    supported: bool


class FaithfulnessVerdict(BaseModel):
    abstained: bool = Field(
        description="True nếu câu trả lời chủ yếu nói tài liệu không đề cập / không đủ thông tin để trả lời."
    )
    claims: list[Claim] = Field(description="Các khẳng định thực tế độc lập trong câu trả lời (tối đa 15).")


class CorrectnessVerdict(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    reason: str


_FAITHFULNESS_SYSTEM = """Bạn là giám khảo nghiêm khắc đánh giá hệ thống RAG.
Cho: câu hỏi, CONTEXT (các đoạn tài liệu được truy xuất, đánh số) và CÂU TRẢ LỜI.

1. Tách câu trả lời thành các KHẲNG ĐỊNH thực tế độc lập (tối đa 15). Bỏ qua: số trích
   dẫn dạng [1], lời dẫn dắt, câu hỏi gợi mở, tiêu đề/định dạng.
2. Với mỗi khẳng định, supported=true CHỈ KHI context nêu trực tiếp hoặc suy ra tất yếu.
   Kiến thức bên ngoài dù ĐÚNG nhưng KHÔNG có trong context => supported=false.
3. abstained=true nếu câu trả lời chủ yếu nói rằng tài liệu không đề cập / không đủ thông tin.
   Khi abstained=true thì claims có thể rỗng."""

_CORRECTNESS_SYSTEM = """Bạn là giám khảo chấm độ đúng của câu trả lời so với ĐÁP ÁN CHUẨN.
- correct   : nêu đủ ý chính của đáp án chuẩn, không mâu thuẫn.
- partial   : đúng một phần / thiếu ý quan trọng nhưng không sai.
- incorrect : sai, mâu thuẫn đáp án chuẩn, hoặc từ chối trả lời / nói không có thông tin.
Chỉ so với đáp án chuẩn, không dùng kiến thức riêng. reason: 1 câu ngắn."""


def build_judge_llm() -> tuple[Any, dict]:
    """Trả về (llm, meta). meta.same_as_generator=True nghĩa là judge trùng model sinh."""
    provider = (os.getenv("EVAL_JUDGE_PROVIDER") or settings.strong_model_provider).lower()
    model = os.getenv("EVAL_JUDGE_MODEL") or settings.strong_model_name
    same = (provider, model) == (settings.cheap_model_provider.lower(), settings.cheap_model_name)
    meta = {"provider": provider, "model": model, "same_as_generator": same}
    if same:
        logger.warning(
            "Judge TRÙNG model sinh (%s/%s) -> điểm có thể bị thổi phồng. Đặt EVAL_JUDGE_MODEL khác.",
            provider, model,
        )

    if provider in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model, google_api_key=settings.effective_google_api_key, temperature=0.0
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, api_key=settings.openai_api_key, temperature=0.0)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=model, api_key=settings.anthropic_api_key, temperature=0.0)
    else:
        raise JudgeError(f"Provider judge không hỗ trợ: {provider}")
    return llm, meta


async def _structured(llm: Any, model_cls: Type[T], system: str, user: str, retries: int = 3) -> T:
    runnable = llm.with_structured_output(model_cls)
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            res = await runnable.ainvoke([("system", system), ("human", user)])
            return res if isinstance(res, model_cls) else model_cls.model_validate(res)
        except Exception as exc:  # mạng, 429, parse, None trả về...
            last = exc
            await asyncio.sleep(2**attempt)
    raise JudgeError(f"Judge thất bại sau {retries} lần: {type(last).__name__}: {last}")


async def judge_faithfulness(
    llm: Any, question: str, chunks: list[dict], answer: str
) -> FaithfulnessVerdict:
    user = (
        f"Câu hỏi: {question}\n\nCONTEXT:\n{format_chunks_for_prompt(chunks)}\n\n"
        f"CÂU TRẢ LỜI:\n{answer}"
    )
    return await _structured(llm, FaithfulnessVerdict, _FAITHFULNESS_SYSTEM, user)


async def judge_correctness(llm: Any, question: str, gold_answer: str, answer: str) -> CorrectnessVerdict:
    user = f"Câu hỏi: {question}\n\nĐÁP ÁN CHUẨN:\n{gold_answer}\n\nCÂU TRẢ LỜI CẦN CHẤM:\n{answer}"
    return await _structured(llm, CorrectnessVerdict, _CORRECTNESS_SYSTEM, user)


def faithfulness_score(v: FaithfulnessVerdict) -> Optional[float]:
    """Tỉ lệ khẳng định được hỗ trợ. None nếu không có khẳng định nào (vd từ chối trả lời):
    những câu này không đưa vào trung bình faithfulness mà được đo bằng chỉ số abstention."""
    if not v.claims:
        return None
    return sum(c.supported for c in v.claims) / len(v.claims)


CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
