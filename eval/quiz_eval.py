# STATUS: STUB — CẦN ĐẦU TƯ KỸ
"""
Eval riêng cho Quiz Agent (LLM-as-judge: câu hỏi có chính xác so với nội
dung gốc không, đáp án đúng có thật sự đúng không, distractor có hợp lý
không) và Summary Agent (coverage: tóm tắt có bỏ sót ý quan trọng không
— so khớp key points được LLM/rule trích từ tài liệu gốc).

Đo thêm latency + token cost mỗi loại request (QA nhanh hơn Quiz/Summary,
mục 5.13) — log lại khi gọi get_llm() trong app/llm/router.py.
"""


def eval_quiz_quality(quiz_questions: list, source_document_id: str) -> dict:
    raise NotImplementedError


def eval_summary_coverage(summary_text: str, source_document_id: str) -> dict:
    raise NotImplementedError
