"""
app/memory/context_window.py

build_context / maybe_summarize — Quản lý context window thông minh cho chat
(mục 5.11) theo chiến lược:
1. Sliding window: N turn gần nhất (mặc định 6 turn ~ 3 cặp Q&A), giữ nguyên văn.
2. Incremental Rolling Summary: Các turn cũ hơn sliding window được LLM nén lại.
   Được theo dõi chính xác qua con trỏ `ChatSession.summarized_turn_count` để chỉ
   tóm tắt phần DELTA mới xuất hiện, không bao giờ recompute lại từ đầu gây lãng phí token.
3. Token Budgeting: Luôn đếm token bằng tiktoken để không bao giờ vượt quá
   ngân sách context window của model được định tuyến tới.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.memory.chat_history import get_recent_turns
from app.models.db_models import ChatSession, ChatTurn
from app.utils.tokens import count_tokens

logger = logging.getLogger(__name__)

# Số turn (không phải cặp hỏi-đáp) giữ nguyên văn trong sliding window.
# 6 turn ~ 3 cặp Q&A gần nhất.
SLIDING_WINDOW_TURNS = 6

# Trigger tóm tắt định kỳ mỗi khi số lượng turn mới tích lũy >= giá trị này.
SUMMARIZE_TRIGGER_TURNS = 10

# Dự phòng token cho phần còn lại của prompt (system prompt, câu hỏi hiện
# tại, retrieved context...) khi build_context tính budget riêng cho lịch sử chat.
_SAFETY_MARGIN_TOKENS = 50


def _format_turn(turn: ChatTurn) -> str:
    role_label = "Người dùng" if turn.role == "user" else "Trợ lý"
    return f"{role_label}: {turn.content}"


def _covered_turn_count(session: ChatSession) -> int:
    """Số turn đã được gộp an toàn vào rolling_summary và an toàn bỏ qua khi fetch raw."""
    return getattr(session, "summarized_turn_count", 0) or 0


def build_context(db: Session, session_id: str, token_budget: int) -> str:
    """
    Trả về 1 string ngắn gọn để đưa vào prompt, gồm (nếu có):

        [Tóm tắt hội thoại trước đó]
        <rolling_summary>

        [Lịch sử gần đây]
        <turn gần nhất, số lượng phụ thuộc đã tóm tắt tới đâu>

    KHÔNG BAO GIỜ vượt quá `token_budget` token. Nếu phải cắt vì vượt ngân
    sách, luôn cắt turn CŨ NHẤT trước.
    """
    if token_budget <= 0:
        return ""

    session = db.get(ChatSession, session_id)
    summary = (session.rolling_summary or "").strip() if session else ""

    parts: list[str] = []
    remaining_budget = token_budget - _SAFETY_MARGIN_TOKENS

    if summary and remaining_budget > 0:
        summary_block = f"[Tóm tắt hội thoại trước đó]\n{summary}"
        summary_tokens = count_tokens(summary_block)
        if summary_tokens <= remaining_budget:
            parts.append(summary_block)
            remaining_budget -= summary_tokens
        else:
            logger.warning(
                "build_context('%s'): rolling_summary (%d token) vượt token_budget — bỏ qua summary.",
                session_id, summary_tokens,
            )

    if remaining_budget > 0 and session is not None:
        total_turns = db.query(ChatTurn).filter(ChatTurn.session_id == session_id).count()
        covered = _covered_turn_count(session)
        window_size = max(SLIDING_WINDOW_TURNS, total_turns - covered)

        recent_turns_desc = get_recent_turns(db, session_id, limit=window_size)  # mới nhất trước

        turn_lines: list[str] = []
        used_tokens = 0
        for turn in recent_turns_desc:  # từ mới nhất -> ưu tiên giữ turn gần hiện tại khi phải cắt
            line = _format_turn(turn)
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens > remaining_budget:
                break
            turn_lines.append(line)
            used_tokens += line_tokens

        if turn_lines:
            turn_lines.reverse()  # trả về thứ tự chronological (cũ -> mới) cho dễ đọc trong prompt
            parts.append("[Lịch sử gần đây]\n" + "\n".join(turn_lines))

    return "\n\n".join(parts)


def maybe_summarize(db: Session, session_id: str) -> None:
    """
    Gộp các turn cũ hơn sliding window vào rolling_summary bằng LLM theo phương thức INCREMENTAL.
    Được trigger định kỳ khi số turn mới tích lũy >= SUMMARIZE_TRIGGER_TURNS.
    Chỉ tóm tắt phần delta mới, cập nhật atomic con trỏ `summarized_turn_count`.
    """
    session = db.get(ChatSession, session_id)
    if session is None:
        logger.warning("maybe_summarize: session '%s' không tồn tại.", session_id)
        return

    total_turns = db.query(ChatTurn).filter(ChatTurn.session_id == session_id).count()
    last_summarized = getattr(session, "summarized_turn_count", 0) or 0

    if total_turns - last_summarized < SUMMARIZE_TRIGGER_TURNS:
        return

    turns_to_fold_count = total_turns - SLIDING_WINDOW_TURNS - last_summarized
    if turns_to_fold_count <= 0:
        return

    turns_to_fold = (
        db.query(ChatTurn)
        .filter(ChatTurn.session_id == session_id)
        .order_by(ChatTurn.created_at.asc())
        .offset(last_summarized)
        .limit(turns_to_fold_count)
        .all()
    )
    if not turns_to_fold:
        return

    new_text = "\n".join(_format_turn(t) for t in turns_to_fold)
    old_summary = (session.rolling_summary or "").strip()

    system_prompt = (
        "Bạn là trợ lý tóm tắt hội thoại học tập giữa sinh viên và AI. "
        "Nhiệm vụ: viết bản tóm tắt súc tích (tối đa khoảng 150 từ), GIỮ LẠI "
        "các thông tin quan trọng (chủ đề đang học, câu hỏi/khái niệm sinh "
        "viên còn thắc mắc, các tham chiếu có thể được nhắc lại sau như "
        "'định lý đó', 'ví dụ trước'). CHỈ trả về bản tóm tắt, không thêm giải thích."
    )
    if old_summary:
        user_prompt = (
            f"Tóm tắt hiện có:\n{old_summary}\n\n"
            f"Đoạn hội thoại mới cần gộp vào:\n{new_text}"
        )
    else:
        user_prompt = f"Đoạn hội thoại cần tóm tắt:\n{new_text}"

    try:
        from app.llm.router import get_llm
        from app.llm.utils import extract_text_from_response

        llm = get_llm("summary")
        response = llm.invoke([("system", system_prompt), ("human", user_prompt)])
        updated_summary = extract_text_from_response(response)
    except Exception as exc:
        logger.error(
            "maybe_summarize('%s'): LLM tóm tắt thất bại (%s) — giữ nguyên summary cũ.", session_id, exc
        )
        return

    if not updated_summary:
        logger.warning("maybe_summarize('%s'): LLM trả về summary rỗng — bỏ qua lần này.", session_id)
        return

    session.rolling_summary = updated_summary
    session.summarized_turn_count = last_summarized + turns_to_fold_count

    db.add(session)
    db.commit()
    logger.info(
        "maybe_summarize('%s'): đã gộp %d turn cũ vào rolling_summary (summarized_turn_count=%d).",
        session_id, turns_to_fold_count, session.summarized_turn_count,
    )