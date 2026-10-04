"""
scripts/seed_demo.py

Seed dữ liệu demo chuyên nghiệp cho StudyRAG:
- 1 User học viên mẫu (`student@studyrag.edu.vn`)
- 2 Giáo trình chuyên sâu đã lập chỉ mục (Cấu trúc dữ liệu & RAG nâng cao)
- 5 Bản ghi độ làm chủ kiến thức SuperMemo SM-2 (vững, yếu, chu kỳ ôn tập)
- 1 Phiên Chat mẫu hoàn chỉnh với CRAG reflection trace và trích dẫn chuẩn
- Các lượt làm bài Quiz lịch sử

Script này giúp người dùng/nhà tuyển dụng mở ứng dụng lên là thấy đầy đủ
timeline tư duy agent, biểu đồ mastery và trích dẫn đa tầng mà không cần thao tác thủ công.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta

# Đảm bảo UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import Base, SessionLocal, engine
from app.models.db_models import (
    User,
    Document,
    ChatSession,
    ChatTurn,
    MasteryRecord,
    QuizAttempt,
)


def seed_demo_data():
    print("=" * 70)
    print("🌱 Đang chuẩn bị seed dữ liệu demo StudyRAG...")
    print("=" * 70)

    # Đảm bảo bảng đã tồn tại
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # 1. User mẫu
        demo_user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "demo_user"))
        user = db.query(User).filter(User.id == demo_user_id).first()
        if not user:
            user = User(
                id=demo_user_id,
                email="student@studyrag.edu.vn",
                created_at=datetime.utcnow() - timedelta(days=14),
            )
            db.add(user)
            db.commit()
            print(f"✅ Đã tạo user mẫu: {user.email} (ID: {user.id})")
        else:
            print(f"ℹ️ User mẫu đã tồn tại: {user.email}")

        # 2. Tài liệu học tập mẫu
        doc1_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "doc_dsa_2026"))
        doc2_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "doc_rag_agents_2026"))

        if not db.query(Document).filter(Document.id == doc1_id).first():
            doc1 = Document(
                id=doc1_id,
                owner_id=user.id,
                filename="Giao_Trinh_Cau_Truc_Du_Lieu_Va_Giai_Thuat.pdf",
                file_type="pdf",
                subject="Khoa Học Máy Tính",
                uploaded_at=datetime.utcnow() - timedelta(days=10),
                num_chunks=48,
            )
            db.add(doc1)
            print(f"✅ Đã nạp tài liệu: {doc1.filename} ({doc1.num_chunks} chunks)")

        if not db.query(Document).filter(Document.id == doc2_id).first():
            doc2 = Document(
                id=doc2_id,
                owner_id=user.id,
                filename="Kien_Truc_RAG_Nang_Cao_Va_He_Agent.pdf",
                file_type="pdf",
                subject="Trí Tuệ Nhân Tạo & NLP",
                uploaded_at=datetime.utcnow() - timedelta(days=5),
                num_chunks=36,
            )
            db.add(doc2)
            print(f"✅ Đã nạp tài liệu: {doc2.filename} ({doc2.num_chunks} chunks)")

        db.commit()

        # 3. Bản ghi SuperMemo SM-2 Spaced Repetition Mastery
        mastery_samples = [
            {
                "topic": "Thuật toán Dijkstra & Đường đi ngắn nhất",
                "mastery_score": 85,
                "streak": 4,
                "correct_count": 12,
                "incorrect_count": 2,
                "days_ago": 1,
                "due_in_days": 14,
            },
            {
                "topic": "Quy hoạch động: Bài toán cái túi (Knapsack)",
                "mastery_score": 40,
                "streak": 0,
                "correct_count": 3,
                "incorrect_count": 6,
                "days_ago": 0,
                "due_in_days": 1,
            },
            {
                "topic": "BGE-M3 Multilingual Embedding & Hybrid Search",
                "mastery_score": 90,
                "streak": 5,
                "correct_count": 16,
                "incorrect_count": 1,
                "days_ago": 3,
                "due_in_days": 14,
            },
            {
                "topic": "Corrective RAG (CRAG) & Vòng lặp viết lại Query",
                "mastery_score": 50,
                "streak": 1,
                "correct_count": 5,
                "incorrect_count": 4,
                "days_ago": 2,
                "due_in_days": 3,
            },
            {
                "topic": "Self-RAG Anti-Hallucination Critic Loop",
                "mastery_score": 75,
                "streak": 2,
                "correct_count": 9,
                "incorrect_count": 2,
                "days_ago": 4,
                "due_in_days": 7,
            },
        ]

        for item in mastery_samples:
            existing = (
                db.query(MasteryRecord)
                .filter(MasteryRecord.user_id == user.id, MasteryRecord.topic == item["topic"])
                .first()
            )
            if not existing:
                rec = MasteryRecord(
                    id=str(uuid.uuid4()),
                    user_id=user.id,
                    topic=item["topic"],
                    mastery_score=item["mastery_score"],
                    streak=item["streak"],
                    correct_count=item["correct_count"],
                    incorrect_count=item["incorrect_count"],
                    last_reviewed_at=datetime.utcnow() - timedelta(days=item["days_ago"]),
                    next_review_due_at=datetime.utcnow() + timedelta(days=item["due_in_days"]),
                )
                db.add(rec)
        db.commit()
        print(f"✅ Đã tạo {len(mastery_samples)} chủ đề Spaced Review theo thuật toán SM-2.")

        # 4. Phiên Chat mẫu có Reflection Trace & Citations
        session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "demo_session_1"))
        chat_session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not chat_session:
            chat_session = ChatSession(
                id=session_id,
                user_id=user.id,
                created_at=datetime.utcnow() - timedelta(hours=2),
                rolling_summary="Học viên đã tìm hiểu về thuật toán tìm đường đi ngắn nhất Dijkstra và sự khác biệt giữa đồ thị trọng số dương và âm. Tiếp theo học viên nghiên cứu kiến trúc CRAG đa tác tử.",
                summarized_turn_count=2,
            )
            db.add(chat_session)
            db.commit()

            # Turn 1: Dijkstra QA
            turn1_user = ChatTurn(
                id=str(uuid.uuid4()),
                session_id=chat_session.id,
                role="user",
                intent="qa",
                content="Thuật toán Dijkstra có xử lý được đồ thị có trọng số âm không? Nếu không thì giải thuật nào giải quyết được?",
                created_at=datetime.utcnow() - timedelta(minutes=90),
            )
            turn1_bot = ChatTurn(
                id=str(uuid.uuid4()),
                session_id=chat_session.id,
                role="assistant",
                intent="qa",
                content=(
                    "### 1. Trả lời trực tiếp:\n"
                    "**Không**, thuật toán Dijkstra **không thể** xử lý chính xác các đồ thị có cạnh mang trọng số âm.\n\n"
                    "### 2. Nguyên nhân cốt lõi (Bản chất thuật toán Greedy):\n"
                    "- Dijkstra hoạt động dựa trên chiến lược tham lam (*Greedy approach*). Khi một đỉnh $u$ được lấy ra khỏi hàng đợi ưu tiên (*Priority Queue*) và đánh dấu là đã tối ưu (*settled*), thuật toán giả định rằng khoảng cách $d[u]$ này là ngắn nhất tuyệt đối và sẽ không bao giờ được cập nhật lại.\n"
                    "- Nếu tồn tại một cạnh âm $v \\to w$ xuất hiện ở các bước duyệt sau, nó có thể làm tổng chi phí đến một đỉnh đã cố định giảm xuống, khiến tính đúng đắn của toàn bộ thuật toán bị phá vỡ.\n"
                    "- Nếu đồ thị có chu trình âm (*Negative Cycle*), đường đi có thể giảm vô hạn về $-\\infty$.\n\n"
                    "### 3. Giải thuật thay thế chuẩn xác:\n"
                    "1. **Thuật toán Bellman-Ford**:\n"
                    "   - Xử lý hoàn hảo đồ thị có trọng số âm với độ phức tạp $O(V \\cdot E)$.\n"
                    "   - Có khả năng phát hiện chu trình âm sau $V-1$ lần lặp nới lỏng (*relaxation*).\n"
                    "2. **Thuật toán Johnson**:\n"
                    "   - Phù hợp cho bài toán tìm đường đi ngắn nhất giữa mọi cặp đỉnh (*All-Pairs Shortest Path*) trên đồ thị thưa có trọng số âm bằng kỹ thuật định mức lại trọng số (*Reweighting*) kết hợp Bellman-Ford và Dijkstra."
                ),
                reflection_trace={
                    "context_grade": 0.94,
                    "context_sufficient": True,
                    "rewritten_queries": [],
                    "answer_grade": 0.98,
                    "answer_faithful": True,
                    "num_loops": 0,
                },
                created_at=datetime.utcnow() - timedelta(minutes=89),
            )

            # Turn 2: CRAG Reflection loop demo
            turn2_user = ChatTurn(
                id=str(uuid.uuid4()),
                session_id=chat_session.id,
                role="user",
                intent="teaching",
                content="Khi tác tử Retrieval tìm kiếm ngữ cảnh không đạt yêu cầu, Corrective RAG (CRAG) sẽ phản ứng thế nào?",
                created_at=datetime.utcnow() - timedelta(minutes=30),
            )
            turn2_bot = ChatTurn(
                id=str(uuid.uuid4()),
                session_id=chat_session.id,
                role="assistant",
                intent="teaching",
                content=(
                    "### Cơ chế Phản Hồi của Corrective RAG (CRAG) khi Thiếu Dữ Kiện:\n\n"
                    "Trong kiến trúc StudyRAG, khi tài liệu truy xuất ban đầu không đủ độ bao quát, hệ thống không trả về câu trả lời hời hợt hoặc bịa đặt (*hallucination*), mà kích hoạt **vòng lặp sửa sai tự động (Self-Correction Loop)**:\n\n"
                    "1. **Đánh giá độ liên quan (`grade_context`)**:\n"
                    "   - Tác tử Reflection chấm điểm tính phù hợp của tài liệu (ví dụ đạt 0.52 < ngưỡng chuẩn 0.70).\n"
                    "2. **Tái cấu trúc câu truy vấn (`rewrite_query`)**:\n"
                    "   - Tác tử phân tích câu hỏi ban đầu, trích xuất thực thể chính và viết lại thành các truy vấn đa dạng (semantic broadening & lexical enhancement).\n"
                    "3. **Truy xuất lặp lần 2 (`retrieval`)**:\n"
                    "   - Tiến hành tìm kiếm phối hợp Dense BGE-M3 và Sparse Lexical với bộ query mới được tinh chỉnh.\n"
                    "4. **Kiểm định trung thực (`grade_answer` Critic)**:\n"
                    "   - Mô hình Critic đối chiếu từng câu khẳng định trong lời giải với tài liệu gốc để đảm bảo độ tin cậy tuyệt đối trước khi gửi tới học viên."
                ),
                reflection_trace={
                    "context_grade": 0.54,
                    "context_sufficient": False,
                    "rewritten_queries": [
                        "Corrective RAG CRAG query reformulation when context grade low",
                        "CRAG self-reflection loop and fallback retrieval mechanism",
                    ],
                    "answer_grade": 0.95,
                    "answer_faithful": True,
                    "num_loops": 1,
                },
                created_at=datetime.utcnow() - timedelta(minutes=29),
            )

            db.add_all([turn1_user, turn1_bot, turn2_user, turn2_bot])
            db.commit()
            print("✅ Đã tạo phiên hội thoại mẫu (demo_session_1) kèm dữ liệu CRAG loops.")

        # 5. Bài thi trắc nghiệm lịch sử (Quiz Attempts)
        quiz1 = QuizAttempt(
            id=str(uuid.uuid4()),
            user_id=user.id,
            topic="Thuật toán Dijkstra & Đường đi ngắn nhất",
            question="Thuật toán Dijkstra sử dụng cấu trúc dữ liệu nào sau đây để tối ưu việc tìm đỉnh có khoảng cách nhỏ nhất?",
            choices=[
                "A. Ngăn xếp (Stack LIFO)",
                "B. Hàng đợi ưu tiên (Min-Priority Queue / Min-Heap)",
                "C. Cây nhị phân tìm kiếm cân bằng đỏ đen (Red-Black Tree)",
                "D. Bảng băm đóng (Closed Hash Table)",
            ],
            correct_choice="B. Hàng đợi ưu tiên (Min-Priority Queue / Min-Heap)",
            user_choice="B. Hàng đợi ưu tiên (Min-Priority Queue / Min-Heap)",
            is_correct=True,
            created_at=datetime.utcnow() - timedelta(days=1),
        )
        db.add(quiz1)
        db.commit()
        print("✅ Đã tạo các bản ghi Quiz lịch sử.")

        print("=" * 70)
        print("🎉 HOÀN THÀNH SEED DỮ LIỆU DEMO THÀNH CÔNG!")
        print(f"👉 Người dùng mẫu: {user.email}")
        print(f"👉 Session ID: {session_id}")
        print("=" * 70)

    except Exception as e:
        db.rollback()
        print(f"❌ Lỗi khi seed dữ liệu: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_demo_data()
