# STATUS: STUB — CẦN ĐẦU TƯ KỸ
"""
UI Streamlit — điểm quan trọng nhất về UX: hiển thị Reflection trace
RA MÀN HÌNH (mục 3), ví dụ dạng expander/timeline:
  🔍 Tìm context... (score: 0.42, chưa đủ)
  ✏️ Viết lại câu hỏi: "..."
  🔍 Tìm lại... (score: 0.81, đủ)
  ✍️ Sinh câu trả lời...
  ✅ Kiểm tra câu trả lời: đạt (faithfulness: 0.9)

Các trang dự kiến: Upload tài liệu, Chat (QA/Teaching), Quiz, Mastery
dashboard (biểu đồ mastery_score theo topic + topic nào due để ôn).
Gọi thẳng các endpoint FastAPI trong app/api/*.
"""
import streamlit as st

st.set_page_config(page_title="StudyRAG", layout="wide")
st.title("StudyRAG")
st.info("STUB — sẽ implement sau khi backend (agents + graph) chạy được end-to-end.")
