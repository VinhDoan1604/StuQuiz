"""
app/agents/constants.py

Ngưỡng dùng chung giữa reflection_agent.py (nơi CHẤM điểm) và
generation_agent.py / retrieval_agent.py (nơi ĐỌC LẠI điểm để quyết định
có cần feedback/regenerate hay không). Tách riêng để 2 phía luôn đồng bộ
— tránh tình huống reflection_agent coi 0.65 là "đạt" nhưng generation_agent
lại đọc feedback với ngưỡng 0.7 rồi hiểu nhầm là "chưa đạt".
"""

# grade_context() coi context đủ dùng nếu (relevance + sufficiency) / 2 >= ngưỡng này.
CONTEXT_SUFFICIENT_THRESHOLD = 0.6

# grade_answer() coi câu trả lời đạt (faithful) nếu faithfulness >= ngưỡng này.
# Cố tình đặt CAO HƠN CONTEXT_SUFFICIENT_THRESHOLD vì faithfulness (chống
# hallucination) là tiêu chí quan trọng nhất của Self-RAG — thà chấp nhận
# context hơi thiếu (0.6) còn hơn chấp nhận câu trả lời bịa đặt (cần 0.7+).
ANSWER_ACCEPT_THRESHOLD = 0.7
