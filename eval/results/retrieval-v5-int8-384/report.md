# Báo cáo đánh giá StudyRAG — run `retrieval-v5-int8-384`

- Ngày: 2026-11-10 14:08 · commit: `874b6cd`
- Dataset: **81 câu** (58 có đáp án, 23 không có đáp án)
- Corpus: 18 tài liệu · embedding `BAAI/bge-m3` · top-k=5
- Reranker: `BAAI/bge-reranker-v2-m3` · 10 ứng viên · max_length=384 · int8=True · device=auto
- Khoảng tin cậy: percentile bootstrap 95% (2000 lần); baseline = `dense_rerank`

## 1. Retrieval (chỉ câu có đáp án)

| Cấu hình | hit@1 | hit@3 | hit@5 | recall@5 | mrr |
|---|---|---|---|---|---|
| `dense_rerank` | 0.741 [0.621–0.845] (n=58) | 0.931 [0.862–0.983] (n=58) | 0.983 [0.948–1.000] (n=58) | 0.948 [0.897–0.991] (n=58) | 0.846 [0.776–0.909] (n=58) |

*Chấm trên chunk con (200-400 token) trước khi mở rộng parent, để mọi cấu hình cùng đơn vị so sánh.*

### 1b. Bằng chứng có trong context gửi LLM

| Cấu hình | hit@5 | recall@5 |
|---|---|---|
| `dense_rerank` | 0.983 [0.948–1.000] (n=58) | 0.948 [0.897–0.991] (n=58) |

*Với `parent`/`crag`/`full`, context là đoạn cha (tới 1500 token) nên dễ chứa bằng chứng hơn — không dùng bảng này để so retrieval giữa các cấu hình.*

## 3. Chi phí & độ trễ

| Cấu hình | Latency p50 (s) | p95 (s) | LLM calls TB | Token TB | Vòng lặp TB | Tỉ lệ lỗi |
|---|---|---|---|---|---|---|
| `dense_rerank` | 8.40 | 9.41 | 0.0 | n/a | 0.00 | 0.0% (0/81) |

*Latency đo tuần tự trên 1 máy, sau warm-up, không cache embedding giữa các lần chạy.*

## 4. Chênh lệch so với baseline `dense_rerank` (bootstrap ghép cặp)

| Cấu hình | Chỉ số | Δ | 95% CI | n cặp | Kết luận |
|---|---|---|---|---|---|

## 5. Phân tích lỗi (`dense_rerank`)

- Retrieval trượt hoàn toàn (hit@5=0): **1** câu: v2s11
- hit@5 theo ngôn ngữ câu hỏi: en: 1.000 [1.000–1.000] (n=10), vi: 0.979 [0.938–1.000] (n=48)

## 6. Hạn chế

- Cỡ mẫu nhỏ: CI rộng; chênh lệch nhỏ không nên được diễn giải là cải thiện.
- Judge là LLM, chưa hiệu chuẩn tuyệt đối — xem `audit.py` để đo độ đồng thuận với người chấm.
- Câu hỏi sinh tự động có xu hướng trùng từ vựng với tài liệu (retrieval dễ hơn thực tế).
- Orchestrator bị bỏ qua ở mọi cấu hình (intent cố định 'qa').
