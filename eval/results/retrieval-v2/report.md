# Báo cáo đánh giá StudyRAG — run `retrieval-v2`

- Ngày: 2026-11-06 14:50 · commit: `67f98c5`
- Dataset: **48 câu** (37 có đáp án, 11 không có đáp án)
- Corpus: 18 tài liệu · embedding `BAAI/bge-m3` · top-k=5
- Khoảng tin cậy: percentile bootstrap 95% (2000 lần); baseline = `dense`

## 1. Retrieval (chỉ câu có đáp án)

| Cấu hình | hit@1 | hit@3 | hit@5 | recall@5 | mrr |
|---|---|---|---|---|---|
| `dense` | 0.541 [0.378–0.703] (n=37) | 0.865 [0.757–0.973] (n=37) | 0.973 [0.919–1.000] (n=37) | 0.919 [0.838–0.973] (n=37) | 0.710 [0.602–0.815] (n=37) |
| `hybrid` | 0.568 [0.405–0.730] (n=37) | 0.811 [0.676–0.919] (n=37) | 0.892 [0.784–0.973] (n=37) | 0.865 [0.743–0.959] (n=37) | 0.691 [0.568–0.811] (n=37) |
| `hybrid_rerank` | 0.784 [0.649–0.919] (n=37) | 0.973 [0.919–1.000] (n=37) | 1.000 [1.000–1.000] (n=37) | 0.946 [0.892–0.986] (n=37) | 0.881 [0.806–0.953] (n=37) |
| `parent` | 0.784 [0.649–0.919] (n=37) | 0.973 [0.919–1.000] (n=37) | 1.000 [1.000–1.000] (n=37) | 0.946 [0.892–0.986] (n=37) | 0.881 [0.806–0.953] (n=37) |

*Chấm trên chunk con (200-400 token) trước khi mở rộng parent, để mọi cấu hình cùng đơn vị so sánh.*

### 1b. Bằng chứng có trong context gửi LLM

| Cấu hình | hit@5 | recall@5 |
|---|---|---|
| `dense` | 0.973 [0.919–1.000] (n=37) | 0.919 [0.838–0.973] (n=37) |
| `hybrid` | 0.892 [0.784–0.973] (n=37) | 0.865 [0.743–0.959] (n=37) |
| `hybrid_rerank` | 1.000 [1.000–1.000] (n=37) | 0.946 [0.892–0.986] (n=37) |
| `parent` | 1.000 [1.000–1.000] (n=37) | 0.973 [0.932–1.000] (n=37) |

*Với `parent`/`crag`/`full`, context là đoạn cha (tới 1500 token) nên dễ chứa bằng chứng hơn — không dùng bảng này để so retrieval giữa các cấu hình.*

## 3. Chi phí & độ trễ

| Cấu hình | Latency p50 (s) | p95 (s) | LLM calls TB | Token TB | Vòng lặp TB | Tỉ lệ lỗi |
|---|---|---|---|---|---|---|
| `dense` | 2.85 | 3.00 | 0.0 | n/a | 0.00 | 0.0% (0/48) |
| `hybrid` | 2.81 | 2.93 | 0.0 | n/a | 0.00 | 0.0% (0/48) |
| `hybrid_rerank` | 37.46 | 50.22 | 0.0 | n/a | 0.00 | 0.0% (0/48) |
| `parent` | 37.27 | 58.33 | 0.0 | n/a | 0.00 | 0.0% (0/48) |

*Latency đo tuần tự trên 1 máy, sau warm-up, không cache embedding giữa các lần chạy.*

## 4. Chênh lệch so với baseline `dense` (bootstrap ghép cặp)

| Cấu hình | Chỉ số | Δ | 95% CI | n cặp | Kết luận |
|---|---|---|---|---|---|
| `hybrid` | hit@5 | -0.081 | [-0.189, +0.000] | 37 | chưa đủ bằng chứng |
| `hybrid` | mrr | -0.019 | [-0.079, +0.045] | 37 | chưa đủ bằng chứng |
| `hybrid_rerank` | hit@5 | +0.027 | [+0.000, +0.081] | 37 | chưa đủ bằng chứng |
| `hybrid_rerank` | mrr | +0.170 | [+0.086, +0.260] | 37 | khác biệt có ý nghĩa |
| `parent` | hit@5 | +0.027 | [+0.000, +0.081] | 37 | chưa đủ bằng chứng |
| `parent` | mrr | +0.170 | [+0.086, +0.260] | 37 | khác biệt có ý nghĩa |

## 5. Phân tích lỗi (`parent`)

- Retrieval trượt hoàn toàn (hit@5=0): **0** câu: —
- hit@5 theo ngôn ngữ câu hỏi: en: 1.000 [1.000–1.000] (n=8), vi: 1.000 [1.000–1.000] (n=29)

## 6. Hạn chế

- Cỡ mẫu nhỏ: CI rộng; chênh lệch nhỏ không nên được diễn giải là cải thiện.
- Judge là LLM, chưa hiệu chuẩn tuyệt đối — xem `audit.py` để đo độ đồng thuận với người chấm.
- Câu hỏi sinh tự động có xu hướng trùng từ vựng với tài liệu (retrieval dễ hơn thực tế).
- Orchestrator bị bỏ qua ở mọi cấu hình (intent cố định 'qa').
