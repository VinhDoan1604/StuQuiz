# Báo cáo đánh giá StudyRAG — run `final-v1`

- Ngày: 2026-11-10 18:02 · commit: `70a5914-dirty`
- Dataset: **81 câu** (58 có đáp án, 23 không có đáp án)
- Corpus: 18 tài liệu · embedding `BAAI/bge-m3` · top-k=5
- Reranker: `BAAI/bge-reranker-v2-m3` · 10 ứng viên · max_length=384 · int8=True · device=auto
- Model sinh/chấm trong pipeline: `shopaikey_google/gemini-2.5-flash` · Judge: `shopaikey_openai/gpt-5.4`
- Khoảng tin cậy: percentile bootstrap 95% (2000 lần); baseline = `dense`

## 1. Retrieval (chỉ câu có đáp án)

| Cấu hình | hit@1 | hit@3 | hit@5 | recall@5 | mrr |
|---|---|---|---|---|---|
| `dense` | 0.586 [0.466–0.724] (n=58) | 0.879 [0.793–0.948] (n=58) | 0.966 [0.914–1.000] (n=58) | 0.931 [0.871–0.983] (n=58) | 0.735 [0.651–0.822] (n=58) |
| `production` | 0.741 [0.621–0.845] (n=58) | 0.931 [0.862–0.983] (n=58) | 0.983 [0.948–1.000] (n=58) | 0.948 [0.897–0.991] (n=58) | 0.846 [0.776–0.909] (n=58) |
| `crag` | 0.741 [0.621–0.845] (n=58) | 0.931 [0.862–0.983] (n=58) | 0.983 [0.948–1.000] (n=58) | 0.948 [0.897–0.991] (n=58) | 0.846 [0.776–0.909] (n=58) |
| `full` | 0.741 [0.621–0.845] (n=58) | 0.931 [0.862–0.983] (n=58) | 0.983 [0.948–1.000] (n=58) | 0.948 [0.897–0.991] (n=58) | 0.846 [0.776–0.909] (n=58) |

*Chấm trên chunk con (200-400 token) trước khi mở rộng parent, để mọi cấu hình cùng đơn vị so sánh.*

### 1b. Bằng chứng có trong context gửi LLM

| Cấu hình | hit@5 | recall@5 |
|---|---|---|
| `dense` | 0.966 [0.914–1.000] (n=58) | 0.931 [0.871–0.983] (n=58) |
| `production` | 0.983 [0.948–1.000] (n=58) | 0.966 [0.922–1.000] (n=58) |
| `crag` | 0.983 [0.948–1.000] (n=58) | 0.966 [0.922–1.000] (n=58) |
| `full` | 0.983 [0.948–1.000] (n=58) | 0.966 [0.922–1.000] (n=58) |

*Với `parent`/`crag`/`full`, context là đoạn cha (tới 1500 token) nên dễ chứa bằng chứng hơn — không dùng bảng này để so retrieval giữa các cấu hình.*

## 2. Chất lượng câu trả lời

| Cấu hình | Faithfulness ↑ | Correctness ↑ | Từ chối nhầm ↓ | Từ chối đúng (câu không có đáp án) ↑ | Trả lời đúng ngôn ngữ (câu hỏi tiếng Anh) ↑ |
|---|---|---|---|---|---|
| `dense` | 0.949 [0.924–0.972] (n=58) | 0.957 [0.905–0.991] (n=58) | 0.034 [0.000–0.086] (n=58) | 0.957 [0.870–1.000] (n=23) | 1.000 [1.000–1.000] (n=15) |
| `production` | 0.968 [0.945–0.988] (n=58) | 0.957 [0.905–0.991] (n=58) | 0.034 [0.000–0.086] (n=58) | 1.000 [1.000–1.000] (n=22) | 1.000 [1.000–1.000] (n=15) |
| `crag` | 0.952 [0.925–0.973] (n=58) | 0.940 [0.888–0.983] (n=58) | 0.034 [0.000–0.086] (n=58) | 0.957 [0.870–1.000] (n=23) | 0.933 [0.800–1.000] (n=15) |
| `full` | 0.963 [0.941–0.982] (n=58) | 0.957 [0.905–0.991] (n=58) | 0.017 [0.000–0.052] (n=58) | 0.957 [0.870–1.000] (n=23) | 1.000 [1.000–1.000] (n=15) |

*Faithfulness chỉ tính trên câu trả lời có ≥1 khẳng định (câu từ chối không tính; chúng được đo ở cột 'Từ chối').*

## 3. Chi phí & độ trễ

| Cấu hình | Latency p50 (s) | p95 (s) | LLM calls TB | Token TB | Vòng lặp TB | Tỉ lệ lỗi |
|---|---|---|---|---|---|---|
| `dense` | 4.61 | 17.00 | 1.0 | 2516 | 0.00 | 0.0% (0/81) |
| `production` | 10.31 | 22.82 | 1.0 | 3019 | 0.00 | 1.2% (1/81) |
| `crag` | 14.10 | 54.27 | 2.7 | 6087 | 0.33 | 0.0% (0/81) |
| `full` | 17.75 | 39.20 | 3.7 | 8923 | 0.31 | 0.0% (0/81) |

*Latency đo tuần tự trên 1 máy, sau warm-up, không cache embedding giữa các lần chạy.*

## 4. Chênh lệch so với baseline `dense` (bootstrap ghép cặp)

| Cấu hình | Chỉ số | Δ | 95% CI | n cặp | Kết luận |
|---|---|---|---|---|---|
| `production` | hit@5 | +0.017 | [+0.000, +0.052] | 58 | chưa đủ bằng chứng |
| `production` | mrr | +0.111 | [+0.035, +0.189] | 58 | khác biệt có ý nghĩa |
| `production` | faithfulness | +0.019 | [-0.004, +0.046] | 58 | chưa đủ bằng chứng |
| `production` | correctness | +0.000 | [-0.026, +0.026] | 58 | chưa đủ bằng chứng |
| `production` | abstain_correct | +0.045 | [+0.000, +0.136] | 22 | chưa đủ bằng chứng |
| `crag` | hit@5 | +0.017 | [+0.000, +0.052] | 58 | chưa đủ bằng chứng |
| `crag` | mrr | +0.111 | [+0.035, +0.189] | 58 | khác biệt có ý nghĩa |
| `crag` | faithfulness | +0.003 | [-0.027, +0.030] | 58 | chưa đủ bằng chứng |
| `crag` | correctness | -0.017 | [-0.052, +0.017] | 58 | chưa đủ bằng chứng |
| `crag` | abstain_correct | +0.000 | [+0.000, +0.000] | 23 | chưa đủ bằng chứng |
| `full` | hit@5 | +0.017 | [+0.000, +0.052] | 58 | chưa đủ bằng chứng |
| `full` | mrr | +0.111 | [+0.035, +0.189] | 58 | khác biệt có ý nghĩa |
| `full` | faithfulness | +0.015 | [-0.007, +0.038] | 58 | chưa đủ bằng chứng |
| `full` | correctness | +0.000 | [-0.026, +0.026] | 58 | chưa đủ bằng chứng |
| `full` | abstain_correct | +0.000 | [+0.000, +0.000] | 23 | chưa đủ bằng chứng |

## 5. Phân tích lỗi (`full`)

- Retrieval trượt hoàn toàn (hit@5=0): **1** câu: v2s11
- hit@5 theo ngôn ngữ câu hỏi: en: 1.000 [1.000–1.000] (n=10), vi: 0.979 [0.938–1.000] (n=48)
- Trả lời sai hẳn (correctness=0): **1** câu: v2s11
- Judge lỗi (không chấm được, đã loại khỏi trung bình): 0 câu

## 6. Hạn chế

- Cỡ mẫu nhỏ: CI rộng; chênh lệch nhỏ không nên được diễn giải là cải thiện.
- Judge là LLM, chưa hiệu chuẩn tuyệt đối — xem `audit.py` để đo độ đồng thuận với người chấm.
- Câu hỏi sinh tự động có xu hướng trùng từ vựng với tài liệu (retrieval dễ hơn thực tế).
- Orchestrator bị bỏ qua ở mọi cấu hình (intent cố định 'qa').
