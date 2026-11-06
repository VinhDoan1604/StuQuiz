# Báo cáo đánh giá StudyRAG — run `full-v1`

- Ngày: 2026-11-03 23:50 · commit: `89e8184`
- Dataset: **48 câu** (37 có đáp án, 11 không có đáp án)
- Corpus: 18 tài liệu · embedding `BAAI/bge-m3` · top-k=5
- Model sinh/chấm trong pipeline: `shopaikey_google/gemini-2.5-flash` · Judge: `shopaikey_openai/gpt-5.4`
- Khoảng tin cậy: percentile bootstrap 95% (2000 lần); baseline = `dense`

## 1. Retrieval (chỉ câu có đáp án)

| Cấu hình | hit@1 | hit@3 | hit@5 | recall@5 | mrr |
|---|---|---|---|---|---|
| `dense` | 0.541 [0.378–0.703] (n=37) | 0.838 [0.703–0.946] (n=37) | 0.946 [0.865–1.000] (n=37) | 0.905 [0.824–0.973] (n=37) | 0.697 [0.583–0.807] (n=37) |
| `hybrid` | 0.514 [0.351–0.676] (n=37) | 0.811 [0.676–0.919] (n=37) | 0.919 [0.811–1.000] (n=37) | 0.892 [0.784–0.973] (n=37) | 0.670 [0.550–0.785] (n=37) |
| `hybrid_rerank` | 0.784 [0.649–0.919] (n=37) | 0.946 [0.865–1.000] (n=37) | 0.973 [0.919–1.000] (n=37) | 0.932 [0.865–0.986] (n=37) | 0.867 [0.784–0.946] (n=37) |
| `parent` | 0.784 [0.649–0.919] (n=37) | 0.946 [0.865–1.000] (n=37) | 0.973 [0.919–1.000] (n=37) | 0.932 [0.865–0.986] (n=37) | 0.867 [0.784–0.946] (n=37) |
| `crag` | 0.784 [0.649–0.919] (n=37) | 0.946 [0.865–1.000] (n=37) | 0.973 [0.919–1.000] (n=37) | 0.932 [0.865–0.986] (n=37) | 0.867 [0.784–0.946] (n=37) |
| `full` | 0.757 [0.622–0.892] (n=37) | 0.946 [0.865–1.000] (n=37) | 0.973 [0.919–1.000] (n=37) | 0.932 [0.865–0.986] (n=37) | 0.849 [0.757–0.932] (n=37) |

*Chấm trên chunk con (200-400 token) trước khi mở rộng parent, để mọi cấu hình cùng đơn vị so sánh.*

### 1b. Bằng chứng có trong context gửi LLM

| Cấu hình | hit@5 | recall@5 |
|---|---|---|
| `dense` | 0.946 [0.865–1.000] (n=37) | 0.905 [0.824–0.973] (n=37) |
| `hybrid` | 0.919 [0.811–1.000] (n=37) | 0.892 [0.784–0.973] (n=37) |
| `hybrid_rerank` | 0.973 [0.919–1.000] (n=37) | 0.932 [0.865–0.986] (n=37) |
| `parent` | 1.000 [1.000–1.000] (n=37) | 0.973 [0.932–1.000] (n=37) |
| `crag` | 1.000 [1.000–1.000] (n=37) | 0.973 [0.932–1.000] (n=37) |
| `full` | 1.000 [1.000–1.000] (n=37) | 0.973 [0.932–1.000] (n=37) |

*Với `parent`/`crag`/`full`, context là đoạn cha (tới 1500 token) nên dễ chứa bằng chứng hơn — không dùng bảng này để so retrieval giữa các cấu hình.*

## 2. Chất lượng câu trả lời

| Cấu hình | Faithfulness ↑ | Correctness ↑ | Từ chối nhầm ↓ | Từ chối đúng (câu không có đáp án) ↑ | Trả lời đúng ngôn ngữ (câu hỏi tiếng Anh) ↑ |
|---|---|---|---|---|---|
| `dense` | 0.916 [0.878–0.947] (n=37) | 0.959 [0.905–1.000] (n=37) | 0.027 [0.000–0.081] (n=37) | 1.000 [1.000–1.000] (n=11) | 1.000 [1.000–1.000] (n=11) |
| `hybrid` | 0.913 [0.873–0.950] (n=37) | 0.946 [0.865–1.000] (n=37) | 0.054 [0.000–0.135] (n=37) | 1.000 [1.000–1.000] (n=11) | 1.000 [1.000–1.000] (n=11) |
| `hybrid_rerank` | 0.893 [0.856–0.925] (n=37) | 0.946 [0.892–0.986] (n=37) | 0.054 [0.000–0.135] (n=37) | 1.000 [1.000–1.000] (n=11) | 0.818 [0.545–1.000] (n=11) |
| `parent` | 0.905 [0.869–0.940] (n=37) | 0.973 [0.932–1.000] (n=37) | 0.027 [0.000–0.081] (n=37) | 1.000 [1.000–1.000] (n=11) | 1.000 [1.000–1.000] (n=11) |
| `crag` | 0.890 [0.852–0.925] (n=37) | 0.973 [0.932–1.000] (n=37) | 0.000 [0.000–0.000] (n=37) | 1.000 [1.000–1.000] (n=11) | 1.000 [1.000–1.000] (n=11) |
| `full` | 0.874 [0.828–0.915] (n=37) | 0.986 [0.959–1.000] (n=37) | 0.000 [0.000–0.000] (n=37) | 1.000 [1.000–1.000] (n=11) | 0.909 [0.727–1.000] (n=11) |

*Faithfulness chỉ tính trên câu trả lời có ≥1 khẳng định (câu từ chối không tính; chúng được đo ở cột 'Từ chối').*

## 3. Chi phí & độ trễ

| Cấu hình | Latency p50 (s) | p95 (s) | LLM calls TB | Token TB | Vòng lặp TB | Tỉ lệ lỗi |
|---|---|---|---|---|---|---|
| `dense` | 5.16 | 8.02 | 1.0 | 2848 | 0.00 | 0.0% (0/48) |
| `hybrid` | 5.36 | 7.99 | 1.0 | 2885 | 0.00 | 0.0% (0/48) |
| `hybrid_rerank` | 42.80 | 58.15 | 1.0 | 2868 | 0.00 | 0.0% (0/48) |
| `parent` | 41.66 | 54.21 | 1.0 | 3385 | 0.00 | 0.0% (0/48) |
| `crag` | 49.00 | 139.88 | 2.9 | 7313 | 0.42 | 0.0% (0/48) |
| `full` | 50.31 | 166.47 | 4.1 | 10719 | 0.48 | 0.0% (0/48) |

*Latency đo tuần tự trên 1 máy, sau warm-up, không cache embedding giữa các lần chạy.*

## 4. Chênh lệch so với baseline `dense` (bootstrap ghép cặp)

| Cấu hình | Chỉ số | Δ | 95% CI | n cặp | Kết luận |
|---|---|---|---|---|---|
| `hybrid` | hit@5 | -0.027 | [-0.081, +0.000] | 37 | chưa đủ bằng chứng |
| `hybrid` | mrr | -0.027 | [-0.092, +0.036] | 37 | chưa đủ bằng chứng |
| `hybrid` | faithfulness | -0.002 | [-0.046, +0.043] | 37 | chưa đủ bằng chứng |
| `hybrid` | correctness | -0.014 | [-0.068, +0.027] | 37 | chưa đủ bằng chứng |
| `hybrid` | abstain_correct | +0.000 | [+0.000, +0.000] | 11 | chưa đủ bằng chứng |
| `hybrid_rerank` | hit@5 | +0.027 | [+0.000, +0.081] | 37 | chưa đủ bằng chứng |
| `hybrid_rerank` | mrr | +0.170 | [+0.086, +0.260] | 37 | khác biệt có ý nghĩa |
| `hybrid_rerank` | faithfulness | -0.023 | [-0.057, +0.012] | 37 | chưa đủ bằng chứng |
| `hybrid_rerank` | correctness | -0.014 | [-0.081, +0.041] | 37 | chưa đủ bằng chứng |
| `hybrid_rerank` | abstain_correct | +0.000 | [+0.000, +0.000] | 11 | chưa đủ bằng chứng |
| `parent` | hit@5 | +0.027 | [+0.000, +0.081] | 37 | chưa đủ bằng chứng |
| `parent` | mrr | +0.170 | [+0.086, +0.260] | 37 | khác biệt có ý nghĩa |
| `parent` | faithfulness | -0.010 | [-0.060, +0.037] | 37 | chưa đủ bằng chứng |
| `parent` | correctness | +0.014 | [-0.027, +0.054] | 37 | chưa đủ bằng chứng |
| `parent` | abstain_correct | +0.000 | [+0.000, +0.000] | 11 | chưa đủ bằng chứng |
| `crag` | hit@5 | +0.027 | [+0.000, +0.081] | 37 | chưa đủ bằng chứng |
| `crag` | mrr | +0.170 | [+0.086, +0.260] | 37 | khác biệt có ý nghĩa |
| `crag` | faithfulness | -0.025 | [-0.070, +0.018] | 37 | chưa đủ bằng chứng |
| `crag` | correctness | +0.014 | [-0.027, +0.054] | 37 | chưa đủ bằng chứng |
| `crag` | abstain_correct | +0.000 | [+0.000, +0.000] | 11 | chưa đủ bằng chứng |
| `full` | hit@5 | +0.027 | [+0.000, +0.081] | 37 | chưa đủ bằng chứng |
| `full` | mrr | +0.152 | [+0.058, +0.252] | 37 | khác biệt có ý nghĩa |
| `full` | faithfulness | -0.041 | [-0.095, +0.007] | 37 | chưa đủ bằng chứng |
| `full` | correctness | +0.027 | [-0.027, +0.081] | 37 | chưa đủ bằng chứng |
| `full` | abstain_correct | +0.000 | [+0.000, +0.000] | 11 | chưa đủ bằng chứng |

## 5. Phân tích lỗi (`full`)

- Retrieval trượt hoàn toàn (hit@5=0): **1** câu: q064
- hit@5 theo ngôn ngữ câu hỏi: en: 0.875 [0.625–1.000] (n=8), vi: 1.000 [1.000–1.000] (n=29)
- Trả lời sai hẳn (correctness=0): **0** câu: —
- Judge lỗi (không chấm được, đã loại khỏi trung bình): 0 câu

## 6. Hạn chế

- Cỡ mẫu nhỏ: CI rộng; chênh lệch nhỏ không nên được diễn giải là cải thiện.
- Judge là LLM, chưa hiệu chuẩn tuyệt đối — xem `audit.py` để đo độ đồng thuận với người chấm.
- Câu hỏi sinh tự động có xu hướng trùng từ vựng với tài liệu (retrieval dễ hơn thực tế).
- Orchestrator bị bỏ qua ở mọi cấu hình (intent cố định 'qa').
