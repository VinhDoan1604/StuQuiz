# Cải thiện hệ thống sau eval `full-v1`: đã làm / chưa làm

Ngày 2026-11-07. Commit gốc `918f58c`; các thay đổi bên dưới **chưa commit**.
Kiểm tra: `pytest` 120/120 pass (có 13 test mới trong `tests/test_eval_followup.py`), `ruff` sạch.

Chú thích trạng thái: ✅ xong và đã đo · 🟡 xong code, chưa đo (cần LLM, tốn credit) · ⬜ chưa làm · 👤 cần bạn làm hoặc quyết.

---

## Tóm tắt

| Hạng mục | Trạng thái | Kết quả |
|---|---|---|
| Giữ reranker | ✅ | Mặc định `RERANK_ENABLED=true` |
| Xem lại sparse/hybrid | ✅ | Khi đã rerank, sparse không giúp gì → mặc định mới `RETRIEVAL_MODE=dense` |
| Bật/tắt CRAG và critic | ✅ | `ENABLE_CRAG`, `ENABLE_ANSWER_CRITIC`, **mặc định tắt** (người dùng quyết 07/11/2026) |
| Giảm độ trễ rerank | ✅ | p50 **37,4s → 12,0s** (p95 48,4s → 14,8s), MRR giữ nguyên 0,881 |
| Prompt: câu từ chối không gắn tiếng Việt | 🟡 | Đã sửa; cần chạy LLM để xác nhận |
| Prompt: giới hạn mục "💡 Lưu ý / Ứng dụng" | 🟡 | Đã sửa; cần chạy LLM để đo faithfulness |
| Critic chặt hơn | 🟡 | Chấm theo từng khẳng định như judge; có thể dùng model riêng |
| Dataset v2 | ✅ | 33 câu mới, người dùng đã duyệt → tổng 81 câu; retrieval-only trên v2: mục 4.1 |
| Chạy lại eval chỉ với các cấu hình cần so sánh | ✅ một phần | `crag-hard-v2` (mục 4.3); còn lại: `lang-fix` lặp lại, `CRITIC_MODEL_*` khác họ |

---

## 1. Chọn cấu hình mặc định cho production

### 1.1 Reranker: giữ ✅
`RERANK_ENABLED=true`. Có thể tắt để debug; khi tắt, search chỉ lấy đúng top-k.

### 1.2 Sparse/hybrid: chuyển sang dense ✅
Thêm cấu hình eval `dense_rerank` (dense top-N → rerank) để so trực tiếp với `hybrid_rerank`,
và thêm `RETRIEVAL_MODE=dense|hybrid` cho production (`qdrant_store.dense_search`, `helpers.asearch`).

Kết quả retrieval-only, 37 câu có đáp án, đều đo lại sau khi sửa lỗi luồng ở mục 2:

| Cấu hình | Ứng viên rerank | MRR | hit@1 | hit@5 | recall@5 |
|---|---|---|---|---|---|
| `dense` (không rerank) | — | 0,710 | 0,541 | 0,973 | 0,919 |
| `hybrid_rerank` | 20 | 0,881 | 0,784 | 1,000 | 0,946 |
| `dense_rerank` | 20 | 0,881 | 0,784 | 1,000 | 0,959 |
| `hybrid_rerank` | 10 | 0,854 | 0,757 | 0,973 | 0,919 |
| **`dense_rerank`** | **10** | **0,881** | **0,784** | **1,000** | **0,946** |

- Ở top-20, dense và hybrid ngang nhau. Ở top-10, dense còn tốt hơn: thêm sparse đẩy ứng viên
  nhiễu vào danh sách vốn đã ngắn.
- Mọi chênh lệch giữa các biến thể rerank đều nằm trong CI. Kết luận đúng là "dense không kém
  hybrid", **không phải** "dense tốt hơn hybrid". Chỉ khoảng cách so với `dense` không rerank
  là có ý nghĩa thống kê (MRR +0,17, CI [+0,08, +0,26]).
- Giả thuyết câu hỏi khác ngôn ngữ với tài liệu (sparse là so khớp từ vựng) **chưa được kiểm
  riêng**. Bằng chứng ở đây chỉ là sparse không giúp gì trên toàn bộ tập.

→ **Mặc định mới: `RETRIEVAL_MODE=dense`, `RERANK_CANDIDATES=10`.**
Index vẫn giữ vector sparse, nên muốn quay lại chỉ cần đặt `RETRIEVAL_MODE=hybrid`.

### 1.3 Bật/tắt CRAG và critic ✅ · mặc định tắt
- `ENABLE_CRAG=false`: sau retrieval sinh luôn, bỏ `grade_context` và `rewrite_query`.
- `ENABLE_ANSWER_CRITIC=false`: sau generation chấp nhận luôn, bỏ `grade_answer` và `retry_generation`.
- Tắt cả hai thì mỗi câu chỉ còn 1 lời gọi LLM. Có test kiểm (`test_graph_with_everything_off_makes_one_llm_call`).
- Các cấu hình eval `crag`/`full` vẫn tự nối node như cũ, nên kết quả ablation không phụ thuộc hai cờ này.

**Mặc định đã chuyển sang tắt** (`ENABLE_CRAG=false`, `ENABLE_ANSWER_CRITIC=false`, người dùng
quyết 07/11/2026). Lý do: theo full-v1 chúng tốn 3–4 lần token, p95 tới ~160s, và chưa cải
thiện có ý nghĩa thống kê ngoài tỉ lệ từ chối nhầm, vốn chỉ 1–2/37 câu. Critic mới chặt hơn
(mục 3.3) nên có thể còn đắt hơn. Bật lại bằng biến môi trường.
- `tests/test_graph.py` có fixture bật cả hai cờ, vì file đó kiểm đồ thị đầy đủ.
- Đây là "chưa chứng minh được lợi ích trên dataset hiện tại", **không phải** "CRAG/critic vô dụng":
  dataset v1 có hit@5 = 1,0 sẵn nên CRAG không có gì để sửa. Cách kiểm cho công bằng: mục 4.1.

---

## 2. Giảm độ trễ rerank ✅

| Thay đổi | File | Tác dụng đo được |
|---|---|---|
| **Sửa lỗi số luồng torch trong thread phụ.** Production và eval gọi rerank qua `asyncio.to_thread`; trong thread đó torch chỉ dùng 4 luồng thay vì 8 | `reranker._ensure_torch_threads`, `RERANKER_THREADS` | 20 ứng viên: **37,1s → 20,3s** |
| Tự viết vòng chấm điểm, bỏ lần forward "dò batch" thừa của FlagEmbedding 1.4.x | `reranker._score_pairs` | ~20–30%; điểm khớp bản gốc (chênh ≤ 6e-7) |
| Số ứng viên cấu hình được, mặc định 20 → **10** | `RERANK_CANDIDATES` | ~1,7× |
| Lượng tử hoá int8 (torch dynamic quantization, không cần ONNX) | `RERANKER_INT8` | thêm ~1,3× trong eval |
| Đổi model / device / max_length qua cấu hình | `RERANKER_MODEL`, `RERANKER_DEVICE`, `RERANKER_MAX_LENGTH` | fp16 tự bật khi device là cuda |

Độ trễ retrieval từ đầu đến cuối (embed + search + rerank), 48 câu, chạy tuần tự trên CPU:

| Run | Cấu hình | p50 | p95 | MRR |
|---|---|---|---|---|
| `full-v1` / `retrieval-v2` (cũ) | hybrid, top-20 | 37–43s | 50–58s | 0,867–0,881 |
| `retrieval-v4` | **dense, top-10 (mặc định mới)** | **12,0s** | **14,8s** | **0,881** |
| `retrieval-v4-int8` | dense, top-10, int8 | 9,1s | 13,0s | 0,878 |

- `scripts/bench_reranker.py` đo độ trễ từng biến thể, không cần Qdrant hay LLM. Kết quả:
  `eval/results/rerank-bench-v1.json`.
- Các run `retrieval-v3-*` (top-20, top-10, int8-top20, int8-top10) cho số chất lượng đúng,
  nhưng **số độ trễ sai** (gấp ~1,8 lần) vì chạy trước khi sửa lỗi luồng. Đừng trích dẫn latency từ các run đó.
- **int8 chưa bật mặc định.** MRR gần như không đổi (0,878 so với 0,881), nhưng benchmark cho
  thấy top-1 lệch so với fp32 ở 15% số câu. Lợi thêm ~3s không đáng để mặc định; muốn bật thì đặt `RERANKER_INT8=true`.
- ⬜ **Chưa làm:** reranker nhỏ hơn (`bge-reranker-base` yếu với tiếng Việt; `jina-reranker-v2-base-multilingual`
  không chạy qua FlagReranker), ONNX/optimum, rerank trên GPU hoặc dịch vụ ngoài. Máy dev không có GPU.
- ⬜ 12s/câu vẫn chậm cho demo công khai. Cách rẻ tiếp theo: chạy `RERANKER_DEVICE=cuda:0` trên
  máy có GPU (fp16 tự bật), hoặc thử `RERANKER_MAX_LENGTH=384`. Benchmark int8-10-384 ra ~5,7s
  nhưng độ trùng top-5 giảm còn 0,86, phải đo lại retrieval trước khi dùng.
- Phát hiện thêm: embedder BGE-M3 cũng chạy qua `to_thread` (~2,5s/câu), nhiều khả năng dính
  cùng lỗi số luồng. Chưa sửa, chưa đo.

---

## 3. Sửa prompt

### 3.1 Câu từ chối không gắn tiếng Việt 🟡
Nguyên nhân đã xác nhận trong `full-v1/raw.jsonl`: 3 câu trả lời cho câu hỏi tiếng Anh không có
đáp án (q065 ở `hybrid_rerank` và `full`, q067 ở `hybrid_rerank`) chép nguyên câu mẫu
*"Tài liệu chưa cung cấp dữ kiện về..."* trong quy tắc 5.
- `generation_agent._QA_SYSTEM_PROMPT` quy tắc 5: bỏ câu mẫu tiếng Việt, yêu cầu viết câu từ chối bằng ngôn ngữ của câu hỏi.
- `_EN_ANSWER_DIRECTIVE` (gắn cuối prompt khi câu hỏi là tiếng Anh): thêm câu mẫu tiếng Anh *"The provided documents do not cover ..."*.
- Xác nhận (07/11/2026, `lang-fix`…`lang-fix4`, 3 câu): `hybrid_rerank` trả lời đúng ngôn ngữ **3/3**.
  `full` lần đầu chỉ **1/3** (q065, q067 trả lời tiếng Việt sau 2 vòng sinh lại), các lần sau 3/3, 2/2, 1/1.
  Số mẫu rất nhỏ và kết quả dao động giữa các lần chạy, nên **chưa thể nói đã hết lỗi ở `full`**.
  Nghi nguyên nhân: khối phản hồi của critic khi sinh lại viết toàn tiếng Việt kéo câu trả lời theo.
  Đã sửa (`run_generation`: khối phản hồi và nhãn "Question:" theo ngôn ngữ câu hỏi), **chưa đo riêng**
  vì các lần chạy sau đó dính lỗi LLM provider (lang-fix3: 1/3 câu lỗi, lang-fix4: 2/3 câu lỗi,
  "LLM sinh câu trả lời thất bại"; thử trực tiếp 2/3 lời gọi lỗi). Cần chạy lại khi provider ổn định.

### 3.2 Mục "💡 Điểm cần lưu ý / Ứng dụng" chỉ dùng context 🟡
Mục này giờ chỉ được nêu ngoại lệ, ví dụ hay ứng dụng **có trong context**, kèm số trích dẫn.
Context không có thì bỏ hẳn mục, không viết tiêu đề rỗng.
- ⬜ Prompt teaching (mục "Ví dụ minh hoạ thực tế") có cùng rủi ro nhưng chưa sửa, vì eval không đo mode teaching.

### 3.3 Critic chặt hơn 🟡
- `grade_answer` giờ chấm **theo từng khẳng định**, cùng định nghĩa với judge `eval/judge.py`:
  tách claim, đánh dấu supported, ví dụ hay ứng dụng ngoài context tính là không được hỗ trợ.
  Điểm faithfulness bằng tỉ lệ claim được hỗ trợ.
- Ngưỡng `CLAIM_SUPPORT_THRESHOLD = 0.9`: 1 ý lạc trong ~10 ý vẫn qua, từ 2 ý trở lên phải sinh lại.
  Feedback gửi lại cho generation liệt kê nguyên văn tối đa 5 claim cần bỏ hoặc sửa,
  thay vì nhận xét chung chung như trước.
- Critic có task riêng `"critic"` trong router. Đặt `CRITIC_MODEL_PROVIDER` và `CRITIC_MODEL_NAME`
  để dùng model khác họ với Gemini; để trống thì dùng CHEAP model như trước. Chấm context (`grade`) không đổi.
- Critic cũng nhận câu hỏi (trước đây không có).
- 👤 Ngưỡng 0,9 là ước lượng, chưa đo. Cần chạy `full` để xem tỉ lệ sinh lại (trước là 1/37),
  số claim tự thêm mỗi câu (trước ~1,2) và chi phí token. Gợi ý: đặt `CRITIC_MODEL_*` thành model judge đang dùng.

---

## 4. Dataset v2 ✅ (đã duyệt 07/11/2026)

`eval/data/questions_v2.jsonl` gồm 48 câu v1 cộng **33 câu mới** (người dùng đã duyệt nội dung,
`reviewed=true`; viết tắt/không dấu là **cố ý**, để mô phỏng câu sinh viên gõ thật), tổng 81 câu
(58 câu có đáp án, 23 câu không có đáp án). File sinh bằng `python -m eval.build_v2_drafts`.
Script tự kiểm trên index Qdrant thật:
- câu có đáp án: mỗi `gold_evidence` khớp nguyên văn một chunk **của đúng file nguồn** (cùng hàm với preflight của eval);
- câu không có đáp án: các cụm từ then chốt **không có** trong corpus.

Ở bản nháp đầu, phần kiểm tra báo 5 câu. 2 câu là nhãn sai thật, 3 câu còn lại do mẫu tìm kiếm quá rộng nên báo nhầm (đã thu hẹp mẫu). Nhãn sai thứ nhất: "Việt Nam có bao nhiêu
loài thằn lằn rắn?" ban đầu bị gán "không có đáp án", nhưng tài liệu có câu "3 loài thằn lằn
rắn của Việt Nam". Câu này đã được chuyển thành câu có đáp án (v2s21). Nhãn sai thứ hai: câu hỏi "câu 41 đề HK232", trong khi bảng đáp án có tới câu 42.

| Nhóm | Số câu | Đặc điểm |
|---|---|---|
| `student-phrasing` (có đáp án) | 21 | viết tắt, không dấu, văn nói ("bn", "đc ko", "lam sao dung bfs..."), trộn Anh–Việt, 2 câu tiếng Anh viết tắt |
| `hard-unanswerable` | 12 | chủ đề có trong corpus nhưng chi tiết được hỏi thì không có: nơi công bố A*, năm của Bellman–Ford, tổng hạn mức vàng tính bằng tấn, mức phạt khi không niêm yết giá, Gantt chart, B+ tree split, SCD loại 2, câu 55 của đề HK232; có 2 câu văn phong sinh viên |

⚠️ Chạy lại `python -m eval.build_v2_drafts` sẽ ghi đè file với `reviewed=false`.
- ⬜ Chưa có câu **tiền đề sai** (false-premise). Tôi cố ý không thêm vì metric "từ chối đúng"
  hiện không chấm được kiểu trả lời "sửa tiền đề".
- Tất cả câu vẫn do Claude soạn (tag `claude-drafted`), nên chưa phải câu sinh viên thật viết.

### 4.1 Retrieval-only trên v2 (run `retrieval-v2data`, miễn phí) ✅

Mặc định mới (dense, top-10), tách theo nhóm câu có đáp án:

| Nhóm | n | `dense` MRR | `dense_rerank` hit@1 | `dense_rerank` hit@5 | `dense_rerank` MRR |
|---|---|---|---|---|---|
| v1 (câu "sạch") | 37 | 0,710 | 0,784 | 1,000 | 0,881 |
| v2 `student-phrasing` | 21 | 0,732 | 0,667 | 0,905 | 0,763 |
| Tất cả | 58 | 0,718 | 0,741 | 0,966 | 0,838 [0,759–0,909] |

- Văn phong sinh viên làm retrieval **có rerank** tụt rõ (MRR 0,881 → 0,763, hit@5 1,0 → 0,905).
  Số câu ít (n=21) nên CI rộng, nhưng đây là lần đầu dataset có câu retrieval thật sự trượt.
- 2 câu trượt hoàn toàn đều là **lệch từ vựng**, đúng loại lỗi `rewrite_query` của CRAG sinh ra để sửa:
  - v2s08 *"lam sao dung bfs de kiem tra do thi hai phia"*: gõ không dấu. Top-5 có `wiki_bfs.md` nhưng không trúng đoạn chứa bằng chứng.
  - v2s11 *"cây khung lớn nhất ... thuật toán cây khung nhỏ nhất"*: tài liệu viết "cây bao trùm". Cả 5 chunk lấy từ `wiki_kruskal.md`, trong khi bằng chứng nằm ở bài khác.
- Thú vị: không rerank, nhóm student lại có MRR ngang v1 (0,732 so với 0,710). Reranker là phần
  bị văn phong sinh viên làm hại nhiều nhất. Có thể nó nhạy với chữ không dấu hơn embedder.

### 4.2 Chuẩn hoá truy vấn (không LLM) ✅

`app/retrieval/query_normalizer.py`, bật bằng `QUERY_NORMALIZE=true` (mặc định bật). Chạy một lần trước
embed + rerank; câu gốc vẫn là thứ gửi cho LLM sinh câu trả lời.
- Mở rộng viết tắt phổ biến (`bn, đc, ko, vs, vd, ntn, mn, z, j`…), bỏ qua từ viết hoa và câu tiếng Anh.
- Câu hoàn toàn không dấu: khôi phục dấu bằng từ điển dựng từ chính các chunk đã index (tần suất từ
  và cặp từ, chọn đường tốt nhất kiểu Viterbi; từ lạ như `bfs`, `dijkstra` giữ nguyên). Từ điển làm mới sau 10 phút.
- **Cố ý không có bảng đồng nghĩa theo miền** ("cây khung" → "cây bao trùm"): đó là chỉnh theo bộ test.

Đo (`retrieval-v2-norm` so với `retrieval-v2data`, cùng cấu hình dense top-10, retrieval-only):

| Nhóm | n | MRR `dense_rerank` trước → sau | hit@5 trước → sau |
|---|---|---|---|
| v1 | 37 | 0,881 → 0,881 | 1,000 → 1,000 |
| student-phrasing | 21 | 0,763 → 0,807 | 0,905 → 0,952 |
| Tất cả | 58 | 0,838 → 0,854 | 0,966 → 0,983 |

- **Chỉ 2 câu đổi:** v2s08 (gõ không dấu) từ trượt hẳn lên hạng 1 (0 → 1,0); v2s10 hơi tệ đi
  (0,33 → 0,25) vì "bn vậy mn" thành "bao nhiêu vậy mọi người" thêm từ nhiễu vào truy vấn.
- **Bằng chứng mỏng:** lợi ích đến từ đúng 1 câu; không có CI nào tách khỏi 0. Chỉ có 1/21 câu student
  gõ hoàn toàn không dấu, nên phần khôi phục dấu gần như chưa được thử. Kết luận đúng: "không làm
  hại v1, sửa được một ca không dấu", chưa phải "cải thiện chắc chắn".
- Rủi ro đã thấy: khôi phục dấu chọn theo tần suất nên có thể sai ("đề thi" → "đề thị", "câu 55" → "cấu 55"
  khi từ điển có "thị/cấu" nhiều hơn). Nếu thấy hại câu thật, đặt `QUERY_NORMALIZE=false`.
- **Không giúp v2s11** ("cây khung" ≠ "cây bao trùm"): cần hiểu ngữ nghĩa, tức multi-query/CRAG.
  Câu này vẫn là bằng chứng để đo CRAG ở mục dưới.

### 4.3 CRAG và critic trên nhóm câu khó ✅ (run `crag-hard-v2`, 07/11/2026)

33 câu (21 `student-phrasing` + 12 `hard-unanswerable`), mặc định mới (dense top-10, chuẩn hoá truy vấn bật),
judge `gpt-5.4`, model sinh `gemini-2.5-flash`. Lệnh:
`python -m eval.run --dataset eval/data/questions_v2.jsonl --configs dense_rerank crag full --tags student-phrasing hard-unanswerable --run-id crag-hard-v2`

| Cấu hình | Faithfulness | Correctness | Từ chối nhầm | Từ chối đúng | LLM calls | Token TB | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| `dense_rerank` (1 lần sinh) | 0,967 | 0,929 | 0,048 | 12/12 | 1,0 | 2.841 | 15,5s | 22,6s |
| `crag` | 0,960 | 0,929 | 0,048 | 12/12 | 3,5 | 8.913 | 22,3s | 62,4s |
| `full` | 0,961 | 0,929 | 0,048 | 12/12 | 4,4 | 11.998 | 25,6s | 57,0s |

- **Không có khác biệt nào**: mọi Δ so với `dense_rerank` đều 0 hoặc CI chứa 0 (faithfulness −0,007).
  Kết quả giống hệt nhau ở correctness, từ chối nhầm và từ chối đúng.
- Cái giá thì rõ: **3,5–4,4 lời gọi LLM và gấp 3–4 lần token** cho cùng chất lượng.
- Câu hỏi "dataset có quá dễ cho CRAG không" giờ đã được kiểm: dù có câu retrieval trượt và câu
  không dấu, CRAG vẫn không cải thiện. Mặc định **tắt** vì vậy có cơ sở trên cả v1 lẫn nhóm câu khó.
- Ca đáng chú ý, v2s11 ("cây khung lớn nhất"): CRAG viết lại truy vấn rất tốt (tự suy ra "đảo chiều
  trọng số", kéo thêm `wiki_minimum_spanning_tree.md`, `wiki_prim.md`) nhưng vẫn không lấy được đoạn
  bằng chứng nên câu trả lời vẫn là từ chối. Đây là lần duy nhất CRAG có việc để làm, và nó chưa đủ.
- 12/12 câu không có đáp án khó đều được từ chối đúng **ngay cả khi không có critic**, nên chưa có
  chỗ nào để critic chứng minh giá trị chống ảo giác trên tập này.
- Giới hạn: n=21 câu có đáp án, CI rộng; 1 lần chạy, không lặp để đo dao động của LLM. Chưa đo
  `CRITIC_MODEL_*` khác họ, và chưa đo "chỉ bật CRAG khi điểm context thấp".


## 5. Chạy lại eval 👤 (cần credit LLM, chưa chạy)

Thêm vào eval: cấu hình reranker được ghi vào `meta.json` và đầu `report.md`, và resume một
`--run-id` với cấu hình reranker khác bị chặn. Lệnh gợi ý, chỉ chạy các cấu hình cần so sánh:

```bash
# 0. Kiểm tra nhanh lỗi ngôn ngữ ở câu từ chối (3 câu)
python -m eval.run --qids q065 q066 q067 --configs hybrid_rerank full --run-id lang-fix

# 1. Đợt chính: dataset v2 sau khi duyệt, cấu hình mặc định mới (dense, top-10)
python -m eval.run --dataset eval/data/questions_v2.jsonl --configs dense dense_rerank full --run-id full-v2

# 2. Thử critic khác họ model (tuỳ chọn)
CRITIC_MODEL_PROVIDER=shopaikey_openai CRITIC_MODEL_NAME=<model> \
  python -m eval.run --dataset eval/data/questions_v2.jsonl --configs full --run-id full-v2-critic
```

Lưu ý: `crag`/`full` đi qua `retrieval_agent` của production, nên giờ dùng dense top-10.
So với `full-v1` (hybrid top-20) là thay **hai** thứ cùng lúc: retrieval và prompt/critic.

---

## 6. Danh sách file thay đổi

**Sửa:** `app/config.py`, `.env.example`, `app/retrieval/reranker.py`, `app/retrieval/helpers.py`,
`app/indexing/qdrant_store.py`, `app/agents/retrieval_agent.py`, `app/agents/constants.py` (bỏ `HYBRID_TOP_K`),
`app/agents/generation_agent.py`, `app/agents/reflection_agent.py`, `app/graph/build_graph.py`,
`app/llm/router.py`, `app/llm/utils.py`, `eval/pipelines.py`, `eval/run.py` (thêm `--tags`), `eval/report.py`, `eval/README.md`,
`tests/test_graph.py` (fixture bật CRAG/critic).

**Mới:** `scripts/bench_reranker.py`, `eval/build_v2_drafts.py`, `eval/data/questions_v2.jsonl`,
`tests/test_eval_followup.py`, `EVAL_FOLLOWUP.md`.

**Kết quả đo** (thư mục `eval/results/` nằm trong `.gitignore`, muốn commit thì cần `git add -f`):
`retrieval-v4/`, `retrieval-v4-int8/` (số chính thức), `retrieval-v3-*` (chỉ dùng số chất lượng),
`rerank-bench-v1.json`.

**Chưa commit.** Khi bạn đồng ý, tôi sẽ tạo branch và commit.
