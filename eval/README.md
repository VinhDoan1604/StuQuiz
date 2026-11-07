# Đánh giá StudyRAG

Bộ eval này thay thế hoàn toàn `ragas_eval.py` cũ (baseline tính ngược, judge trả hằng số).

## Cấu hình ablation (mỗi bước thêm đúng 1 thành phần)

| Cấu hình | Thêm gì | Ghi chú |
|---|---|---|
| `dense` | — | **Baseline = Naive RAG**: dense top-5 → 1 lần sinh |
| `hybrid` | + sparse lexical, RRF | |
| `hybrid_rerank` | + cross-encoder reranker | hybrid top-N → rerank → top-5 (N = `RERANK_CANDIDATES`, mặc định 10) |
| `dense_rerank` | nhánh phụ: dense top-N → rerank | so với `hybrid_rerank` để biết sparse còn giúp gì khi đã rerank |
| `parent` | `hybrid_rerank` + mở rộng parent | |
| `crag` | + Retrieval Agent, chấm context, viết lại truy vấn | tái dùng node trong `build_graph.py` |
| `full` | + Critic chấm faithfulness, sinh lại | đồ thị đầy đủ |

## Quy trình

```bash
# 1. Chuẩn bị corpus: 3-5 tài liệu CÔNG KHAI (để người khác chạy lại được)
mkdir -p eval/corpus && cp <file>.pdf eval/corpus/
python -m eval.ingest_corpus eval/corpus            # -> eval/data/corpus_manifest.json

# 2. Tạo dataset (mục tiêu >= 50 câu: ~35 có đáp án + >= 8 không có đáp án + tự viết thêm)
python -m eval.build_dataset --n 40                 # -> questions.candidates.jsonl
#    Mở file, đọc từng câu, sửa/xoá, đổi "reviewed": true, lưu thành eval/data/questions.jsonl
#    Tự viết thêm câu diễn đạt khác tài liệu + câu "answerable": false (xem questions.example.jsonl)

# 3. Retrieval-only trước (không tốn LLM)
python -m eval.run --retrieval-only

# 4. Chạy đầy đủ với judge KHÁC model sinh
export EVAL_JUDGE_PROVIDER=openai EVAL_JUDGE_MODEL=<model>   # hoặc anthropic/google model khác
python -m eval.run

# 5. Kiểm tra độ tin cậy của judge (nên làm)
python -m eval.audit eval/results/<run_id> --n 20   # điền human_correctness trong audit.csv
python -m eval.audit eval/results/<run_id> --score
```

Chạy lại tiết kiệm credit: chỉ chạy cấu hình cần so sánh, vd `python -m eval.run --configs dense hybrid_rerank full`.

Thử reranker nhanh hơn (miễn phí, chỉ retrieval):

```bash
python -m scripts.bench_reranker                    # latency từng biến thể, không cần Qdrant
RERANK_CANDIDATES=10 RERANKER_INT8=true python -m eval.run --retrieval-only \
    --configs dense hybrid_rerank dense_rerank --run-id <id>
```

Cấu hình reranker được ghi vào `meta.json` và đầu `report.md`; resume với cấu hình khác bị chặn.

### Dataset v2

`eval/data/questions_v2.jsonl` = 48 câu v1 + 33 câu mới (đã duyệt 07/11/2026, sinh bởi
`python -m eval.build_v2_drafts`, tự kiểm bằng chứng/nhãn trên index): 21 câu viết theo văn
phong sinh viên (`student-phrasing`) và 12 câu không có đáp án khó (`hard-unanswerable`).
Chạy lại `build_v2_drafts` sẽ ghi đè file với `reviewed=false`. Chạy eval:
`python -m eval.run --dataset eval/data/questions_v2.jsonl ...`.

Kết quả: `eval/results/<run_id>/{report.md, summary.json, raw.jsonl, meta.json}`.
Chạy bị ngắt/lỗi 429 → chạy lại cùng `--run-id`, chỉ các câu lỗi được chạy lại.

## Cách đọc kết quả (và cách nói trong phỏng vấn)
- Mọi số kèm CI 95% bootstrap. **CI chứa 0 ở bảng "Chênh lệch" = chưa chứng minh được cải thiện.**
  Với n≈50 thì chênh lệch < ~8-10 điểm % thường không có ý nghĩa — hãy nói thẳng điều đó.
- Nếu `full` không hơn `crag` đáng kể nhưng tốn gấp đôi latency/LLM calls, đó là một
  *phát hiện hợp lệ* (trade-off), không phải thất bại. Đừng giấu.
- Chỉ số chống ảo giác thật sự nằm ở **"Từ chối đúng"** (câu không có đáp án) và **Faithfulness**.

## Hạn chế đã biết (ghi vào README chính)
- Câu hỏi sinh tự động trùng từ vựng với tài liệu → retrieval dễ hơn thực tế.
- Judge là LLM; nếu cùng model với bên sinh, điểm bị thổi phồng (báo cáo tự cảnh báo).
- Token đếm qua callback của `router.py`; nếu provider không trả usage thì cột token là `n/a`
  (số LLM calls vẫn đúng).
- Orchestrator bị bỏ qua (intent cố định `qa`) để chỉ đo chất lượng RAG.

## Công bố kết quả
- README chính không chứa số liệu cho tới khi có một lần chạy thật: commit riêng 1 `report.md`
  đại diện vào thư mục tài liệu rồi dẫn link từ README.
- `eval/results/` và `eval/corpus/` đã nằm trong `.gitignore`.
