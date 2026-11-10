# Roadmap

Status of StudyRAG as of **November 2026**. Items move to "Done" only when they are merged and covered by tests or a reproducible run.

## ✅ Done

**Core RAG**
- [x] Ingestion: PDF (PyMuPDF → pypdf → Tesseract `vie` OCR), DOCX, PPTX, TXT, MD; NFC cleaning
- [x] Heading-aware parent/child semantic chunking (+ fixed-size baseline for ablation)
- [x] BGE-M3 dense + sparse embeddings, Qdrant hybrid search with RRF, mandatory `doc_id` filter
- [x] Query rewrite / decomposition, `bge-reranker-v2-m3`, child → parent expansion

**Agents & orchestration**
- [x] LangGraph `StateGraph`: orchestrator → retrieval → CRAG loop → generation → Self-RAG critic loop
- [x] Honest reflection trace (`*_verified`, `*_forced`), bounded by `MAX_REFLECTION_LOOPS`
- [x] QA + teaching modes, quiz & summary agent
- [x] Cheap/strong LLM router, provider-agnostic, per-task token accounting

**Memory & learning**
- [x] Token-budgeted context window, background rolling summary
- [x] Server-graded quiz, per-topic mastery, Leitner-style review schedule, weak-topic injection

**Product & ops**
- [x] React 19 UI: spaces, upload progress, SSE reasoning timeline, clickable citations, quiz, mastery
- [x] Auth: JWT, argon2id, email OTP, Google Sign-In; ownership checks on every id; rate limiting (Redis)
- [x] PostgreSQL + Alembic, Docker Compose (postgres · qdrant · redis · api · web), nginx
- [x] CI: ruff, migration round-trip, pytest (145 tests), frontend lint + build
- [x] Ablation eval harness with bootstrap CIs, cross-model judge, human audit of the judge
- [x] **Published evaluation results** — 18-document public corpus ([`eval/CORPUS.md`](eval/CORPUS.md)),
  48 reviewed questions (37 answerable, 11 unanswerable, 11 in English); all 6 configs run;
  [`eval/results/full-v1/report.md`](eval/results/full-v1/report.md) + free re-score
  [`retrieval-v2`](eval/results/retrieval-v2/report.md); judge–human agreement 85 % (17/20, judge more lenient).

**Production readiness (November 2026)**
- [x] Conversation history reaches the generator; follow-ups rewritten in the intent call
- [x] Token streaming, LLM fallback, prompt-injection mitigation, daily quotas, background parsing
- [x] Per-node latency and token metrics stored per answer, 👍/👎 feedback

## 🚧 Next (in priority order)

1. **Public demo** — production files and a read-only demo space are ready; hosting is next.
2. **Demo assets** — GIF of the full flow + screenshots.
3. **Reranker fine-tuning** — small base model, trained on real student queries, held-out documents.
4. **Adaptive CRAG** — run the loop only when the retrieval score is low.
5. **Background indexing queue** — move ingestion off the API process (e.g. arq/RQ on Redis).

## 💡 Later / ideas

- Multimodal ingestion: figures and formulas from slides (OCR → captioning).
- Smarter scheduling (SM-2 / FSRS) using response confidence, not only correctness.
