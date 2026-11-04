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
- [x] CI: ruff, migration round-trip, pytest (96 tests), frontend lint + build
- [x] Ablation eval harness with bootstrap CIs, cross-model judge, human audit of the judge

## 🚧 Next (in priority order)

1. **Publish evaluation results** — public corpus (3–5 open course PDFs), ≥ 50 reviewed questions
   incl. ≥ 8 unanswerable and hand-written paraphrases; run all 5 configs; commit one `report.md`.
2. **Demo assets** — GIF of the full flow + screenshots of the reasoning timeline and mastery view.
3. **Public demo** — hosted instance with a read-only demo space (rate-limited, cheap model only).
4. **Observability** — LangSmith (or Langfuse) traces per graph run; latency/cost per intent.
5. **Background indexing queue** — move ingestion off the API process (e.g. arq/RQ on Redis).
6. **Answer streaming** — stream generation tokens, not only graph steps.

## 💡 Later / ideas

- Multimodal ingestion: figures and formulas from slides (OCR → captioning).
- Smarter scheduling (SM-2 / FSRS) using response confidence, not only correctness.
- Retrieval fine-tuning on student-style queries mined from chat logs.
