# 🎓 StudyRAG — Multi-Agent RAG Study Assistant

> Ask questions about **your own lecture slides and textbooks** and get answers that are
> **grounded, cited and self-checked** — then turn the same material into quizzes that track
> what you actually remember.

[![CI](https://github.com/VinhDoan1604/StuQuiz/actions/workflows/ci.yml/badge.svg)](https://github.com/VinhDoan1604/StuQuiz/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB)
![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C)
![Qdrant](https://img.shields.io/badge/Qdrant-hybrid_search-DC244C)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688)
![React](https://img.shields.io/badge/React_19-TypeScript-61DAFB)

🇻🇳 Bản tiếng Việt: [README.vi.md](README.vi.md)

<!-- TODO(Vinh): replace with a real GIF (upload → ask → reflection timeline → citation click → quiz).
     Record with ScreenToGif / OBS, keep it < 10 MB, save as assets/demo.gif, then uncomment:
<p align="center"><img src="assets/demo.gif" alt="StudyRAG demo" width="820"/></p>
-->


---

## Why this project

Students learn from long, messy, often Vietnamese PDFs and slide decks. A plain "chat with your PDF"
bot has three problems on this kind of material:

1. **It answers confidently when the context is wrong or missing** (hallucination).
2. **You can't verify it** — no pointer back to the slide or page.
3. **It doesn't help you remember** — every session starts from zero.

StudyRAG addresses each one explicitly: a **Corrective-RAG + Self-RAG reflection loop** grades the
retrieved context *before* generation and the answer's faithfulness *after* it; every answer carries
**clickable citations** to the source chunk; and a **server-graded quiz + spaced-review tracker**
keeps a per-topic mastery score.

## Key features

| | |
|---|---|
| 🔁 **Reflection loop (CRAG + Self-RAG)** | Grade context → rewrite query & re-retrieve if insufficient → generate → critic checks faithfulness → regenerate with feedback. Bounded by `MAX_REFLECTION_LOOPS`; forced acceptances are flagged in the trace, not hidden. |
| 🔎 **Hybrid retrieval** | BGE-M3 dense (1024-d) + learned sparse vectors in one Qdrant collection, fused with RRF, then `bge-reranker-v2-m3` cross-encoder, then child→parent chunk expansion. |
| 🧭 **Intent routing** | Orchestrator classifies `qa / teaching / quiz / summary`; teaching mode explains step by step instead of just answering. |
| 📡 **Live reasoning trace** | `/chat/stream` (SSE) streams each graph step; the React UI renders it as a timeline with context/answer scores and rewritten queries. |
| 🧠 **Memory** | Token-budgeted context window + rolling summary (summarised in the background after the response) + weak-topic injection from mastery records. |
| 📝 **Quiz & mastery** | Answers never leave the server before grading; mastery score (asymmetric +10 / −15) and a Leitner-style review schedule (1 → 3 → 7 → 14 days). |
| 💸 **Cost-aware LLM routing** | Cheap model for rewrite/decompose/grade/qa, strong model for teaching/quiz/summary; provider-agnostic (Gemini, OpenAI, Anthropic). Token usage tracked per task via callbacks. |
| 🇻🇳 **Vietnamese-ready ingestion** | PDF (PyMuPDF → pypdf fallback → Tesseract `vie` OCR for scans), DOCX, PPTX, TXT, MD; Unicode NFC normalisation; heading-aware semantic chunking. |

## Architecture

```mermaid
flowchart TD
    User([Student]) -->|SSE /chat/stream| Orchestrator[Orchestrator<br/>intent: qa / teaching / quiz / summary]

    subgraph "LangGraph StateGraph"
        Orchestrator --> Retrieval[Retrieval Agent<br/>BGE-M3 dense+sparse · RRF · rerank · parent expansion]
        Retrieval -->|qa / teaching| GradeContext{Grade context}
        Retrieval -->|quiz / summary| QuizSummary[Quiz & Summary Agent]

        GradeContext -->|insufficient & budget left| RewriteQuery[Rewrite query]
        RewriteQuery --> Retrieval
        GradeContext -->|sufficient / budget spent| Generation[Generation Agent<br/>answer + citations]

        Generation --> GradeAnswer{Critic: faithfulness}
        GradeAnswer -->|unfaithful & budget left| RetryGen[Regenerate with feedback]
        RetryGen --> Generation
        GradeAnswer -->|faithful / budget spent| Accept[Accept + reflection trace]
    end

    Accept --> Memory[(Chat history · rolling summary<br/>Mastery · spaced review)]
    QuizSummary --> Memory
```

**Ingestion path:** upload → parse (OCR fallback) → clean/NFC → parent/child semantic chunks
(child 200–400 tokens, 50 overlap; parent ≤ 1 500) → BGE-M3 dense + sparse → Qdrant, filtered by `doc_id`.

| Layer | Tech |
| :--- | :--- |
| API | FastAPI, SSE, JWT (HS256), argon2id, Google Sign-In, email OTP, Redis-backed rate limiting |
| Orchestration | LangGraph `StateGraph` ([`app/graph/build_graph.py`](app/graph/build_graph.py)), LangChain Core |
| Models | Gemini / OpenAI / Anthropic via [`app/llm/router.py`](app/llm/router.py); BGE-M3; bge-reranker-v2-m3 |
| Storage | PostgreSQL 16 + Alembic migrations; Qdrant |
| Frontend | React 19 + Vite + TypeScript, served by nginx |
| Ops | Docker Compose (postgres · qdrant · redis · api · web), GitHub Actions CI, `/health` + `/ready`, JSON logs with `X-Request-ID` |

## Design decisions & trade-offs

- **Why a reflection loop instead of naive RAG?** Course material is dense and terminology-heavy;
  the first retrieval is often near-miss. Grading context *before* generating is cheaper than
  generating and then failing the critic. The loop is capped, and the trace records whether a
  grade was actually computed (`*_verified`) or the answer was accepted because the budget ran out
  (`*_forced`) — so the UI never claims a check that didn't happen.
- **Why BGE-M3 sparse instead of BM25/Elasticsearch?** One model produces both dense and lexical
  vectors, so hybrid search lives in a single Qdrant collection with no extra search engine to run.
  It is also multilingual, which matters for Vietnamese + English mixed slides.
- **Why parent/child chunks?** Small child chunks retrieve precisely; the parent section is what
  the LLM reads, so answers don't lose surrounding definitions.
- **Why a cheap/strong router?** Grading and rewriting are short classification-style calls that run
  several times per question; routing them to a cheaper model keeps the loop affordable.
- **Known cost:** the full loop makes more LLM calls and adds latency vs. single-shot RAG. The
  evaluation below is designed to measure whether that cost buys accuracy.

## Evaluation

An ablation harness ([`eval/`](eval/README.md)) adds one component at a time so each gain is attributable:

| Config | Adds |
|---|---|
| `dense` | Naive RAG baseline: dense top-5 → single generation |
| `hybrid` | + sparse vectors, RRF fusion |
| `hybrid_rerank` | + cross-encoder rerank (top-20 → top-5) |
| `crag` | + context grading & query rewriting |
| `full` | + faithfulness critic & regeneration |

Metrics: retrieval **hit@k / recall@k / MRR**; answer **correctness & faithfulness** (LLM-as-judge
using a *different* model from the generator); **correct refusal rate** on unanswerable questions;
latency and LLM calls per question. All numbers come with **95 % bootstrap CIs** and paired-bootstrap
differences, and judge reliability is audited against human labels (`eval/audit.py`).

<!-- TODO(Vinh): fill from eval/results/<run_id>/report.md after a real run. Do not invent numbers. -->
> **Results:** _pending first full run on a public corpus — table will be published from
> `report.md`, including configurations that did **not** improve._

## Security & data isolation

- Every route except login requires a JWT; identity always comes from the token, never the body.
- Every client-supplied id (space, document, session) is ownership-checked in
  [`app/core/ownership.py`](app/core/ownership.py); other users' resources return **404**.
- Vector search **requires** a verified `doc_id` list — no collection-wide queries.
- Quiz answers are never sent to the client before server-side grading.
- OTPs are stored as HMAC, with attempt limits, expiry and resend cooldown.

## Quick start (Docker)

```bash
git clone https://github.com/VinhDoan1604/StuQuiz.git && cd StuQuiz
cp .env.example .env      # set POSTGRES_PASSWORD and one LLM key (e.g. GOOGLE_API_KEY)
docker compose up -d --build
```

Open **http://localhost:8080**. Migrations run on start. On first ingestion BGE-M3 and the reranker
(~2–3 GB) are downloaded into the `model_cache` volume (`PRELOAD_MODELS=true` to download at boot).

<details>
<summary>Local development</summary>

Requires Python 3.11, Node 20+, Docker (for Postgres/Qdrant). Optional: `tesseract-ocr` + `tesseract-ocr-vie`.

```bash
docker compose up -d postgres qdrant
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python scripts/init_db.py                              # alembic upgrade head
python scripts/seed_demo.py                            # optional demo data
uvicorn app.main:app --reload --port 8000              # Swagger at /docs (APP_ENV=dev only)
cd frontend && npm install && npm run dev              # http://localhost:5173
```

- With `APP_ENV=dev`, if Postgres is unreachable the backend falls back to SQLite; other envs fail fast.
- In dev without SMTP, the OTP is shown on the verification screen.
- Optional tracing: set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` to see every graph run in LangSmith.
- Google Sign-In setup: see [README.vi.md](README.vi.md#bật-đăng-nhập-google-tuỳ-chọn).

</details>

## Tests & quality

```bash
pytest -q                                  # 96 tests: API, auth isolation, graph routing, memory, quiz, rate limit — no LLM/Qdrant needed
ruff check app tests migrations scripts
cd frontend && npm run lint && npm run build
```

CI runs lint, an Alembic upgrade/downgrade round-trip, pytest, and the frontend build on every push.

## Project layout

```
app/
├── api/         routes: auth, spaces, documents, chat (SSE), quiz, mastery
├── agents/      orchestrator, retrieval, generation, reflection, quiz/summary
├── graph/       LangGraph StateGraph + run/stream
├── retrieval/   query rewrite/decompose, rerank, parent expansion
├── ingestion/   parsers, OCR, cleaning        chunking/  semantic + fixed baseline
├── embedding/   BGE-M3                        indexing/  Qdrant store
├── memory/      chat history, context window, mastery tracker
├── llm/         provider registry + cheap/strong router
└── core/        security, ownership, rate limit, logging
eval/            ablation harness, judge, bootstrap metrics, judge audit
frontend/        React + Vite (Dockerfile + nginx)
migrations/      Alembic          tests/  pytest
```

## Limitations & next steps

- Auto-generated eval questions share vocabulary with the source, which makes retrieval look easier
  than real student questions; the dataset is being extended with hand-written paraphrases.
- Indexing runs in-process; a task queue would be needed for many concurrent uploads.
- One worker per container because models live in process memory; scale out horizontally.

See [ROADMAP.md](ROADMAP.md) for what's done and what's next.

## Author

**Đoàn Công Vinh** — Computer Science, Ho Chi Minh City University of Technology (HCMUT) ·
[GitHub](https://github.com/VinhDoan1604) <!-- TODO(Vinh): add LinkedIn / email -->

<!-- TODO(Vinh): choose a license (e.g. MIT) and add LICENSE; until then all rights reserved. -->
