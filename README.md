# 🎓 StudyRAG — Multi-Agent RAG Study Assistant

> Ask questions about **your own lecture slides and textbooks** and get answers that are
> **grounded, cited and self-checked** — then turn the same material into quizzes that track
> what you actually remember.

[![CI](https://github.com/VinhDoan1604/StuQuiz/actions/workflows/ci.yml/badge.svg)](https://github.com/VinhDoan1604/StuQuiz/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB)
![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C)
![Qdrant](https://img.shields.io/badge/Qdrant-vector_search-DC244C)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688)
![React](https://img.shields.io/badge/React_19-TypeScript-61DAFB)

🇻🇳 Bản tiếng Việt: [README.vi.md](README.vi.md)

---

## The short version

I built the full agentic pipeline (corrective RAG, self-critic, hybrid search), then measured every
component with an ablation harness. On 81 questions, **CRAG and the critic cost 2–3× the tokens with
no measurable quality gain**, and sparse retrieval added nothing once reranking was on, so production
runs the simpler pipeline: dense search, cross-encoder rerank, one generation. The reranking stack
clearly improved ranking (MRR 0.74 → 0.85), but answer correctness did not change, because the
evidence was usually already in the top 5. The expensive paths are still in the code behind flags,
and the numbers are below.

![StudyRAG demo: upload a document, ask a question, open a citation](assets/demo.gif)

---

## Why this project

Students learn from long, messy, often Vietnamese PDFs and slide decks. A plain "chat with your PDF"
bot has three problems on this kind of material:

1. **It answers confidently when the context is wrong or missing** (hallucination).
2. **You can't verify it** — no pointer back to the slide or page.
3. **It doesn't help you remember** — every session starts from zero.

StudyRAG addresses each one explicitly: answers are constrained to retrieved context and refuse when
it is missing (22 of 22 unanswerable questions refused); every answer carries **clickable
citations** to the source chunk; and a **server-graded quiz + spaced-review tracker** keeps a
per-topic mastery score.

## Key features

| | |
|---|---|
| 🔎 **Measured retrieval** | BGE-M3 dense search → `bge-reranker-v2-m3` cross-encoder → child→parent expansion. Sparse/RRF stays available with `RETRIEVAL_MODE=hybrid`. Rule-based normalization handles Vietnamese abbreviations and missing diacritics. |
| 🔁 **Optional reflection loop** | CRAG + Self-RAG behind `ENABLE_CRAG` / `ENABLE_ANSWER_CRITIC`, off by default because evaluation showed no gain. Bounded loops; forced acceptances are flagged in the trace. |
| 🧭 **Intent routing** | Orchestrator classifies `qa / teaching / quiz / summary`; teaching mode explains step by step instead of just answering. |
| 📡 **Streaming** | `/chat/stream` (SSE) streams each graph step and the answer token by token. |
| 🧠 **Conversation context** | Token-budgeted history (rolling summary + recent turns, old citations stripped) reaches the generator. Follow-ups are rewritten into standalone questions inside the intent call, so it adds no LLM call. |
| 🛡️ **Guardrails** | Retrieved text is fenced as data against prompt injection, with per-user and global daily quotas, upload/OCR limits and an LLM fallback provider. |
| 📈 **LLMOps** | Latency per graph node, tokens and fallbacks are stored with every answer; 👍/👎 feedback; `scripts/usage_report.py` summarises them; optional LangSmith tracing. |
| 📝 **Quiz & mastery** | Answers never leave the server before grading; mastery score (asymmetric +10 / −15) and a Leitner-style review schedule (1 → 3 → 7 → 14 days). |
| 💸 **Cost-aware LLM routing** | Cheap model for rewrite/decompose/grade/qa, strong model for teaching/quiz/summary; provider-agnostic (Gemini, OpenAI, Anthropic). Token usage tracked per task via callbacks. |
| 🇻🇳 **Vietnamese-ready ingestion** | PDF (PyMuPDF → pypdf fallback → Tesseract `vie` OCR for scans), DOCX, PPTX, TXT, MD; Unicode NFC normalisation; heading-aware semantic chunking. |

## Architecture

```mermaid
flowchart TD
    User([Student]) -->|SSE /chat/stream| Orchestrator[Orchestrator<br/>intent: qa / teaching / quiz / summary]

    subgraph "LangGraph StateGraph"
        Orchestrator --> Retrieval[Retrieval Agent<br/>BGE-M3 dense · rerank · parent expansion]
        Retrieval -->|qa / teaching| Generation[Generation Agent<br/>answer + citations]
        Retrieval -->|quiz / summary| QuizSummary[Quiz & Summary Agent]
        Generation --> Accept[Accept + trace]

        Retrieval -. ENABLE_CRAG .-> GradeContext{Grade context}
        GradeContext -. insufficient .-> RewriteQuery[Rewrite query]
        RewriteQuery -.-> Retrieval
        GradeContext -. sufficient .-> Generation
        Generation -. ENABLE_ANSWER_CRITIC .-> GradeAnswer{Critic: faithfulness}
        GradeAnswer -. unfaithful .-> Generation
        GradeAnswer -. faithful .-> Accept
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
| Models | Gemini / OpenAI / Anthropic via [`app/llm/router.py`](app/llm/router.py) with fallback; self-hosted BGE-M3 and bge-reranker-v2-m3 |
| Storage | PostgreSQL 16 + Alembic migrations; Qdrant |
| Frontend | React 19 + Vite + TypeScript, served by nginx |
| Ops | Docker Compose (postgres · qdrant · redis · api · web, Caddy for HTTPS in production), GitHub Actions CI, `/health` + `/ready`, JSON logs with `X-Request-ID` |

## Design decisions & trade-offs

- **Why is the reflection loop off by default?** I expected it to be the headline feature. On 81
  questions it produced no measurable quality gain at 2–3× the tokens, so it became opt-in. When
  enabled, the trace records whether a grade was computed (`*_verified`) or the answer was accepted
  because the budget ran out (`*_forced`), so the UI never claims a check that didn't happen.
- **Why self-host embeddings and the reranker while calling an LLM API?** Generation is where a hosted
  model is worth paying for; embeddings and reranking run on every document and question, and BGE
  handles mixed Vietnamese/English well. The cost shows up in deployment: see below.
- **Why parent/child chunks?** Small child chunks retrieve precisely; the parent section is what
  the LLM reads, so answers don't lose surrounding definitions.
- **Why a cheap/strong router?** Grading and rewriting are short classification-style calls that run
  several times per question; routing them to a cheaper model keeps the loop affordable.
- **Why rewrite follow-ups inside the intent call?** It needs conversation history anyway, so a
  second field costs no extra round trip. The previous pronoun regex caught 2 of 15 follow-ups;
  this catches 15 of 15.

## Evaluation

An ablation harness ([`eval/`](eval/README.md)) adds one component at a time so each gain is attributable:

| Config | Adds |
|---|---|
| `dense` | Naive RAG baseline: dense top-5 → single generation |
| `hybrid` | + sparse vectors, RRF fusion |
| `hybrid_rerank` | + cross-encoder rerank (top-20 → top-5) |
| `parent` | + child → parent expansion (context up to 1,500 tokens) |
| `crag` | + context grading & query rewriting |
| `full` | + faithfulness critic & regeneration |
| `production` | What ships: dense top-10 → rerank → parent expansion, query decomposition, one generation |
| `teaching` | `production` with the teaching prompt |

Metrics: retrieval **hit@k / recall@k / MRR**; answer **correctness & faithfulness** (LLM-as-judge
using a *different* model from the generator); **correct refusal rate** on unanswerable questions;
latency and LLM calls per question. All numbers come with **95 % bootstrap CIs** and paired-bootstrap
differences, and judge reliability is audited against human labels (`eval/audit.py`: 85 % agreement,
17/20 — see Limitations).

### Final results (`final-v1`, 81 questions)

These measure the code that ships. Same 18-document corpus, 81 reviewed questions (58 answerable,
23 unanswerable, including 12 near-miss questions and 23 written in informal student phrasing).
Generator `gemini-2.5-flash`, judge `gpt-5.4`, brackets are 95 % bootstrap CIs. Reports:
[`report`](eval/results/final-v1/report.md), [`production vs CRAG/full`](eval/results/final-v1/compare.md),
[`dense vs production`](eval/results/final-v1/compare_dense.md).

| Config | Faithfulness | Correctness | False refusal ↓ | Correct refusal | p50 latency | LLM calls | Tokens |
|---|---|---|---|---|---|---|---|
| `dense` | 0.949 [0.924–0.972] | 0.957 [0.905–0.991] | 0.034 | 22/23 | 4.6 s | 1.0 | 2,516 |
| **`production`** | **0.968** [0.945–0.988] | **0.957** [0.905–0.991] | 0.034 | **22/22**¹ | 10.3 s | 1.0 | 3,019 |
| `crag` | 0.952 [0.925–0.973] | 0.940 [0.888–0.983] | 0.034 | 22/23 | 14.1 s | 2.7 | 6,087 |
| `full` | 0.963 [0.941–0.982] | 0.957 [0.905–0.991] | 0.017 | 22/23 | 17.8 s | 3.7 | 8,923 |

¹ One of the 23 requests got no response from the LLM provider (timeout, 0 tokens), so it has no score.

| Retrieval (n = 58) | hit@1 | hit@5 | MRR |
|---|---|---|---|
| `dense` | 0.586 [0.466–0.724] | 0.966 [0.914–1.000] | 0.735 [0.651–0.822] |
| `production` | 0.741 [0.621–0.845] | 0.983 [0.948–1.000] | 0.846 [0.776–0.909] |

Paired differences:

- **CRAG and the critic did not pay for themselves.** Against `production`, no quality metric improved
  (every CI includes 0), and retrieval was identical. `crag` added 3,069 tokens [2,662–3,531] per
  question and `full` added 5,851 [5,432–6,310]. On near-miss questions the loop averaged 4–5 LLM calls
  per question to reach a refusal that `production` reached in one.
- **The retrieval stack improved ranking but not answers.** `production` vs. `dense`: MRR +0.111
  [+0.035, +0.189], correctness +0.000 [−0.026, +0.026], at +5.8 s per question. With hit@5 already
  at 0.966, the evidence usually reached the generator either way. Ranking still decides which passage
  is cited first, and a larger corpus would likely depend on it more, but this set does not show that.
- **The ranking gain is concentrated in clean questions.** MRR +0.168 [+0.081, +0.262] on the original
  48, but +0.010 [−0.120, +0.143] on informal student phrasing.
- These `dense` and `production` runs differ by the whole retrieval stack (rerank, normalization,
  decomposition). The reranker's isolated effect is the `dense` vs. `hybrid_rerank` row below.
- `llm_calls` excludes the orchestrator, which every real chat request also runs. Measured through the
  full Docker stack, a question takes 16–22 s end to end on CPU, most of it retrieval.

### Earlier ablation (`full-v1`, 48 questions)

**Setup:** 18 public documents ([`eval/CORPUS.md`](eval/CORPUS.md): Vietnamese Wikipedia, OpenStax, a
Vietnamese decree, VOER, course notes/slides), **48 reviewed questions** (37 answerable, 11 unanswerable;
11 in English against Vietnamese sources). Generator `gemini-2.5-flash`, judge `gpt-5.4` (different
vendor). BGE-M3, top-k = 5. Brackets are 95 % bootstrap CIs.
Full reports: [`full-v1`](eval/results/full-v1/report.md) (all configs, with LLM) and
[`retrieval-v2`](eval/results/retrieval-v2/report.md) (retrieval re-scored for free after fixing the
label of q064 — see Limitations).

**Retrieval** (answerable questions, n = 37, scored on child chunks — `retrieval-v2`)

| Config | hit@1 | hit@5 | recall@5 | MRR |
|---|---|---|---|---|
| `dense` | 0.54 [0.38–0.70] | 0.97 [0.92–1.00] | 0.92 [0.84–0.97] | 0.71 [0.60–0.82] |
| `hybrid` | 0.57 [0.41–0.73] | 0.89 [0.78–0.97] | 0.86 [0.74–0.96] | 0.69 [0.57–0.81] |
| `hybrid_rerank` | **0.78** [0.65–0.92] | **1.00** [1.00–1.00] | **0.95** [0.89–0.99] | **0.88** [0.81–0.95] |

`parent` and `crag` rank chunks exactly like `hybrid_rerank` (they change what happens after
retrieval); `full` was within 0.03 on every metric in `full-v1`.

**Answers** (`full-v1`; faithfulness & correctness on n = 37 answerable, refusal on n = 11 unanswerable)

| Config | Faithfulness | Correctness | False refusal ↓ | Correct refusal | p50 latency | LLM calls |
|---|---|---|---|---|---|---|
| `dense` | **0.92** [0.88–0.95] | 0.96 [0.91–1.00] | 0.03 | 11/11 | 5.2 s | 1.0 |
| `hybrid` | 0.91 [0.87–0.95] | 0.95 [0.87–1.00] | 0.05 | 11/11 | 5.4 s | 1.0 |
| `hybrid_rerank` | 0.89 [0.86–0.93] | 0.95 [0.89–0.99] | 0.05 | 11/11 | 42.8 s | 1.0 |
| `parent` | 0.91 [0.87–0.94] | 0.97 [0.93–1.00] | 0.03 | 11/11 | 41.7 s | 1.0 |
| `crag` | 0.89 [0.85–0.93] | 0.97 [0.93–1.00] | 0.00 | 11/11 | 49.0 s | 2.9 |
| `full` | 0.87 [0.83–0.92] | **0.99** [0.96–1.00] | 0.00 | 11/11 | 50.3 s | 4.1 |

**What the numbers say**

- **The reranker is the only component with a statistically clear gain:** MRR +0.17 [+0.09, +0.26]
  and hit@1 0.54 → 0.78 over `dense` (paired bootstrap). It is also the main latency cost (CPU,
  ~35 s/question). Example (q025, "what do the white/grey/black vertex colours mean in Dijkstra?"):
  with `dense` the evidence was the 5th, LaTeX-cluttered chunk and the model opened with "the
  documents don't say" — then listed all three meanings from that same chunk; with `hybrid` the
  evidence was not retrieved at all and the model (honestly) refused. Once the reranker moved the
  evidence to position 1, every config answered correctly.
  That gain is in ranking. Answer correctness was 0.96 for `dense` and 0.95 for `hybrid_rerank`.
- **Hybrid (sparse + RRF) did not help** on this corpus: hit@5 −0.08 [−0.19, 0.00], MRR −0.02
  [−0.08, +0.05] vs. `dense`. Production now uses dense candidates for the reranker.
- **CRAG / the full loop** raise correctness (0.96 → 0.99) and remove false refusals (3 % → 0 %), but
  neither difference is significant at n = 37 — while costing ~4× the LLM calls and ~10× the latency of
  `dense`.
- **Faithfulness drops slightly** as components are added (0.92 `dense` → 0.87 `full`, Δ −0.04
  [−0.10, +0.01]) — not significant, but the opposite of what the critic loop is meant to do. Not yet
  diagnosed; longer parent contexts and multi-source synthesis are the first suspects.
- **Correct refusal is 11/11 for every config** — the unanswerable set is too easy to discriminate
  between configs (ceiling effect).

### After `full-v1`

Those results drove the current defaults. The details are in [`eval/FOLLOWUP.md`](eval/FOLLOWUP.md).

| Finding | Evidence |
|---|---|
| CRAG/critic give no gain on hard questions | 33 student-style and near-miss questions: correctness 0.929 for all three configs, at 1.0 / 3.5 / 4.4 LLM calls |
| Rerank latency cut without losing quality | p50 37–43 s → 13.0 s (dense, top-10, thread fix) → 8.4 s (int8, max length 384), MRR 0.846–0.854 on 58 answerable questions |
| Informal phrasing is the hard case | MRR 0.881 on clean questions vs. 0.763 on abbreviated, undiacritized ones |
| Follow-up handling | 15/15 follow-ups rewritten to the right topic; the old regex caught 2/15 |
| Intent classifier | 34/36 correct, 95 % CI 0.82–0.98 |
| Prompt injection | 13/20 → 0/20 attacks succeeded (10 hand-written patterns × 2 runs, one model) |

## Preparing for deployment

Not hosted yet. These findings come from running the production image locally, including under a
VPS-like cap of 2 CPUs and 3 GB:

- **The LLM API is the easy part; the self-hosted models are not.** The API image is 2.9 GB, the
  container peaked at 2.5 GB of RAM, and the first request after boot took about a minute because
  models load lazily (`PRELOAD_MODELS=true` moves that to startup).
- **CPU, not memory, is the limit.** Retrieval took 8–13 s on the dev machine and 38–57 s capped at 2
  CPUs. The production config uses int8 quantization and a shorter max length to reduce this.
- **Proxies break defaults.** Behind Caddy every user would share one IP and one rate limit until
  nginx was told to trust the Docker network's `X-Forwarded-For`, and gzip has to be off for the SSE route.

Production files: [`docker-compose.prod.yml`](docker-compose.prod.yml), [`deploy/`](deploy/), [`.env.prod.example`](.env.prod.example).

## Security & data isolation

- Every route except login requires a JWT; identity always comes from the token, never the body.
- Every client-supplied id (space, document, session) is ownership-checked in
  [`app/core/ownership.py`](app/core/ownership.py); other users' resources return **404**.
- Vector search **requires** a verified `doc_id` list — no collection-wide queries.
- Quiz answers are never sent to the client before server-side grading.
- OTPs are stored as HMAC, with attempt limits, expiry and resend cooldown.
- Demo accounts read a shared public document set; write and delete still require ownership.

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
pytest -q                                  # 150 tests: API, auth isolation, graph routing, memory, quiz, quotas, streaming — no LLM/Qdrant needed
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

- **Teaching mode is less grounded than Q&A.** It scored faithfulness 0.736 [0.670–0.802] on 40
  questions, against 0.968 for Q&A, mainly because the model invented examples and cited them. The
  prompt now restricts examples to the retrieved context, but that fix has not been re-measured yet.
- **Small eval set → wide CIs.** Only 58 answerable questions; quality differences between configs fall
  inside the 95 % CI, so only the ranking gain and the cost differences are claimed as real.
- **The judge agrees with a human on 85 % of answers (17/20)** (`eval/audit.py`, random sample from
  `full-v1`, exact correct/partial/incorrect label). All 3 disagreements were the judge being **more
  lenient** (correct vs. partial ×2 on the same question, partial vs. incorrect ×1); it was never
  stricter. Mean correctness on the sample: judge 0.95 vs. human 0.875, so absolute correctness above
  is likely somewhat optimistic; comparisons between configs are less affected because the same judge
  grades all of them. One grader, 20 answers (16 distinct questions, 18/20 judged correct) — a small
  check, not a calibration. Faithfulness was not audited.
- **One label was fixed after the run (q064).** Its gold evidence sat in a pseudo-code chunk that no
  config retrieves, while a neighbouring chunk states the answer directly. An alternative evidence was
  added and retrieval re-scored (`retrieval-v2`, no LLM calls); answer metrics are from `full-v1`.
- **False refusals are slightly under-counted.** The judge flags `abstained` only when an answer
  *mostly* says the information is missing, so a self-contradictory answer that claims "not in the
  documents" and then answers anyway (q025 with `dense`) is not counted as a refusal. A cheap fix —
  flag answers that both claim missing information and cite sources, then regenerate — is planned
  but not yet measured.
- **Correct-refusal rate is near the ceiling** (22 of 23 or better everywhere), even after adding 12
  near-miss questions, so it barely separates configs.
- **`hybrid` is not fully deterministic:** Qdrant's RRF produces score ties at the top-5 cut-off and
  breaks them arbitrarily, so `hybrid` numbers move slightly between runs (`hybrid_rerank` is stable
  because it reranks the top 20).
- Eval questions were drafted by an AI and reviewed by the author, and tend to share vocabulary with
  the source, which makes retrieval look easier than it would be with real student queries.
- Indexing runs in a single in-process slot; a task queue would be needed for many concurrent uploads.
- One worker per container because models live in process memory; scale out horizontally.

See [ROADMAP.md](ROADMAP.md) for what's done and what's next.

## Author

**Đoàn Công Vinh** — Computer Science, Ho Chi Minh City University of Technology (HCMUT) ·
[GitHub](https://github.com/VinhDoan1604)

## License

[AGPL-3.0](LICENSE), matching the PyMuPDF dependency.
