# Evaluation follow-up: what changed after `full-v1`, and why

`full-v1` showed that the reranker was the only component with a statistically clear gain. This
report records the production decisions that followed and the evidence behind each. Runs live in
`eval/results/`; only selected reports are committed.

Generator `gemini-2.5-flash`, judge `gpt-5.4` (different vendor). Brackets are 95 % bootstrap CIs.

## 1. Production defaults

| Setting | Value | Evidence |
|---|---|---|
| `RERANK_ENABLED` | `true` | MRR +0.17 [+0.08, +0.26] over dense without rerank |
| `RETRIEVAL_MODE` | `dense` | Equal to hybrid once reranked (MRR 0.881 both at top-20) and better at top-10 |
| `RERANK_CANDIDATES` | `10` | Same MRR as 20 for dense + rerank, ~1.7× faster |
| `ENABLE_CRAG`, `ENABLE_ANSWER_CRITIC` | `false` | 3–4× tokens with no measurable gain (section 4.3) |

The sparse vectors are still indexed, so `RETRIEVAL_MODE=hybrid` works without re-indexing.
The correct reading is "dense is not worse than hybrid", not "dense is better": every difference
between reranked variants falls inside the CI. CRAG and the critic are off because no gain has been
demonstrated on this dataset, which is not proof that they are useless.

## 2. Rerank latency

| Change | Effect |
|---|---|
| Fixed torch thread count inside `asyncio.to_thread` (worker threads used 4 of 8) | 37.1 s → 20.3 s for 20 candidates |
| Own scoring loop, removing FlagEmbedding's extra probe forward pass | ~20–30 % |
| Candidates 20 → 10 | ~1.7× |
| Multi-query requests rerank once on a capped pool | Removes repeated reranking |

End-to-end retrieval (embed + search + rerank), sequential on an 8-thread dev CPU:

| Run | Config | n | p50 | p95 | MRR |
|---|---|---|---|---|---|
| `retrieval-v2` (old) | hybrid, top-20 | 48 | 37–43 s | 50–58 s | 0.867–0.881 |
| `retrieval-v2-norm` | dense, top-10 | 81 | 13.0 s | 16.8 s | 0.854 |
| `retrieval-v5-int8-384` | dense, top-10, int8, max length 384 | 81 | 8.4 s | 9.4 s | 0.846 |

The light configuration kept hit@5 at 0.983 and recall@5 at 0.948, and is what `.env.prod.example` configures.

**Measured under a VPS-like cap** (API container limited to 2 CPUs and 3 GB, default fp32 config,
3 questions, so indicative only): retrieval took 38–57 s and memory peaked at 2.5 GB with no
out-of-memory kill. On small CPU servers the self-hosted cross-encoder is the bottleneck.

## 3. Prompt changes

- Refusals follow the question language, including the critic's regeneration feedback.
- The "Notes / Applications" section may only use material from the context.
- The critic grades claim by claim with a 0.9 support threshold.
- Retrieved chunks are wrapped in `<document>` tags that content cannot close, and the prompt states
  that document content is data (section 4.5).

## 4. Dataset v2 and results

`questions_v2.jsonl` has 81 questions: the 48 from v1 plus 33 new ones, 58 answerable and 23
unanswerable. All new questions were drafted by Claude and reviewed by the author, so they are
**not real student queries**. Abbreviations and missing diacritics are intentional.

| Group | n | Character |
|---|---|---|
| `student-phrasing` | 21 | Abbreviations, no diacritics, colloquial, mixed Vietnamese/English |
| `hard-unanswerable` | 12 | The topic is in the corpus, the specific detail is not |

### 4.1 Retrieval on v2 (`dense_rerank`, top-10)

| Group | n | hit@1 | hit@5 | MRR |
|---|---|---|---|---|
| v1 | 37 | 0.784 | 1.000 | 0.881 |
| student-phrasing | 21 | 0.667 | 0.905 | 0.763 |
| all | 58 | 0.741 | 0.966 | 0.838 |

Informal phrasing hurts the reranked pipeline noticeably. Both complete misses were vocabulary
mismatches.

### 4.2 Query normalization (no LLM)

Expands common abbreviations and restores diacritics using a lexicon built from indexed chunks.

| Group | n | MRR before → after | hit@5 before → after |
|---|---|---|---|
| v1 | 37 | 0.881 → 0.881 | 1.000 → 1.000 |
| student-phrasing | 21 | 0.763 → 0.807 | 0.905 → 0.952 |
| all | 58 | 0.838 → 0.854 | 0.966 → 0.983 |

The evidence is thin: the gain comes from one question and no CI excludes zero. A domain synonym
table was deliberately not added, because that would be fitting the test set.

### 4.3 CRAG and critic on the hard set (`crag-hard-v2`, 33 questions)

| Config | Faithfulness | Correctness | False refusal | Correct refusal | LLM calls | Tokens | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| `dense_rerank` | 0.967 | 0.929 | 0.048 | 12/12 | 1.0 | 2,841 | 15.5 s | 22.6 s |
| `crag` | 0.960 | 0.929 | 0.048 | 12/12 | 3.5 | 8,913 | 22.3 s | 62.4 s |
| `full` | 0.961 | 0.929 | 0.048 | 12/12 | 4.4 | 11,998 | 25.6 s | 57.0 s |

No difference on quality, 3–4× on tokens. Single run, n = 21 answerable.

### 4.4 Orchestrator (`orchestrator-v1`)

The ablation skips the intent classifier, so production makes at least one more LLM call per
question than the ablation's "1 LLM call". It is measured separately:

| Check | Result |
|---|---|
| Intent accuracy | 34/36 = 0.94 [0.82, 0.98]; both errors were `qa` predicted as `teaching` |
| Follow-up rewritten with the right topic | 15/15 [0.80, 1.00] |
| Self-contained question left unchanged | 5/5 |
| Follow-ups the previous pronoun regex would have caught | 2/15 |

The rewrite happens in the same LLM call as classification, so it adds no call.

### 4.5 Prompt injection (`injection-v2`)

Ten attack patterns planted in a retrieved chunk, two runs each. An attack counts as successful if
its marker appears in the answer at all.

| Prompt | Successful attacks |
|---|---|
| Before (no tags or rule) | 13/20 |
| `<document>` tags + system rule | 5/20 |
| + reminder after the context | 0/20 |

This is a small hand-written set against one model. It shows the mitigation works on these
patterns, not that the system is immune.

## 5. Open items

- The answer-quality control run after the latest prompt changes (`prompt-check-v1`) has not
  completed, so the injection mitigations have no before/after faithfulness number yet:
  `python -m eval.run --dataset eval/data/questions_v2.jsonl --configs dense_rerank --tags student-phrasing hard-unanswerable --run-id prompt-check-v1`
- No LLM run covers all 81 questions, so answer metrics here are not directly comparable with `full-v1`.
- Restoring diacritics by frequency can pick the wrong word ("đề thi" → "đề thị").
- The teaching prompt's example section has the same grounding risk as the old "Notes" section.
- No false-premise questions. No real student questions.
- The critic's 0.9 threshold and a cross-vendor critic are unmeasured.
