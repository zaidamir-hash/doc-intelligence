# Phase 8 — Question-passage reranking

## Purpose and boundary

Phase 7 retrieves a broad, high-recall candidate list. Phase 8 improves the
order by examining the question together with each candidate passage:

```text
question
  → Phase 7 dense + lexical + RRF candidates
  → explicit (question, passage) pairs
  → independent 0–3 relevance scores
  → score descending, fused rank as tie-breaker
  → complete reranked list for evaluation
  → relevance cutoff + final evidence limit
```

Reranking cannot recover a passage that retrieval missed. Lexis therefore
reports candidate-pool recall before reporting reranked metrics.

## Approach comparison and selection

| Approach | Quality and operation | Cost/latency | Project trade-off |
| --- | --- | --- | --- |
| Local cross-encoder | Specialized pair scorer; keeps text local | No per-call fee, but requires model download and local CPU/GPU inference | Rejected for this phase because `torch`, `transformers`, and `sentence-transformers` are absent and would be large new dependencies |
| Dedicated hosted reranking API | Specialized and easy to call | Adds another paid provider, API key, and network dependency | Rejected to keep the learning project bounded |
| Existing OpenAI small model | General model, explicit rubric, strict structured output | Selected run cost `$0.011591`; mean `7.49 s` per question | Selected because the project already has the SDK/key and the complete mechanics remain visible |

The selected model is the pinned `gpt-4o-mini-2024-07-18` snapshot. Official
OpenAI documentation describes GPT-4o mini as a fast, affordable model for
focused tasks, lists Structured Outputs support, and publishes prices of `$0.15`
per million input tokens and `$0.60` per million output tokens:
<https://developers.openai.com/api/docs/models/gpt-4o-mini>.

Pinning a snapshot reduces model-version drift. It does not guarantee identical
scores: two Phase 8 runs showed small ordinal-score variation even at
temperature zero. The fixed human labels, raw outputs, and run IDs make that
variation visible.

## Explicit pairs and scoring rubric

`build_question_passage_pairs` creates one visible pair per candidate. A passage
contains its section title, when present, and clean `passage_text`; it does not
use the heading/page-enriched embedding representation as if that were source
prose.

Pairs are sent in configurable batches to reduce network round trips, but the
instructions require every pair to be judged independently:

| Score | Meaning |
| ---: | --- |
| 0 | Unrelated or no useful evidence |
| 1 | Topically related but does not answer the question |
| 2 | Useful supporting or partial evidence |
| 3 | Directly answers the question |

The response must match a Pydantic-backed Structured Output containing exactly
one result for every stable pair ID. Missing, duplicate, unknown, or out-of-range
scores fail validation.

The ranking rule is deliberately simple:

```text
sort key = (-reranker_score, original_fused_rank)
```

Higher relevance comes first. When scores tie, Phase 7 order is retained. This
prevents arbitrary model output order from changing equal-score candidates.

## Separate ranking and evidence selection

`RerankResult` contains two outputs:

- `ranked_candidates` retains the complete reranked prefix for MRR, Hit@K,
  Recall@K, and nDCG;
- `selected_candidates` applies the relevance cutoff and configurable final
  evidence count for later generation.

This separation prevents a cutoff experiment from being confused with a ranking
experiment. Every output keeps the original fused rank, final rank, score,
rationale, and fallback marker, as well as all Phase 7 diagnostics.

## Cutoff calibration

Cutoffs `0`, `1`, `2`, and `3` were compared offline on the fixed Phase 7
development subset. The selection rule prioritizes an empty result for the
labelled unanswerable case, then answerable Hit@5, Recall@5, and nDCG@5. On a
complete tie, score `2` is preferred because the rubric explicitly defines it as
useful supporting evidence.

Selected cutoff: `2`.

- The unanswerable Bitcoin-reserve question received ten zero scores and
  selected no evidence.
- Every answerable case retained at least one score-2-or-3 passage.
- On the nine held-out answerable questions, selected evidence achieved 1.0000
  Hit@5, MRR, and Recall@5.
- Cutoff `3` was not selected because an earlier fixed run showed that
  direct-answer-only filtering can discard labelled supporting evidence.

Only one unanswerable question exists, and it participated in development
calibration. This proves the code path and behavior on that case; it does not
establish a general refusal threshold. A larger unanswerable validation set is
needed before production use.

## Candidate recall and ranking results

Named selected run: `81754b6e9c15`.

The ten fused candidates contained all labelled evidence before reranking:

```text
candidate Hit     = 1.0000
candidate Recall  = 1.0000
```

| Baseline | Hit@1 | Hit@5 | MRR | Recall@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 7 hybrid | 0.5385 | 1.0000 | 0.7115 | 0.9231 | 0.7542 |
| Phase 8 reranked | 1.0000 | 1.0000 | 1.0000 | 0.9615 | 0.9579 |

Six questions were promoted, seven were unchanged, and none were demoted. For
example:

- NCPI moved from fused rank 2 to reranked rank 1;
- policy-rate reduction moved from rank 3 to rank 1;
- IFSB adoption moved from rank 3 to rank 1;
- PRISM+ moved from rank 4 to rank 1;
- statutory submission moved from rank 3 to rank 1;
- NFIS targets moved from rank 2 to rank 1.

Recall@5 is below the candidate Recall@10 because some questions have multiple
labelled supporting chunks. Reranking placed the first correct evidence at rank
1 for every answerable question but did not always fit every supporting passage
inside the first five.

## Latency and cost

The selected run measured:

- 14 requests for 140 question-passage pairs;
- 35,068 input tokens and 10,552 output tokens;
- estimated standard-token cost: `$0.011591`;
- total reranker latency: `104.79 s`;
- mean reranker latency: `7.49 s` per question;
- complete end-to-end evaluation duration: `115.38 s`;
- zero fallback cases.

The cost is acceptable for this bounded learning evaluation. The latency is
acceptable for a demo but not an optimized production interaction. Caching,
parallel calls, a local cross-encoder, or a dedicated service could be evaluated
in a larger project; they are not added here without measurement and scope.

## Failure behavior

Each hosted request has a 30-second timeout and zero automatic retries. If a
request fails or its structured scores are invalid:

1. the original fused order is preserved;
2. no relevance cutoff is applied to missing scores;
3. the ordinary final evidence limit is applied;
4. every returned candidate is marked `used_fallback=True`;
5. the exception type/message and fallback case count are recorded;
6. the evaluation CLI exits nonzero after writing the diagnostic report.

This maintains availability without silently presenting Phase 7 order as a
successful reranked result.

## Commands

Run the selected configuration:

```powershell
.venv\Scripts\python.exe eval_reranked_retrieval.py `
  --dense-candidate-k 20 `
  --lexical-candidate-k 20 `
  --rrf-k 10 `
  --rerank-k 10 `
  --final-k 5 `
  --relevance-cutoff 2 `
  --batch-size 10 `
  --model gpt-4o-mini-2024-07-18 `
  --output-dir evaluation/reports/phase8-selected
```

Analyze the named Phase 7 and Phase 8 reports without new model calls:

```powershell
.venv\Scripts\python.exe evaluate_reranking.py
```

Tracked comparison outputs:

- `evaluation/experiments/phase8_reranking_comparison.json`;
- `evaluation/experiments/phase8_reranking_comparison.md`.

## Known risks and Phase 9 handoff

- Ordinal LLM scores are relevance judgments, not calibrated probabilities.
- A pinned model and temperature zero reduce but do not eliminate variation.
- Batch scoring may still allow candidates in one batch to influence judgments,
  despite the independent-scoring instruction.
- The benchmark is small and exact-signal-heavy.
- The single unanswerable case is insufficient for general threshold claims.
- Hosted latency has a long tail; one unbounded attempt was stopped, motivating
  the explicit timeout and zero-retry policy.
- Cost estimates use published standard token rates; account-specific pricing,
  caching, or future price changes may differ.
- Phase 8 does not implement query expansion, generation, or citations.

Phase 9 can consume the ordered, scored candidate list and the score-2 evidence
cutoff, but it must preserve the original question and evaluate expansion as a
separate retrieval signal.
