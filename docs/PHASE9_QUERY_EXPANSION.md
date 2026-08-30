# Phase 9 — Safe query expansion and rewriting

## Purpose and boundary

Phase 9 addresses **vocabulary mismatch**: the user and the source document may
describe the same idea with different words. It adds one retrieval-oriented
alternative query without replacing the user's question:

```text
original question ───────────────→ dense + lexical retrieval ─┐
        │                                                     │
        └→ validated alternative → dense + lexical retrieval ─┤
                                                              ↓
                                                  four-signal RRF + dedup
                                                              ↓
                                           Phase 8 reranks using the original
```

Expansion is an optional recall signal, not an answer and not a correction of
the user. The original question is always searched first and remains the
question passed to the reranker. Conversational rewriting is explicitly
deferred because Lexis does not yet have persistent conversation state.

## When expansion is used

The caller makes the policy explicit with `expansion_enabled`:

- use expansion for evaluation or requests where vocabulary mismatch or
  ambiguous document terminology is a concern;
- use the original-only path for latency-sensitive requests, precise queries
  that already contain the document's terms, and controlled ablations;
- if generation or validation fails, automatically use the exact original-only
  Phase 7 pipeline.

The first implementation generates at most **one** alternative. This keeps the
added cost, latency, and fusion behavior understandable. An automatic query
classifier was deliberately not introduced: the small current dataset is not
enough to calibrate one reliably.

## Protected exact signals

`extract_protected_terms` deterministically identifies text that a paraphrase
must keep verbatim:

- quoted phrases;
- tokens containing digits or identifier punctuation, such as `FY2024-25`,
  `39(1)`, and `PRISM+`;
- all-uppercase acronyms such as `MPC` and `SBP`;
- title-cased name sequences.

The hosted model returns one alternative query, inspectable lexical terms, an
intent-preservation flag, and a rationale through a structured schema. Lexis
then validates the result itself. It rejects an empty or identical alternative,
a reported intent shift, a removed protected term, a newly invented number or
acronym, excessive length, or a paraphrase with no content-word overlap.

Validation failure is not fatal. `expand_query` records the generated text,
lexical terms, usage, and rejection reason, then returns an original-only
result. In the fixed 14-question run, 12 expansions were accepted and two were
safely rejected because the generated text omitted an exact protected signal
(`Pakistan's` and `FY2024-25`). The validator was not loosened merely to
increase the acceptance count.

The selected pinned model is `gpt-4o-mini-2024-07-18`. The official model page
documents Structured Outputs support and the prices used by the evaluation's
cost estimate: <https://developers.openai.com/api/docs/models/gpt-4o-mini>.

## Four-signal fusion and deduplication

For an accepted expansion, Phase 9 runs the existing Phase 7 hybrid retriever
twice. It therefore has four independent ranked signals:

1. original-query dense rank;
2. original-query lexical rank;
3. expanded-query dense rank;
4. expanded-query lexical rank.

For each stable `(document_id, chunk_content_hash)` identity, the score is:

```text
score(chunk) = Σ 1 / (rrf_k + rank_in_signal)
```

A chunk receives at most one contribution from each signal. If the same chunk
is found by all four lists it becomes one output row with four traceable
contributions—it is never duplicated four times. Raw ranks, similarities,
lexical scores, contribution values, and final fused rank remain visible in
the JSON and Markdown reports.

When expansion is disabled or rejected, `retrieve_expanded_hybrid_candidates`
returns the original Phase 7 fused candidates unchanged. This exact fallback
is covered by a focused unit test.

## Controlled evaluation

The comparison uses identical retrieval and reranking settings in both modes:
dense pool `20`, lexical pool `20`, RRF `k=10`, rerank pool `10`, reranker batch
size `5`, relevance cutoff `2`, and final evidence limit `5`.

### Fixed precise-query benchmark

This is the existing Phase 3 dataset. Candidate metrics isolate retrieval and
fusion; final metrics also include the hosted Phase 8 reranker.

| Mode | Candidate Hit@5 | Candidate MRR | Final Hit@5 | Final MRR | Final Recall@5 | Final nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Disabled (`2ad2cdc1a4ca`) | 1.0000 | 0.7115 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Enabled (`5332eb857d15`) | 1.0000 | 0.6859 | 1.0000 | 1.0000 | 1.0000 | 0.9938 |

Candidate-rank analysis found one question helped, eleven unchanged, and one
harmed. The small candidate-MRR reduction is real and is retained in the
report. Final Hit@5, MRR, and Recall@5 did not regress; final nDCG@5 decreased
slightly by 0.0062 because relevant chunks changed order.

### Development vocabulary-mismatch stress set

`evaluation/datasets/phase9_vocabulary_mismatch.json` contains four deliberately
paraphrased development queries. They inherit previously verified evidence
hashes from their Phase 3 parent questions. This is a diagnostic development
set, not a new held-out or independently human-labelled test set.

| Mode | Candidate Hit@5 | Candidate MRR | Final Hit@5 | Final MRR |
| --- | ---: | ---: | ---: | ---: |
| Disabled (`9ee0d45a5939`) | 0.5000 | 0.2604 | 1.0000 | 0.5833 |
| Enabled (`adf34b91969a`) | 0.5000 | 0.2708 | 0.7500 | 0.7917 |

At the retrieval layer, expansion improved candidate MRR and moved the flood
support evidence from rank 8 to rank 6; three cases were unchanged and none
were harmed. Candidate Hit@5 remained 0.5000, so Phase 9 should not be described
as a Hit@5 improvement. The enabled run's final MRR increased, while final
Hit@5 decreased because the same available financial-literacy candidate was
scored differently by the hosted reranker. This is why candidate metrics—not
the nondeterministic downstream reranker alone—are used to isolate expansion's
incremental retrieval effect.

The fixed precise-query run shows no final Hit@5, MRR, or Recall@5 regression
and a small nDCG@5 regression, while the stress set shows a small, measurable
difficult-category ranking gain. This is considered an acceptable trade-off
for the Phase 9 learning-project gate, but the four-query development set is too
small to support a production claim.

## Cost, latency, and operational findings

On the fixed benchmark, expansion added 14 model requests, 2,943 input tokens,
1,154 output tokens, and an estimated `$0.001134`. This includes the two
generated-but-rejected alternatives. End-to-end duration increased from about
`138.6 s` to `183.7 s` in these two sequential runs. These timings also include
retrieval and reranking, so they are observational rather than a formal latency
benchmark.

An early run exposed two operational issues:

- the shared embedding helper inherited long SDK timeout/retry defaults, so
  Phase 9 now opts into a 30-second timeout and zero automatic retries while
  preserving the helper's old default behavior for existing callers;
- a reranker batch of ten produced an invalid/missing pair-ID response on one
  stress query. The controlled comparison uses batches of five. Validation and
  fallback made this failure visible rather than silently accepting bad output.

## Reproducing the comparison

With PostgreSQL running and the Phase 3 document index loaded:

```powershell
.venv\Scripts\python.exe eval_expanded_retrieval.py --expansion disabled --reranker-batch-size 5 --output-dir evaluation/reports/phase9-precise-disabled-batch5
.venv\Scripts\python.exe eval_expanded_retrieval.py --expansion enabled --reranker-batch-size 5 --output-dir evaluation/reports/phase9-precise-enabled-batch5
.venv\Scripts\python.exe eval_expanded_retrieval.py --dataset evaluation/datasets/phase9_vocabulary_mismatch.json --expansion disabled --reranker-batch-size 5 --output-dir evaluation/reports/phase9-vocab-disabled-batch5
.venv\Scripts\python.exe eval_expanded_retrieval.py --dataset evaluation/datasets/phase9_vocabulary_mismatch.json --expansion enabled --reranker-batch-size 5 --output-dir evaluation/reports/phase9-vocab-enabled-batch5
.venv\Scripts\python.exe evaluate_query_expansion.py
```

The analyzer writes the machine-readable and human-readable comparisons to
`evaluation/experiments/phase9_query_expansion_comparison.json` and `.md`.

## Known limitations and deferred work

- Expansion adds a hosted-model request and is not deterministic enough to
  guarantee identical paraphrases on every run.
- Exact-signal extraction is conservative and heuristic; it can reject a safe
  paraphrase, as the two observed fallbacks demonstrate.
- All four RRF signals currently have equal weight.
- The vocabulary-mismatch set is small and development-derived. A larger fixed,
  independently labelled held-out set is needed before production claims.
- One alternative cannot cover every possible interpretation of an ambiguous
  question, but more alternatives would increase cost and duplicate risk.
- Conversational standalone-question rewriting remains deferred until Lexis has
  persistent conversation context and a corresponding evaluation set.
