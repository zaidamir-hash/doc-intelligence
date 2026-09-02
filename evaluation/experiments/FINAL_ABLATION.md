# Lexis final ablation and project closure

This report consolidates the versioned experiments already performed during
Phases 1–11. It does not rerun historical configurations, change labels, or
select only favorable outcomes. The machine-checkable companion is
`evaluation/experiments/final_ablation_report.json`.

## Comparison policy

- The original 91-page benchmark contains 13 answerable questions and one
  unanswerable diagnostic. The questions were fixed; each index-changing phase
  uses labels reviewed against that phase's exact chunk identities.
- Hit@5, MRR, Recall@5, and nDCG@5 evaluate retrieval only.
- Reranker and expansion rows also report hosted-model cost and observed total
  duration where recorded. Earlier stages did not consistently record model
  cost, so the report leaves those values unknown rather than inventing them.
- The historical two-page contract result is context only. Its questions and
  labels are unavailable, and reproducing it was explicitly waived.

## Retrieval ablation

| Stage | Run | Hit@5 | MRR | Recall@5 | nDCG@5 | Observed duration | Recorded model cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original dense baseline | `f4c150dab9ef` | 0.8462 | 0.5955 | 0.7564 | 0.5842 | not recorded | not recorded |
| Clean page-aware extraction + dense | `bebc552d57bd` | 0.9231 | 0.7179 | 0.7385 | 0.6480 | not recorded | not recorded |
| Token/structure chunks + dense | `9475274a22ba` | 1.0000 | 0.5179 | 0.9231 | 0.6246 | not recorded | not recorded |
| Final two-stage dense | `ae70de100568` | 1.0000 | 0.5179 | 0.9231 | 0.6246 | 16.89 s | not recorded |
| PostgreSQL lexical | `e9ccf0353cc7` | 1.0000 | 0.7244 | 0.8974 | 0.7485 | 1.09 s | $0 |
| Hybrid RRF | `8d6a5317d321` | 1.0000 | 0.7115 | 0.9231 | 0.7542 | 1.75 s | not recorded |
| Hybrid + reranking | `81754b6e9c15` | 1.0000 | 1.0000 | 0.9615 | 0.9579 | 115.38 s | $0.011591 |
| Hybrid + reranking + expansion | `5332eb857d15` | 1.0000 | 1.0000 | 1.0000 | 0.9938 | 183.68 s | $0.013545 |

The final row's candidate MRR was `0.6859`, below hybrid's `0.7115`; its perfect
final MRR comes after the hosted reranker. This distinction prevents the answer
pipeline from hiding a retrieval regression.

## What each addition changed

- Cleaning and page-aware extraction improved Hit@5 and MRR, but slightly
  reduced Recall@5 because the chunk identities and evidence distribution
  changed.
- Token/structure chunking recovered evidence for all 13 answerable questions
  inside the first five, but MRR fell. Coverage improved while early ordering
  became worse—the reason later retrieval stages were necessary.
- Final dense retrieval kept complete Hit@5 and added inspectable two-stage
  selection, thresholds, and deduplication; it did not magically repair MRR.
- Lexical retrieval was much faster in the recorded run and improved MRR on
  dates, names, numbers, and acronyms. Dense still had higher Recall@5 and a
  slightly better exact-term category MRR (`0.3444` versus `0.3056`).
- RRF used the complementary lists. It improved dense MRR and lexical recall and
  nDCG, but its MRR (`0.7115`) remained slightly below lexical (`0.7244`). The
  union contained all labelled evidence, so remaining loss was ordering.
- Reranking promoted the first labelled passage to rank one for every
  answerable question. It provided the largest ordering gain and also the
  largest latency increase before expansion.
- Expansion was deliberately safe: the original query was always searched and
  invalid rewrites fell back. On the precise test set it preserved final
  Hit@5/MRR/Recall@5 with a small nDCG decrease versus the disabled Phase 9 run.
  On the four-query development vocabulary-stress set it improved candidate MRR
  from `0.2604` to `0.2708`, but final Hit@5 varied with reranker scoring. This
  is useful evidence, not a production-quality generalization claim.

## Category behavior

On the fixed labelled corpus, selected dense versus lexical category MRR was:

| Category | Dense | Lexical |
| --- | ---: | ---: |
| Acronym | 0.5067 | 0.6500 |
| Date | 0.3833 | 0.7500 |
| Name | 0.7500 | 1.0000 |
| Number | 0.5200 | 0.7833 |
| Exact term | 0.3444 | 0.3056 |

This small corpus is exact-signal-heavy, so the category results explain why
lexical search helps without proving it will dominate on every document type.

## Answer quality remains separate

Phase 11 run `01903932da6b` reported retrieval success `1.0000`, answer success
`0.8333`, and unsupported-claim rate `0.0000`. Five of six answer cases passed;
one answer-model timeout occurred after successful retrieval and was correctly
attributed to generation. The answer score is not substituted for any retrieval
metric.

## Closure decision

Lexis now includes inspectable extraction, token/structure chunking, versioned
PostgreSQL/pgvector storage, dense retrieval, lexical retrieval, explicit RRF,
reranking, safe query expansion, relevance/refusal handling, grounded answers,
page-aware citations, separate retrieval/answer evaluation, and an integrated
API/UI. Remaining limitations are documented in the root README and are closed
as scope rather than automatically expanded. The next learning step should be a
larger project, not more unmeasured features in Lexis.
