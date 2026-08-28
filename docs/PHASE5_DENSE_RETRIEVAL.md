# Phase 5 — Dense retrieval

## What changed

Dense retrieval now has two explicit stages in `dense_retrieval.py`:

1. `retrieve_dense_candidates` embeds the question and asks pgvector for a
   broad, distance-ranked candidate pool.
2. `select_dense_context` applies an optional relevance threshold, suppresses
   near-duplicate passages, and returns the smaller set intended for context.

This boundary matters because Phase 7 needs broad dense candidates for fusion,
while answer generation needs a small, non-redundant context. Each
`DenseCandidate` retains stable IDs, page/section provenance, dense rank,
distance metric, raw distance, and a higher-is-better diagnostic similarity.

For cosine, `similarity = 1 - cosine_distance`. For L2, the diagnostic mapping
is `similarity = 1 / (1 + L2_distance)`. The L2 mapping is useful for display
and threshold experiments; it does not change pgvector's raw ordering.

Near-duplicate suppression uses case-insensitive token-set Jaccard similarity:

```text
J(A, B) = |tokens(A) ∩ tokens(B)| / |tokens(A) ∪ tokens(B)|
```

The highest-ranked passage wins. A suppressed passage records its reason, the
winning chunk ID, and measured text similarity, so deduplication is inspectable.

`expand_adjacent_context` implements optional ±N document-order expansion as a
separate operation. It labels every item as a dense anchor or adjacent neighbor
and avoids inserting one neighbor more than once.

## Reproducible experiment

Run the complete ablation (it calls the configured embedding API because it
must create passage-only comparison embeddings):

```powershell
.venv\Scripts\python.exe evaluate_dense_retrieval.py
```

The tracked machine-readable result is
`evaluation/experiments/phase5_dense_comparison.json`. The experiment uses a
fixed five-case development subset for decisions and a disjoint nine-case
held-out subset for verification. The development subset contains the only
labelled unanswerable question. The original complete dataset remains unchanged
for direct comparison with earlier phases.

Run the selected full-dataset baseline with:

```powershell
.venv\Scripts\python.exe eval_retrieval.py `
  --metric cosine `
  --candidate-k 30 `
  --top-k 5 `
  --duplicate-threshold 0.9 `
  --output-dir evaluation/reports/phase5-selected
```

The saved named run is `ae70de100568`:

| Metric | Full fixed dataset |
| --- | ---: |
| Hit@1 | 0.2308 |
| Hit@3 | 0.8462 |
| Hit@5 | 1.0000 |
| MRR | 0.5179 |
| Recall@5 | 0.9231 |
| nDCG@5 | 0.6246 |

The one unanswerable case still returned candidates, reported as an
empty-result/refusal rate of `0.0`. It is not counted as a retrieval hit.

## Decisions and evidence

### Cosine instead of L2

The 424 current document embeddings have norms from `0.999474` to `1.000601`
(mean `0.999979`, population standard deviation `0.000240`). L2 and cosine
produced identical rankings and metrics. Cosine was chosen because its
`1 - distance` score is easier to interpret when vectors are near unit length.
This is an empirical result, not an assumption about the embedding provider.

### Candidate pool 30; final top-k 5

Candidate pools of 10, 20, 30, and 60 produced the same labelled ranking
metrics. Five final chunks retained Hit@5 `1.0`, while 30 broad candidates keep
the candidate stage useful for deduplication and future Phase 7 fusion. Larger
pools added no measured benefit on this 929-chunk database.

### Heading-enriched embeddings retained

The stored embedding input contains source-page and section labels plus passage
text. On development it achieved Hit@5 `1.0`; passage-only embeddings dropped
to `0.75`. Passage-only changed some held-out early ranks, but selection was
made on development data, so heading-enriched content remains the baseline.

### Conservative near-duplicate suppression

Thresholds `0.70`, `0.80`, `0.90`, and an effectively-disabled `1.0` variant
were compared. `0.80` removed four held-out results and reduced multi-evidence
Recall@5 from `0.9444` to `0.8889`. The selected `0.90` avoids that
over-suppression; no pair above `0.90` consumed the selected five-result
contexts. The mechanism remains active for genuinely near-identical passages.

### Adjacent expansion disabled by default

Expanding five anchors by ±1 chunk changed held-out evidence coverage from
`1.0` to `1.0`: it helped zero cases. It grew context from 45 anchor chunks to
109 chunks, adding 64 chunks of noise/cost. The feature remains optional, but
the selected baseline uses a window of zero.

### No production relevance threshold yet

The only unanswerable query had top cosine similarity `0.6116`. Some genuinely
relevant evidence scored lower (`0.5422` and `0.4726`). No threshold could
refuse that query while retaining all development evidence; the best
recall-first calibration is no cutoff (`0.0`/disabled). This is a measured
negative result. A safe threshold requires more varied labelled unanswerable
development cases, so production does not treat cosine similarity as a
calibrated probability.

### No pgvector approximate index yet

`EXPLAIN (ANALYZE, BUFFERS)` was measured on 929 total chunks and 424 chunks in
the target document. Exact cosine search used a sequential scan plus top-N sort
and executed in approximately `12.93 ms` in the captured cold-ish run. A
temporary HNSW cosine index was created and the query replanned; PostgreSQL still
chose the sequential scan, executing in about `8.77 ms` with warm buffers. The
temporary index was removed.

Because the planner did not use HNSW at this size, keeping it would add
write/storage complexity without demonstrated benefit and could introduce
approximate-recall behavior. Reconsider it when corpus growth changes the plan;
compare query plans and candidate recall before adding a migration.

## Selected dense-only baseline

The named configuration is `cosine-c30-k5-dedup90`:

- embedding input: page/heading-enriched `content`;
- embedding model: `text-embedding-3-small`;
- exact pgvector cosine distance;
- 30 broad candidates;
- 5 final context anchors;
- token-Jaccard near-duplicate threshold `0.90`;
- no relevance cutoff;
- no adjacent expansion;
- no approximate vector index at the current corpus size.

This is the independent dense candidate source that Phase 7 may consume. Phase
6 must still build and measure lexical retrieval separately before fusion.
