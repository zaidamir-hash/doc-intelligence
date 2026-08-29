# Phase 7 — Hybrid retrieval with Reciprocal Rank Fusion

## Purpose and pipeline

Dense and lexical retrieval notice different kinds of relevance. Dense search
is good at semantic similarity; lexical search is good at exact words, numbers,
names, and identifiers. Phase 7 runs both independently and combines their rank
positions:

```text
question
  ├─ embedding → dense candidate ranks ───┐
  └─ lexical parsing → lexical ranks ─────┤
                                          ↓
                         stable-identity union and deduplication
                                          ↓
                               Reciprocal Rank Fusion
                                          ↓
                              inspectable hybrid ranking
```

The source retrievers remain independently callable. Phase 7 does not replace
their raw scores, change either source's ranking logic, or implement Phase 8
reranking.

## Why ranks are fused instead of raw scores

Dense cosine similarity and PostgreSQL lexical score are different units. A
cosine similarity of `0.7` is not directly comparable to a lexical score of
`0.7`, so averaging them would assign meaning that the numbers do not have.

Reciprocal Rank Fusion (RRF) uses only each result's position in a list:

```text
RRF(chunk) = Σ 1 / (k + rank_retriever(chunk))
```

`k` is a configurable smoothing constant. A small `k` makes differences near
the top matter more. For the selected `k = 10`, a chunk at dense rank 1 and
lexical rank 2 receives:

```text
dense contribution   = 1 / (10 + 1) = 0.090909
lexical contribution = 1 / (10 + 2) = 0.083333
fused score          = 0.090909 + 0.083333 = 0.174242
```

The selected report stores those ranks, contributions, raw source scores, and
the final fused rank. An automated audit found zero mismatches between stored
fused scores and the sum of their contributions.

If a chunk occurs in only one source list, the missing source contributes zero.
For example, dense rank 3 with no lexical rank scores `1 / (10 + 3) + 0`.
This lets a strong one-source result survive while naturally rewarding chunks
that both retrievers found.

## Stable identity and deterministic ordering

The two lists may contain the same database chunk. Fusion deduplicates their
union with:

```text
(document_id, chunk_content_hash)
```

This identity is stable across database row replacement when document content
and chunk content have not changed. It is safer than using a transient list
position or database UUID alone.

Results sort first by descending fused score. Exact ties use, in order, number
of contributing sources, best source rank, chunk index, and chunk UUID. These
tie-breakers make repeated runs deterministic and inspectable.

## Implementation map

`hybrid_retrieval.py` contains the transparent fusion algorithm:

- `rrf_contribution` implements `1 / (k + rank)` and returns zero for a missing
  rank;
- `fuse_candidates` performs stable-identity union, deduplication, contribution
  calculation, deterministic sorting, and final truncation;
- `retrieve_hybrid_candidates` obtains independently configurable dense and
  lexical pools before calling the pure fusion function;
- `FusedCandidate` retains provenance, raw scores, source ranks, individual RRF
  contributions, fused score, and fused rank.

`evaluation/runner.py` converts fused candidates into the common evaluation
format. `evaluation/reporting.py` displays their diagnostic fields. The fusion
function does not depend on an opaque retrieval framework and is unit-testable
without a database.

## Bounded experiment and selection

`evaluate_hybrid_retrieval.py` compares 12 deliberately bounded configurations:

- balanced pools: `10/10`, `20/20`, and `30/30`;
- asymmetric pools: `30/10` and `10/30`;
- RRF constants: `10`, `30`, `60`, and `100` on selected pool sizes.

The five fixed Phase 5 calibration questions select the configuration. The
other nine answerable questions remain held out during selection. Embeddings
are cached within the experiment so each question is embedded once rather than
once per configuration.

The selected configuration is:

```text
dense pool = 20, lexical pool = 20, RRF k = 10, final top k = 10
```

Ten-candidate source pools lost labelled development evidence. Twenty-candidate
pools restored development Recall@5; increasing both pools to 30 did not improve
development top-five metrics. Among tied configurations, `k=10` gave the best
development nDCG@10.

## Union recall versus final ranking

Candidate union recall asks whether either source retrieved the evidence before
fusion ordering and final truncation. This separates two failure types:

- missing from the union means neither retriever supplied the evidence;
- present in the union but absent from top K means fusion ordered it too low.

For the selected configuration:

| Split | Union Hit | Union mean recall | Final Hit@5 | Final Recall@5 |
| --- | ---: | ---: | ---: | ---: |
| Development | 1.0000 | 1.0000 | 1.0000 | 0.8750 |
| Held out | 1.0000 | 1.0000 | 1.0000 | 0.9444 |

The complete labelled evidence reached the union. The remaining top-five recall
loss is therefore ranking/truncation, not candidate generation. Final Recall@10
is `1.0000` on the complete dataset.

## Named baseline comparison

| Baseline | Hit@1 | Hit@5 | MRR | Recall@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 5 dense | 0.2308 | 1.0000 | 0.5179 | 0.9231 | 0.6246 |
| Phase 6 lexical | 0.5385 | 1.0000 | 0.7244 | 0.8974 | 0.7485 |
| Phase 7 hybrid | 0.5385 | 1.0000 | 0.7115 | 0.9231 | 0.7542 |

Named hybrid run: `8d6a5317d321`.

Hybrid improves dense MRR substantially and improves lexical Recall@5,
nDCG@5, and Recall@10. It does not beat lexical MRR: `0.7115` is slightly below
`0.7244`. This is a diagnosed precision/coverage trade-off, not an across-the-
board win.

Compared with dense, 7 answerable questions were helped, 6 were unchanged, and
none were harmed. Compared with lexical, 1 was helped, 9 were unchanged, and 3
were harmed. The lexical regressions were NCPI (rank 1 to 2), policy rate (rank
2 to 3), and IFSB (rank 2 to 3). Hybrid helped the securities-gains question
relative to lexical (rank 3 to 1). Full per-question and configuration details
are tracked in `evaluation/experiments/phase7_hybrid_comparison.json` and its
Markdown companion.

## Reproduction commands

Dense-only ablation:

```powershell
.venv\Scripts\python.exe eval_retrieval.py
```

Lexical-only ablation:

```powershell
.venv\Scripts\python.exe eval_lexical_retrieval.py `
  --exact-matching supplement `
  --candidate-k 30 `
  --top-k 10 `
  --exact-match-boost 0.25
```

Selected hybrid configuration:

```powershell
.venv\Scripts\python.exe eval_hybrid_retrieval.py `
  --dense-candidate-k 20 `
  --lexical-candidate-k 20 `
  --rrf-k 10 `
  --top-k 10 `
  --output-dir evaluation/reports/phase7-selected
```

Run the bounded hybrid comparison:

```powershell
.venv\Scripts\python.exe evaluate_hybrid_retrieval.py
```

## Known limitations and Phase 8 handoff

- The benchmark has only 13 answerable questions and is rich in exact signals;
  the result should not be generalized to every document domain.
- Equal RRF treatment assumes both rankers deserve equal influence. Weighted
  fusion was intentionally deferred.
- RRF sees rank positions but not whether adjacent raw scores are nearly equal
  or far apart.
- The one unanswerable question still receives results. Refusal calibration is
  not solved here.
- All labelled evidence enters the broad union, but some is below the final top
  five. Phase 8 may rerank the broad candidates; it has not been implemented.
- Experiment selection used a fixed development subset, but the held-out sample
  is too small for strong statistical conclusions.

The Phase 8 input is the deduplicated broad fused list. Keeping `fused_top_k`
separate from source pool sizes allows a future reranker to consume more than
the final answer-context count without changing Phase 7 fusion behavior.
