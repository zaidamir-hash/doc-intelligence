# Lexis retrieval evaluation

Phase 1 evaluates retrieval only. It never calls `generate_answer` or the chat
model. Its purpose is to answer one narrow question: **did the current dense
retriever rank the human-labelled evidence highly enough?**

Phase 5 adds configurable two-stage dense retrieval. The selected dense-only
command and ablation results are documented in `docs/PHASE5_DENSE_RETRIEVAL.md`.
Diagnostics now retain candidate rank, distance metric, raw distance, and
interpreted similarity; unanswerable empty-result behavior is reported
separately from answerable ranking metrics.

Phase 6 adds an independently runnable PostgreSQL lexical baseline. Its
configuration, exact-identifier ablation, category comparison, and commands are
documented in `docs/PHASE6_LEXICAL_RETRIEVAL.md`. Lexical diagnostics expose
lexical rank, combined score, raw FTS score, and matched identifier signals;
they never require question embeddings.

Phase 7 combines independent dense and lexical candidate lists with transparent
Reciprocal Rank Fusion. The selected `20/20` pools and `k=10` configuration,
source-rank contribution traces, bounded ablation, union-recall analysis, and
standalone commands are documented in `docs/PHASE7_HYBRID_RRF.md`. The named
hybrid run `8d6a5317d321` produced 1.0000 Hit@5, 0.7115 MRR, 0.9231 Recall@5,
and 0.7542 nDCG@5.

Phase 8 reranks the ten fused candidates with explicit question-passage scores.
The selected cutoff, pre-rerank candidate recall, rank movements, failure
fallback, measured tokens/cost/latency, and reproducible commands are documented
in `docs/PHASE8_RERANKING.md`. Named run `81754b6e9c15` achieved 1.0000 MRR,
1.0000 Hit@5, 0.9615 Recall@5, and 0.9579 nDCG@5, with complete candidate recall
before reranking.

Phase 9 optionally adds one validated retrieval-oriented alternative while
always searching the original question. Original and expanded dense/lexical
ranks are fused by the same transparent RRF mechanics and deduplicated by stable
chunk identity. On the four-query development vocabulary-mismatch stress set,
candidate MRR improved from 0.2604 to 0.2708 while candidate Hit@5 remained
0.5000. On the fixed precise-query benchmark, final Hit@5, MRR, and Recall@5
remained 1.0000; nDCG@5 changed from 1.0000 to 0.9938. Full diagnostics,
safe-fallback cases, cost, limitations, and commands are documented in
`docs/PHASE9_QUERY_EXPANSION.md`.

Phase 10 connects the advanced retrieval pipeline to grounded answer
generation. Only reranker score-2-or-3 passages enter a page-aware structured
context; every factual claim must cite a supplied stable evidence ID, and the
API exposes broad candidates, generation evidence, and citations separately.
A bounded six-case validation covered two labelled answers, an unanswerable
question, a partial question, conflicting evidence, and a document prompt
injection; all six structural checks passed. This is a smoke validation rather
than Phase 11 answer scoring. See `docs/PHASE10_GROUNDED_GENERATION.md`.

## Baseline history and limitation

The repository did not contain the questions or complete human labels used to
produce the previously reported long-document result (75% Hit@5, 0.494 MRR) or
the two-page result (100% Hit@5, 0.917 MRR). The old `eval_retrieval.py` only
identified long-document chunks 52, 53, 143, and 188 and printed their text.
Questions cannot be reconstructed reliably from chunk IDs, so the harness does
not fabricate them.

The uploaded 91-page report was used to create a fresh, human-labelled
long-document baseline containing 14 questions: 13 answerable questions and one
unanswerable diagnostic question. Labels were fixed before retrieval was run and
were checked against the complete indexed chunk set, not only the retriever's top
results.

The first fresh baseline run, `f4c150dab9ef`, produced:

- Hit@5: 0.8462
- MRR: 0.5955
- Hit@10: 1.0000
- Recall@10: 0.9744

These figures do not reproduce or disprove the old 75% Hit@5 and 0.494 MRR
figures because the questions and labels are different. They establish the first
versioned, reproducible long-document baseline for future Lexis experiments.

Phase 2 has since replaced the live 340-chunk index with a 404-chunk,
page-bounded cleaned index. Its matching labels are stored in
`evaluation/datasets/phase2.json`, and its first run (`bebc552d57bd`) produced
0.9231 Hit@5 and 0.7179 MRR. See `docs/PHASE2_EXTRACTION.md` for the full
before/after analysis. The original dataset and report remain preserved; their
chunk labels must not be run against the Phase 2 index.

Phase 3 has now replaced the live evaluation index with 424 token-aware,
structure-aware chunks using `cl100k_base`, a 200-token maximum, and 30-token
requested overlap. Its matching labels are in `evaluation/datasets/phase3.json`.
The database-backed pgvector run `9475274a22ba` produced 1.0000 Hit@5 and
0.5179 MRR. Coverage improved over Phase 2, but first-result ranking worsened;
see `docs/PHASE3_CHUNKING.md` for the bounded comparison and tradeoff analysis.
The default `eval_retrieval.py` dataset is now the Phase 3 dataset. Older labels
remain historical and must only be used with their matching index.

Phase 4 moved that index into the versioned `documents`/`document_chunks`
schema and bound the current dataset to stable raw-document and chunk-content
hashes. Database-backed run `f4f3d2874679` reproduced the Phase 3 result exactly:
1.0000 Hit@5 and 0.5179 MRR. Current reports expose document UUID/hash, chunk
UUID/hash, page range, and section title. Historical datasets without hashes
remain supported in explicit `legacy_chunk_index` mode. See
`docs/PHASE4_DATABASE.md` for schema and lifecycle details.

The developer explicitly waived the two-page contract comparison on 2026-08-27.
For this learning project's Phase 1 scope, the reproducible 91-page stress test
is the accepted baseline. This is a documented scope decision, not a claim that
the old contract result was reproduced.

## Dataset format

Use `evaluation/datasets/example.json` only as a schema example. It is fictional
and must never be reported as a Lexis benchmark.

Each real case contains:

- `id`: a stable, unique question identifier.
- `filename`: the exact filename currently stored in PostgreSQL.
- `question`: the exact query sent to the embedding model.
- `answerable`: whether the indexed document contains relevant evidence.
- `relevant_chunk_indices`: every chunk accepted as relevant by a human labeler.
- `relevance_grades`: a positive integer grade for every relevant chunk. Use 1
  for ordinary binary relevance; larger values identify stronger evidence for
  nDCG.
- `reference_answer`: optional and reserved for later answer evaluation.
- `notes`: optional explanation of ambiguous or multiple valid evidence.
- `tags`: question categories such as `paraphrase`, `number`, `name`,
  `acronym`, `date`, `clause`, `exact-term`, or `unanswerable`.

Chunk indices are tied to the current index. If a document is re-chunked, its
labels must be reviewed rather than silently reused. Stable document/chunk
identity and versioning are scheduled for later phases.

## One-command baseline

Start PostgreSQL, ensure the labelled documents are already indexed, and run from
the repository root:

```powershell
.venv\Scripts\python.exe eval_retrieval.py
```

Optional arguments:

```powershell
.venv\Scripts\python.exe eval_retrieval.py `
  --dataset evaluation/datasets/phase3.json `
  --output-dir evaluation/reports `
  --top-k 10 `
  --preview-characters 240
```

The command creates one JSON report for machine analysis and one Markdown
report for human inspection. Both share a run ID and include configuration,
latency, aggregate metrics, and per-question rankings with L2 distances.

## Metric definitions

All retrieval metrics are macro-averaged: each answerable question contributes
equally, regardless of how many relevant chunks it has.

### Hit@K

Hit@K is 1 for a question when at least one relevant chunk appears in the first
K results; otherwise it is 0. The reported Hit@K is the mean over answerable
questions.

### Reciprocal rank and MRR

Reciprocal rank is `1 / rank` for the first relevant result. If no relevant
chunk is retrieved, it is 0. Mean Reciprocal Rank (MRR) is the mean reciprocal
rank over answerable questions. MRR rewards placing the first useful result
near the top.

### Recall@K

Recall@K is the number of labelled relevant chunks found in the first K results
divided by the total number of labelled relevant chunks. Unlike Hit@K, it shows
whether multiple pieces of evidence were recovered.

### nDCG@K

Normalized Discounted Cumulative Gain rewards highly graded evidence appearing
early in the ranking. The score is the ranking's discounted gain divided by the
gain of the ideal ordering. With binary grades it still rewards early relevant
results; with graded labels it distinguishes primary from supporting evidence.

### Unanswerable questions

The current retriever always returns nearest chunks and has no calibrated
relevance threshold. Therefore an unanswerable question has no meaningful
"successful retrieval" under the current design. Such cases are retained in
diagnostics but excluded from Hit/MRR/Recall/nDCG. A later phase will add and
evaluate a relevance/refusal decision.

## Labelling procedure

1. Write questions independently of the retriever's ranking where possible.
2. Inspect the complete document/chunk set, not only the current top results.
3. Label every chunk that contains sufficient or supporting evidence.
4. Use notes for ambiguous labels and multiple valid answers.
5. Do not relabel evidence merely to improve a metric.
6. Keep the test split fixed after it becomes the comparison baseline.
7. Use a development or validation split for tuning future chunk sizes,
   thresholds, and retrieval parameters; do not tune repeatedly on the test
   split.
