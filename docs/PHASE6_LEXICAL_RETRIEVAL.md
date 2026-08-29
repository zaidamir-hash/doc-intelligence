# Phase 6 — Lexical retrieval

## Purpose and pipeline

Lexical retrieval searches words and identifiers directly. It does not create a
question embedding, call OpenAI, or use pgvector:

```text
question
  → safe query parsing
  → PostgreSQL tsquery
  → GIN-backed full-text candidates
  + optional literal identifier candidates
  → transparent lexical score and rank
```

`lexical_retrieval.py` is independent from `dense_retrieval.py`. Phase 7 can
consume both candidate lists later, but Phase 6 does not fuse them.

## Stored text representation

Migration `0004_weight_lexical_search_vector.sql` rebuilds the generated
`search_vector` as:

```sql
setweight(to_tsvector('english', coalesce(section_title, '')), 'A')
|| setweight(to_tsvector('english', coalesce(passage_text, '')), 'B')
```

PostgreSQL maintains this column automatically. Section terms receive weight
`A`; body terms receive `B`. A GIN index named `ix_chunks_search_vector` indexes
the lexemes, while `ix_chunks_document_id` restricts search to one document.

The `english` text-search configuration was selected because the evaluation
corpus is English. It lowercases and stems ordinary words and removes English
stop words. PostgreSQL's parser keeps many numeric lexemes but tokenizes
punctuation according to token type—for example, `PRISM+` becomes `prism`, and
`39(1)` becomes `39` and `1`.

## Safe query parsing

Passing a normal question directly to `websearch_to_tsquery` joins useful terms
with `AND`. The PRISM+ question initially became approximately:

```text
acronym & prism & stand
```

No passage contained all three lexemes, so it returned no rows. Lexis therefore
extracts inspectable tokens and joins them with `OR` before passing the whole
string as a bound value to PostgreSQL's `websearch_to_tsquery` function.
SQLAlchemy constructs the SQL expressions; user text is never concatenated into
raw SQL.

An explicitly quoted user phrase remains quoted. For example,
`"Climate Risk Fund-I"` becomes a PostgreSQL phrase query. Stop-word-only input
produces an empty tsquery and safely returns no rows.

## Ranking and diagnostics

Primary full-text relevance uses:

```sql
ts_rank_cd(search_vector, tsquery, 32)
```

`ts_rank_cd` considers term frequency, weights, and positional coverage.
Normalization flag `32` maps rank to `rank / (rank + 1)`, reducing unbounded
score growth without changing rank order.

Each `LexicalCandidate` exposes:

- lexical rank and combined lexical score;
- raw `ts_rank_cd` score;
- literal exact-match count and matched terms;
- parsed search text;
- stable document/chunk IDs, pages, section, and passage text.

These scores are meaningful inside one lexical result list, but they are not
probabilities and must not be numerically averaged with cosine scores. Phase 7
will combine ranks rather than incompatible score scales.

## Exact-identifier supplement

PostgreSQL FTS deliberately normalizes punctuation. That is useful for prose but
can lose distinctions in identifiers such as `PRISM+`, `FY2024-25`, `39(1)`,
and `Fund-I`.

Lexis extracts only explicit quoted phrases, uppercase acronyms, and
punctuation-bearing identifiers for a case-insensitive literal supplement. Each
unique literal match contributes `0.25` to the combined lexical score:

```text
lexical score = ts_rank_cd score + 0.25 × exact identifier matches
```

The FTS query and literal query run separately. This preserves the GIN-backed
primary path; Python merges candidates by stable chunk ID and sorts the visible
components. Pure FTS remains independently callable with
`--exact-matching disabled`.

The supplement was retained only after measurement:

| Lexical mode | Hit@5 | MRR | Recall@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| Pure PostgreSQL FTS | 0.7692 | 0.4559 | 0.7051 | 0.5058 |
| FTS + exact identifiers | 1.0000 | 0.7244 | 0.8974 | 0.7485 |

Representative first-relevant-rank changes:

- Climate Risk Fund-I: outside top 10 → rank 1;
- AAOIFI: rank 7 → rank 1;
- NCPI FY25/FY24: rank 5 → rank 1;
- public debt-to-GDP: rank 4 → rank 1;
- PRISM+: rank 6 → rank 4.

## Named baselines and commands

Pure FTS:

```powershell
.venv\Scripts\python.exe eval_lexical_retrieval.py `
  --exact-matching disabled `
  --output-dir evaluation/reports/phase6-pure-fts-v2
```

Named run: `9b87597ceb5b`.

Selected supplemented lexical baseline:

```powershell
.venv\Scripts\python.exe eval_lexical_retrieval.py `
  --exact-matching supplement `
  --candidate-k 30 `
  --top-k 10 `
  --exact-match-boost 0.25 `
  --output-dir evaluation/reports/phase6-supplement-v2
```

Named run: `e9ccf0353cc7`.

Generate the tracked dense/lexical category comparison after those named runs:

```powershell
.venv\Scripts\python.exe evaluate_lexical_retrieval.py
```

Outputs:

- `evaluation/experiments/phase6_lexical_comparison.json`;
- `evaluation/experiments/phase6_lexical_comparison.md`.

## Dense versus lexical

| Baseline | Hit@1 | Hit@5 | MRR | Recall@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 5 dense | 0.2308 | 1.0000 | 0.5179 | 0.9231 | 0.6246 |
| Selected lexical | 0.5385 | 1.0000 | 0.7244 | 0.8974 | 0.7485 |

Selected category MRR comparisons show where exact signals help:

| Category | Dense | Lexical |
| --- | ---: | ---: |
| acronym | 0.5067 | 0.6500 |
| date | 0.3833 | 0.7500 |
| name | 0.7500 | 1.0000 |
| number | 0.5200 | 0.7833 |
| exact-term | 0.3444 | 0.3056 |

Lexical retrieval is stronger overall on this small exact-signal-heavy set, but
it is not universally better: the exact-term category's early rank is slightly
worse and dense Recall@5 is higher (`0.9231` versus `0.8974`). This measurable
complementarity is the reason to proceed to rank fusion later—not permission to
implement it during Phase 6.

The full lexical evaluation completed in about `1.09 s`, versus `16.89 s` for
the recorded dense run. The dense duration includes an external question
embedding call, so this is end-to-end latency, not a database-only benchmark.

## Query plans and indexes

For a PRISM+ query, PostgreSQL used a `BitmapAnd` of:

- `ix_chunks_document_id`;
- `ix_chunks_search_vector`.

The measured GIN-backed FTS plan completed in approximately `2.7–6.4 ms` across
captured runs. The separate literal scan used the ordered document index,
checked 424 document chunks, and completed in about `7.0 ms`. Keeping the paths
separate avoids the combined `OR` plan that forced a full 929-row sequential
scan.

The literal path may eventually need a trigram index or stricter identifier
lookup if the corpus grows. Adding `pg_trgm` now is not justified by the current
size or measured latency.

## Known limitations

- Stemming can merge related word forms and cannot preserve every domain nuance.
- English stop words cannot retrieve a passage by themselves.
- Unquoted words are OR terms, not an exact phrase or all-terms requirement.
- Quoted phrases use PostgreSQL lexeme positions; they are not byte-for-byte
  string equality and punctuation may still be normalized.
- Punctuation-heavy identifiers are split by FTS; the literal supplement
  compensates only for inspectable query patterns.
- Literal matching is case-insensitive substring matching, not Unicode-aware
  fuzzy matching, typo correction, or token-boundary matching.
- Common terms can create broad candidate sets; `ts_rank_cd` must order them.
- A lexical score is query-relative, not a calibrated confidence probability.
- The only unanswerable question still returns lexical candidates. Refusal
  calibration is not solved by Phase 6.
- The current dataset is small and exact-signal-heavy, so results should not be
  generalized to every document domain.

## Selected Phase 6 baseline

The selected lexical candidate source is PostgreSQL English FTS with weighted
headings, OR-style safe query parsing, normalized `ts_rank_cd`, 30 candidates,
and a `0.25` literal boost for transparent identifier signals. It remains
independently callable and provides the lexical ranks/scores required by Phase
7. No hybrid retrieval or RRF has been implemented.
