# Lexis architecture and learning guide

## System boundary

Lexis has four layers:

```text
React UI
   ↓ stable document UUID + question
FastAPI integration and policy
   ↓
Explicit RAG pipeline
   ↓
PostgreSQL + pgvector and hosted model APIs
```

The frontend never performs retrieval. FastAPI never invents document state.
PostgreSQL is authoritative for document identity/lifecycle, and the query
pipeline exposes every ranking boundary needed for learning and evaluation.

## Ingestion data flow

```text
PDF bytes
  → filename/MIME/size/signature validation
  → page-aware extraction
  → conservative cleaning
  → structural units (headings, paragraphs, lists, sentences)
  → ≤200-token chunks with bounded overlap
  → embedding batch
  → transactional chunk replacement
  → ready version in PostgreSQL
```

`pdf_processing.py` preserves page numbers and removes only repeated page-edge
furniture. `chunking.py` uses `cl100k_base`, respects structure when possible,
and hard-splits only when one unit exceeds the token limit. Every chunk records
page range, heading, token/overlap counts, a content hash, and chunker version.

`ingestion.py` gives the exact PDF bytes a SHA-256 identity. It creates a
visible `processing` record before expensive external work. New chunks only
become `ready` in one database transaction after extraction, chunking, and the
complete embedding batch succeed. A failed replacement cannot erase an older
ready version. Identical current input is idempotent; successful changed input
supersedes the previous ready version.

## Persistent schema

`documents` stores stable UUID, filename, content hash, page/chunk counts,
lifecycle status, extraction/chunk/embedding versions, configuration, and
timestamps. `document_chunks` stores stable UUID/hash, passage text, provenance,
computed PostgreSQL `tsvector`, and a 1,536-dimensional pgvector embedding.

Checksum-tracked SQL migrations are discovered and applied in numeric order.
An advisory transaction lock prevents concurrent migration runners, and an
edited already-applied migration is rejected by checksum.

## Query data flow

```text
Question + stable document UUID
  → resolve exact ready filename/content hash
  → safe optional query expansion
  → original dense candidates + original lexical candidates
  → optional expanded dense + expanded lexical candidates
  → Reciprocal Rank Fusion
  → question-passage reranker (0–3)
  → cutoff and duplicate/overlap suppression
  → page-aware generation evidence
  → grounded structured answer
  → quote/claim/citation validation
  → answer or safe refusal
```

### Dense retrieval

The question is embedded with the same embedding model as the chunks. pgvector
computes cosine distance against chunks from the exact ready document version.
The code returns raw distance, interpreted similarity, and dense rank. Candidate
pool size and final selection size are separate so recall is not confused with
context size.

### Lexical retrieval

PostgreSQL `websearch_to_tsquery` and `ts_rank_cd` search the generated
`tsvector`. A separate literal signal supplements identifiers, acronyms,
numbers, and clauses that word-based search can miss. It never calls the
embedding model, so it remains a real independent ablation.

### Reciprocal Rank Fusion

Dense and lexical scores have incompatible units. RRF combines rank positions:

```text
RRF score(chunk) = Σ 1 / (k + rank_in_source)
```

A missing source contributes zero. Stable document/chunk hashes deduplicate the
union. Each contribution and original rank remains visible; no framework hides
the calculation.

### Safe expansion

The original question is always searched. A model may propose one
retrieval-oriented alternative, but validation rejects a rewrite that drops
protected names/numbers/identifiers, introduces conflicting exact signals, or
drifts too far. Failure returns the exact original-only pipeline.

### Reranking and relevance

The reranker receives explicit `(question, passage)` pairs and assigns:

- `0`: irrelevant;
- `1`: related topic but does not answer;
- `2`: useful partial/supporting evidence;
- `3`: directly answers.

Only score-2-or-3 passages are eligible for generation. Reranker failure is
visible and preserves fused order for diagnostics, but unscored fallback
candidates are not silently trusted as answer evidence.

### Grounded generation and citations

Generation receives structured evidence IDs, stable identities, page ranges,
headings, and passage text. The document is untrusted quoted data—not model
instructions. Every factual claim must name supplied evidence and quote text
that exists in that passage and substantively overlaps the claim. Invalid
claims are rejected before display. No adequate evidence produces a refusal;
partial/conflicting evidence produces an explicit limitation.

## API and frontend

`main.py` applies authentication, rate limiting, CORS, upload validation, safe
errors, and stable-ID document routing. Normal query responses contain the
answer, claims, citations, and selected evidence. `debug: true` additionally
returns dense/lexical/fused/reranked ranks, relevance decisions, expansion, and
safe fallback diagnostics.

React restores backend documents after validated session refresh, displays
lifecycle state, queries/deletes by UUID, increments only successful queries,
and renders citations separately from the optional learning panel.

## Evaluation architecture

Retrieval and answer evaluation deliberately remain separate:

```text
retrieval evaluation: question → ranked chunk identities → Hit/MRR/Recall/nDCG
answer evaluation: selected evidence → claims/citations/refusal → quality rubric
```

The fixed retrieval corpus records human-labelled chunk hashes and document
hashes. Index-changing phases require reviewed labels for the new chunk
identities. Answer evaluation uses fixed expected facts/evidence and a
deterministic grader—never an LLM judge—to score correctness, faithfulness,
citation accuracy, completeness, refusal accuracy, and unsupported claims.

This separation can diagnose “retrieval succeeded, generation failed,” which a
single end-to-end answer score would hide.

## Failure boundaries to understand

- Extraction failure: the evidence never entered clean pages.
- Chunking failure: clean evidence exists but not in a labelled chunk.
- Candidate failure: neither dense nor lexical retrieved it.
- Fusion failure: it existed in the union but ranked outside the fused prefix.
- Reranking failure: fused evidence failed relevance/order selection.
- Evidence-selection failure: eligible evidence did not reach context.
- Generation failure: correct evidence reached the model but the answer failed.

## What to practice before the next project

1. Calculate Hit@K, Recall@K, MRR, nDCG, and one RRF example by hand.
2. Trace one chunk through dense rank, lexical rank, RRF contributions,
   reranker score, evidence ID, supporting quote, and page citation.
3. Explain why candidate recall must be measured before blaming a reranker.
4. Explain why overlap improves continuity but can create duplicate evidence.
5. Regrade a saved Phase 11 JSON and identify the earliest failing stage.
6. Change one development-only retrieval setting, predict the trade-off, and
   evaluate it without modifying fixed test labels.

If those are clear, Lexis has accomplished its learning purpose.
