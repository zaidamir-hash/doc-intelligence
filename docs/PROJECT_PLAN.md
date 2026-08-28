# Lexis Advanced RAG Project Plan

## Project objective

Lexis began as a learning project for understanding the basic retrieval-augmented generation (RAG) pipeline: PDF extraction, chunking, embedding, vector storage, retrieval, prompt augmentation, and answer generation. The current implementation successfully demonstrates that baseline, but evaluation on a 91-page PDF exposed its limitations: approximately 75% Hit@5 and 0.494 MRR, compared with 100% Hit@5 and 0.917 MRR on a simple two-page contract.

The objective is to evolve Lexis from a vector-only baseline into a bounded, production-shaped advanced RAG system while preserving its primary educational purpose. The finished project must demonstrate and measure the progression from basic dense retrieval to stronger document processing, hybrid retrieval, rank fusion, reranking, query expansion, grounded generation, citations, and separate retrieval/answer evaluation.

This is not a race to assemble popular tools. Each major mechanism must remain visible and understandable. Work will proceed one phase at a time, with a measurable baseline, focused implementation, line-by-line review, and before/after evaluation. The project must remain time-boxed so that it teaches the transferable core of advanced RAG without expanding into an indefinite research or infrastructure project.

The target pipeline is:

```text
PDF extraction and cleaning
        ↓
Structure-aware, token-based chunking
        ↓
Dense embeddings + lexical index
        ↓
Original query + controlled query expansion
        ↓
Dense retrieval + lexical retrieval
        ↓
Reciprocal Rank Fusion (RRF)
        ↓
Question-passage reranking
        ↓
Relevance threshold + optional neighboring context
        ↓
Grounded generation
        ↓
Page-aware citations
        ↓
Separate retrieval and answer evaluation
```

## Architectural decisions already agreed upon

### Learning and delivery principles

- [ ] Introduce one major RAG concept at a time.
- [ ] Explain the problem, algorithm, data structures, and expected effect before implementing a phase.
- [ ] Establish a reproducible measurement before changing the behavior being measured.
- [ ] Keep implementations small and explicit where practical instead of hiding core mechanics behind a large RAG framework.
- [ ] Review every material code change line by line.
- [ ] Test each addition independently and compare it with the previous version.
- [ ] Use ablation-style experiments so improvements can be attributed to a specific change.
- [ ] Record unsuccessful experiments as well as successful ones.
- [ ] Complete the defined advanced-RAG scope without turning Lexis into an endless research project.

### Retrieval and generation decisions

- The current Lexis behavior is the baseline: character-based chunks, OpenAI embeddings, pgvector L2 search, top-eight dense results, and grounded answer generation.
- The target is advanced RAG, not a vector-only final system.
- Improve and evaluate extraction/chunking before assuming that retrieval algorithms alone can repair damaged document representation.
- Dense retrieval and lexical retrieval must first be inspectable and measurable independently.
- Hybrid retrieval will combine rankings using Reciprocal Rank Fusion rather than directly averaging incompatible dense and lexical score scales.
- RRF will be implemented transparently so that each retriever's rank contribution can be inspected.
- Reranking will operate after broad candidate retrieval; it is responsible for precision/order, not for recovering evidence absent from the candidate pool.
- Candidate recall before reranking and final ranking quality after reranking must be measured separately.
- Query rewriting must not discard or replace the original user question. The original query will always be searched, with controlled expansion used as an additional retrieval signal.
- Retrieval quality must be measured before trusting generated answers.
- Retrieval evaluation and answer evaluation must remain separate so a failure can be attributed to extraction, chunking, retrieval, reranking, or generation.
- Weak evidence must not automatically be sent to the answer model. A calibrated relevance check/refusal path is required.
- Answers must eventually cite page-aware evidence and refuse when reliable evidence is unavailable.

### Data and experiment decisions

- Evaluation data will be versioned and reproducible.
- The evaluation set will include short and long documents, tables or structured content where extractable, exact-number questions, exact-identifier questions, paraphrased questions, vocabulary-mismatch questions, and unanswerable questions.
- A question may have multiple relevant passages; evaluation must support multiple relevance labels rather than assuming exactly one correct chunk.
- Raw retrieval results, ranks, scores, configurations, latency, and run identifiers will be retained for diagnosis.
- Changes to extraction, chunking, or embedding configuration must be versioned and must support controlled re-indexing.
- Page and section metadata must be preserved through ingestion, retrieval, API responses, and citations.

### Scope constraints

- Prefer PostgreSQL capabilities already present in the project: pgvector for dense retrieval and PostgreSQL full-text search for lexical retrieval.
- Do not introduce a separate search engine unless PostgreSQL is empirically inadequate for the learning objectives.
- Do not introduce a large RAG framework that obscures retrieval, fusion, reranking, or evaluation mechanics.
- Do not add OCR or advanced layout infrastructure unless evaluation shows that scanned/image-based documents are a required failure case.
- Keep the application monolithic and understandable; microservices and distributed processing are outside the completion scope.
- Model, reranker, and library choices will be made at the relevant phase and recorded with their cost, latency, and learning tradeoffs.

## Phase 1 — Reproducible retrieval evaluation

### Purpose

Create a trustworthy baseline before modifying retrieval. The checked-in `eval_retrieval.py` currently prints selected chunks and does not calculate Hit@K or MRR, so the previously reported metrics must be reconstructed as a reproducible repository-owned evaluation.

### Tasks

- [x] Document that the original questions and calculation inputs behind the reported metrics were unavailable, then establish a fresh human-labelled 91-page baseline without fabricating the missing inputs.
- [x] Define a versioned, human-readable evaluation dataset format, initially JSON or JSONL.
- [x] Represent at least:
  - [x] Stable question ID.
  - [x] Document identity or filename.
  - [x] Question text.
  - [x] Answerable/unanswerable label.
  - [x] One or more relevant chunk IDs and/or page/evidence labels.
  - [x] Optional reference answer for later answer evaluation.
  - [x] Notes explaining ambiguous or multiple-valid-evidence cases.
- [x] Create a small but varied labelled baseline dataset.
- [x] Include the 91-page `Gov-AR_1 test.pdf` report; the developer explicitly waived the two-page contract comparison on 2026-08-27 because the long-document stress test is sufficient for this project's Phase 1 scope.
- [x] Include semantic/paraphrased, exact-term, name, acronym, date, number, clause/section identifier, and unanswerable questions.
- [x] Implement a retrieval-only evaluation runner that does not call answer generation.
- [x] Calculate Hit@1, Hit@3, Hit@5, Hit@10, MRR, and Recall@K.
- [x] Add nDCG@K when graded or multiple relevance labels make it useful.
- [x] Define metric behavior clearly for questions with multiple relevant chunks and unanswerable questions.
- [x] Produce per-question diagnostics containing expected evidence, retrieved chunk IDs, ranks, raw scores/distances, and text previews.
- [x] Capture aggregate metrics, configuration, latency, timestamp, and a run identifier.
- [x] Save machine-readable and human-readable evaluation reports.
- [x] Make the baseline runnable with one documented command.
- [x] Walk through the metric formulas and implementation line by line.

### Acceptance criteria

- [x] One command reproduces a dense-only baseline from the current Lexis index.
- [x] The reported 75% Hit@5 and 0.494 MRR result is reproduced, or differences are explained by documented changes in questions, labels, index state, or metric definitions.
- [x] Record the developer's decision to remove reproduction of the two-page contract result from Phase 1 scope and accept the reproducible 91-page baseline as sufficient.
- [x] Every aggregate metric can be traced to individual question results.
- [x] Retrieval evaluation runs without invoking the answer model.
- [x] The dataset and report formats support multiple relevant passages and unanswerable questions.
- [x] No retrieval behavior has been changed during baseline capture.

### Dependencies

- Uses the existing database, embeddings, and dense retrieval implementation.
- Must be completed before phases that alter extraction, chunking, retrieval, fusion, reranking, or query expansion.

## Phase 2 — Better PDF extraction and cleaning

### Purpose

Preserve document boundaries and reduce extraction noise before improving retrieval. The current pipeline concatenates every page into one string, losing page provenance and allowing headers, footers, broken lines, and layout artifacts to influence chunks.

### Tasks

- [x] Extract and process PDF content page by page.
- [x] Preserve original page numbers through the ingestion pipeline.
- [x] Normalize whitespace without destroying meaningful paragraph or list boundaries.
- [x] Repair unnecessarily broken lines where this can be done safely.
- [x] Detect and remove repeated headers and footers conservatively.
- [x] Handle empty or low-text pages explicitly.
- [x] Record extraction warnings and diagnostics per page.
- [x] Inspect representative output from the 91-page PDF; the two-page contract remains explicitly waived by the developer and was not reintroduced in Phase 2.
- [x] Identify multi-column, table, or reading-order failures in the actual evaluation corpus.
- [x] Avoid OCR or complex layout services because visual inspection confirmed that flagged empty pages are genuinely blank and the corpus does not require OCR.
- [x] Add focused tests for cleaning functions and page provenance.
- [x] Re-run the fixed Phase 1 questions against the newly labelled Phase 2 index as a distinct experiment.

### Acceptance criteria

- [x] Every processed passage can be traced to its source page or page range.
- [x] Repeated page furniture no longer creates or dominates retrieved chunks in the tested 91-page document.
- [x] Cleaning does not silently merge unrelated sections or remove meaningful content in inspected samples.
- [x] Extraction warnings make empty or problematic pages visible.
- [x] Before/after retrieval metrics and representative chunk diffs are recorded.

### Dependencies

- Requires Phase 1 so effects can be measured.
- Supplies clean, page-aware input to Phases 3 and 4.

## Phase 3 — Token-aware, structure-aware chunking

### Purpose

Replace fixed character-only chunking with retrievable passages that preserve evidence and document structure. The current 1,000-character chunks with 200-character overlap may separate headings from content, split evidence, and create overlapping near-duplicates.

### Tasks

- [x] Choose and document the tokenizer used to measure chunk size.
- [x] Replace character-size limits with configurable token-size and token-overlap settings.
- [x] Preserve page start/end metadata for each chunk.
- [x] Detect and retain basic section or heading context where reliably available.
- [x] Prefer boundaries in this order where appropriate: sections, paragraphs, sentences, lists, then a safe hard boundary.
- [x] Prevent empty, extremely small, or non-progressing chunks.
- [x] Calculate a stable content hash for chunk identity and duplicate detection.
- [x] Record token count and chunker version on every chunk.
- [x] Create a chunk inspection command/report displaying chunk text, page range, heading, token count, and overlap.
- [x] Compare several bounded chunk size/overlap configurations using the Phase 1 dataset.
- [x] Measure near-duplicate retrieval caused by overlap.
- [x] Decide empirically whether parent-child chunking is warranted.
- [x] Defer parent-child retrieval: all bounded child configurations preserve 13/13 evidence cases and the selected configuration reaches 13/13 Hit@5; reconsider only with answer-context evidence.
- [x] Keep the parent-child decision and any future behavior inspectable rather than hiding it behind a framework.
- [x] Add unit tests for boundaries, overlaps, page transitions, tiny inputs, and empty inputs.

### Acceptance criteria

- [x] Important labelled evidence remains intact in at least one retrievable chunk.
- [x] Every chunk includes useful page, token, identity, and version metadata.
- [x] Chunking always makes progress and does not produce accidental empty chunks.
- [x] Selected chunk settings are justified with evaluation results, not intuition alone.
- [x] Parent-child chunking is either implemented with measured justification or explicitly deferred with evidence.
- [x] The new chunker and its tradeoffs have been reviewed line by line.

### Dependencies

- Requires clean page-aware extraction from Phase 2.
- Requires Phase 1 metrics for comparing configurations.
- Defines metadata needed by Phase 4 and later citations.

## Phase 4 — Improve the database model

### Purpose

Provide stable document identity, provenance, versioning, lexical-search support, and controlled re-indexing. The current single `document_chunks` table identifies documents only by filename and stores too little metadata for advanced retrieval.

### Tasks

- [x] Add a `documents` table containing stable document ID, original filename, content hash, page count, chunk count, processing status, and timestamps.
- [x] Extend chunk storage with:
  - [x] `document_id` foreign key.
  - [x] `page_start` and `page_end`.
  - [x] `section_title` or equivalent nullable heading metadata.
  - [x] `content_hash`.
  - [x] `token_count`.
  - [x] `chunking_version`.
  - [x] `embedding_model` and/or embedding version.
  - [x] Do not add `parent_chunk_id`: Phase 3 empirically deferred parent-child chunking, and the decision is documented.
  - [x] Lexical search representation required by Phase 6.
- [x] Add appropriate foreign keys, uniqueness constraints, and indexes.
- [x] Replace filename-only duplicate handling with a documented document/content identity policy.
- [x] Define behavior for re-uploading the same filename with different content.
- [x] Add a controlled schema migration approach rather than relying only on `create_all`.
- [x] Add a controlled re-indexing command for extraction/chunking/embedding version changes.
- [x] Ensure failed ingestion remains transactional and does not leave partially indexed documents.
- [x] Preserve the ability to connect evaluation labels to stable document/chunk identities.
- [x] Document the schema and lifecycle states.

### Acceptance criteria

- [x] Documents and chunks have stable, traceable identities independent of transient frontend state.
- [x] Page and section provenance survive storage and retrieval.
- [x] Duplicate/re-upload behavior is deterministic and documented.
- [x] Schema upgrades and re-indexing can be performed in a controlled, repeatable way.
- [x] Failed processing does not expose partial indexes as successfully processed documents.
- [x] The schema supports both pgvector and PostgreSQL lexical retrieval.

### Dependencies

- Depends on metadata decisions from Phases 2 and 3.
- Must support Phase 6 lexical search and Phase 10 citations.
- Can be designed alongside Phase 3 but should be migrated after chunk metadata is settled.

## Phase 5 — Strengthen dense retrieval

### Purpose

Establish the strongest understandable dense-only baseline before hybrid retrieval, while making scores, candidate selection, duplicate handling, and relevance visible.

### Tasks

- [x] Separate candidate retrieval from final context selection.
- [x] Return raw distance/similarity values with every dense candidate.
- [x] Make candidate count and final `top_k` configurable.
- [x] Compare L2 and cosine ranking on the evaluation set.
- [x] Verify assumptions about embedding normalization rather than relying on them.
- [x] Test bounded candidate-pool and top-k configurations.
- [x] Add overlap-aware near-duplicate suppression instead of exact-content-only deduplication.
- [x] Evaluate heading-enriched embedding text versus content-only embedding text.
- [x] Add optional adjacent-chunk expansion and measure when it helps or adds noise.
- [x] Investigate and calibrate a dense relevance threshold, including behavior for unanswerable questions; record that no safe nonzero threshold is supported by the current labels.
- [x] Add a pgvector similarity index when corpus size and query plans justify it; Phase 5 plans showed that HNSW was not used at 929 chunks, so the temporary test index was removed and a permanent index was explicitly deferred.
- [x] Inspect database query plans before and after temporary vector indexing.
- [x] Record accuracy, latency, candidate count, and configuration for each experiment.
- [x] Select and document the strongest dense-only configuration.

### Acceptance criteria

- [x] Every dense result exposes an interpretable rank and raw score/distance.
- [x] Candidate retrieval is independent from final context construction.
- [x] The chosen distance function, chunk settings, candidate count, and top-k are justified by evaluation.
- [x] Near-duplicate chunks no longer consume a disproportionate share of final results in tested cases.
- [x] Unanswerable-query behavior is measured rather than inferred.
- [x] A named dense-only baseline is saved for comparison with lexical, hybrid, reranked, and expanded variants.

### Dependencies

- Requires Phases 1–4.
- Produces the dense candidate list consumed by Phase 7.

## Phase 6 — Lexical retrieval

### Purpose

Add an independently testable lexical retriever for exact terms and identifiers that semantic embeddings may under-rank, using PostgreSQL full-text search where suitable.

### Tasks

- [ ] Define and populate a PostgreSQL text-search representation for chunks.
- [ ] Choose and document text-search configuration, tokenization, normalization, and weighting behavior.
- [ ] Implement query parsing suitable for user questions without making unsafe raw SQL expressions.
- [ ] Implement lexical candidate retrieval independently of vector search.
- [ ] Return lexical ranks and scores.
- [ ] Test names, numbers, dates, acronyms, clause identifiers, exact phrases, rare terms, and common-word-heavy questions.
- [ ] Document limitations of PostgreSQL full-text search, especially punctuation-heavy identifiers and exact phrase behavior.
- [ ] Add explicit fallback or supplementary matching only where tests justify it.
- [ ] Add the required lexical indexes and inspect query plans.
- [ ] Evaluate lexical retrieval independently using the Phase 1 dataset.
- [ ] Compare category-level dense and lexical performance.

### Acceptance criteria

- [ ] Lexical search can run and be evaluated without dense search.
- [ ] Every result includes lexical score and rank diagnostics.
- [ ] The implementation improves or clearly complements dense retrieval on exact-signal question categories.
- [ ] Known lexical limitations are documented with examples.
- [ ] Dense and lexical metrics are stored as separate named baselines.

### Dependencies

- Requires the Phase 4 schema and indexes.
- Requires Phase 1 evaluation categories.
- Produces the lexical candidate list consumed by Phase 7.

## Phase 7 — Hybrid search with Reciprocal Rank Fusion

### Purpose

Combine the complementary recall of dense and lexical retrieval without directly averaging incompatible score scales.

### Tasks

- [ ] Retrieve configurable dense and lexical candidate pools independently.
- [ ] Implement Reciprocal Rank Fusion explicitly using rank positions and a configurable RRF constant.
- [ ] Deduplicate fused results by stable chunk identity.
- [ ] Define behavior for chunks present in only one candidate list.
- [ ] Preserve dense and lexical raw ranks/scores alongside the fused score.
- [ ] Expose each retriever's RRF contribution and final fused rank.
- [ ] Test candidate-pool sizes and RRF constant values in a bounded experiment.
- [ ] Compare union recall before fusion ranking with final hybrid Hit@K/MRR/Recall/nDCG.
- [ ] Analyze questions helped, unchanged, and harmed by fusion.
- [ ] Retain standalone dense and lexical modes for debugging and ablation.
- [ ] Document the RRF formula and walk through example calculations by hand and in code.

### Acceptance criteria

- [ ] A result can be traced from its original dense/lexical ranks to its exact fused score.
- [ ] Hybrid evaluation is reproducible and directly comparable with named dense-only and lexical-only baselines.
- [ ] Hybrid retrieval meaningfully improves aggregate or targeted-category performance, or its lack of improvement is diagnosed before proceeding.
- [ ] Retrieval modes remain independently callable for ablation testing.
- [ ] No opaque framework hides rank fusion behavior.

### Dependencies

- Requires Phase 5 dense retrieval and Phase 6 lexical retrieval.
- Produces the broad candidate list for Phase 8 reranking.

## Phase 8 — Reranking

### Purpose

Improve precision and ordering by evaluating the question and each fused passage together. Reranking must not be treated as a substitute for candidate recall.

### Tasks

- [ ] Select a reranking approach after comparing quality, cost, latency, local/hosted operation, and educational clarity.
- [ ] Keep reranking in a separate function/module from candidate retrieval.
- [ ] Construct explicit question-passage pairs.
- [ ] Rerank a configurable number of fused candidates.
- [ ] Return reranker scores, original fused ranks, and final ranks.
- [ ] Select a configurable number of final evidence chunks.
- [ ] Calibrate a reranker relevance cutoff, especially for unanswerable questions.
- [ ] Measure candidate recall before reranking.
- [ ] Measure MRR, Hit@K, Recall@K, and nDCG after reranking.
- [ ] Record latency and monetary/computational cost.
- [ ] Inspect examples promoted and demoted by the reranker.
- [ ] Add fallback behavior for reranker failure without silently changing retrieval semantics.
- [ ] Compare hybrid-without-reranking and hybrid-with-reranking as separate named runs.

### Acceptance criteria

- [ ] The correct evidence is measured for presence in the candidate pool before judging reranker quality.
- [ ] Every reranked result exposes its input rank, output rank, and reranker score.
- [ ] Reranking provides measurable benefit or a documented reason not to keep it.
- [ ] Relevance cutoff behavior is tested on answerable and unanswerable questions.
- [ ] Added latency and cost are measured and considered acceptable for the project.

### Dependencies

- Requires the fused candidate list from Phase 7.
- Supplies ordered, relevance-scored evidence to Phases 9 and 10.

## Phase 9 — Safe query expansion and rewriting

### Purpose

Reduce vocabulary mismatch and improve ambiguous retrieval without allowing a rewrite to erase exact terms or change the user's intent.

### Tasks

- [ ] Define when query expansion is used and when the original query alone is sufficient.
- [ ] Always preserve and search the original user question.
- [ ] Generate one controlled retrieval-oriented alternative query initially.
- [ ] Preserve exact names, numbers, dates, acronyms, and identifiers.
- [ ] Optionally extract important lexical terms/entities as an inspectable signal.
- [ ] Add validation or fallback when an expansion is empty, malformed, or meaning-shifted.
- [ ] Expose original and expanded queries in diagnostics.
- [ ] Retrieve candidates for each query through the appropriate dense/lexical paths.
- [ ] Fuse multi-query results without allowing duplicate passages to dominate.
- [ ] Evaluate expansion enabled versus disabled on the same dataset.
- [ ] Analyze vocabulary-mismatch improvements and precise-query regressions.
- [ ] Defer conversational standalone-question rewriting until backend conversation context exists and is explicitly required.

### Acceptance criteria

- [ ] The original query is never discarded.
- [ ] Every generated expansion is visible in evaluation/debug output.
- [ ] Exact entities and identifiers are preserved in tested cases.
- [ ] Query expansion measurably improves difficult query categories without unacceptable regression elsewhere.
- [ ] Failures fall back safely to the original-query pipeline.

### Dependencies

- Follows stable hybrid retrieval and reranking so its incremental effect can be isolated.
- Uses Phase 1 evaluation categories and Phase 7 fusion mechanics.
- Conversational rewriting depends on future persistent conversation state and is not required for initial completion.

## Phase 10 — Grounded answer generation and citations

### Purpose

Generate answers only from selected, sufficiently relevant evidence and make the support traceable to document pages and sections.

### Tasks

- [ ] Separate candidate retrieval, final evidence selection, context construction, and answer generation.
- [ ] Build structured context containing stable chunk ID, document identity, filename, page/page range, section title, and passage text.
- [ ] Order context to preserve relevance while using document order where it improves coherence.
- [ ] Prevent duplicate or overlapping context from wasting the model's context window.
- [ ] Apply the calibrated relevance/refusal decision before answer generation.
- [ ] Strengthen grounding instructions so answers use only provided evidence.
- [ ] Instruct the model to cite factual claims using the supplied page-aware source labels.
- [ ] Define and validate a machine-readable answer/citation response structure where appropriate.
- [ ] Return page-aware citations through the API.
- [ ] Improve the insufficient-evidence response.
- [ ] Add protection against instructions embedded inside uploaded document content.
- [ ] Ensure source previews correspond to evidence actually used/sent, not merely every broad candidate.
- [ ] Test answers spanning adjacent chunks or pages.
- [ ] Test unanswerable, partially answerable, conflicting-evidence, and prompt-injection cases.

### Acceptance criteria

- [ ] Every citation maps to stored evidence and correct page metadata.
- [ ] The API distinguishes broad retrieval candidates from evidence selected for generation.
- [ ] Weak-evidence queries refuse or qualify appropriately instead of confidently answering from noise.
- [ ] Generated factual claims are supported by supplied context in the labelled test cases.
- [ ] Instructions found inside documents do not override system grounding behavior in tested cases.

### Dependencies

- Requires page metadata from Phases 2–4.
- Requires relevance-scored final evidence from Phases 8–9.
- Supplies outputs evaluated in Phase 11 and displayed in Phase 12.

## Phase 11 — Answer evaluation

### Purpose

Measure generation independently from retrieval so that good retrieval is not confused with a good answer, and a wrong answer can be attributed correctly.

### Tasks

- [ ] Extend the evaluation dataset with reference answers and claim/evidence expectations for a bounded subset.
- [ ] Define human-readable rubrics for correctness, faithfulness, citation accuracy, completeness, and refusal accuracy.
- [ ] Measure unsupported-claim rate.
- [ ] Evaluate answerable, unanswerable, partially answerable, and conflicting-evidence cases.
- [ ] Record the exact retrieved evidence alongside every evaluated answer.
- [ ] Start with human-labelled checks for the core dataset.
- [ ] If model-assisted grading is added, treat it as a supporting signal rather than unquestioned ground truth.
- [ ] Version judge prompts/models and preserve raw grader outputs.
- [ ] Report retrieval success and answer success separately.
- [ ] Classify failures by extraction, chunking, candidate retrieval, fusion, reranking, evidence selection, or generation.

### Acceptance criteria

- [ ] A report can distinguish retrieval failure from generation failure for every evaluated question.
- [ ] Correctness, faithfulness, citations, completeness, refusal accuracy, and unsupported claims are reported separately.
- [ ] Core acceptance decisions are not based solely on an unvalidated model judge.
- [ ] Evaluation inputs, outputs, configurations, and grader versions are reproducible.

### Dependencies

- Requires grounded, citation-bearing answers from Phase 10.
- Reuses the versioned dataset and reporting infrastructure from Phase 1.

## Phase 12 — API and frontend integration

### Purpose

Expose the advanced pipeline accurately and make retrieval behavior inspectable without allowing frontend-only state to misrepresent the database.

### Tasks

#### Backend/API

- [ ] Add a persistent document-list endpoint.
- [ ] Add a document-delete endpoint with clear database and UI behavior.
- [ ] Expose document processing/indexing status.
- [ ] Use stable document IDs rather than filename-only selection.
- [ ] Add a retrieval debug mode or dedicated diagnostics endpoint.
- [ ] Expose configured retrieval mode: dense, lexical, hybrid, hybrid plus reranking, and expansion where useful for learning/ablation.
- [ ] Return page-aware selected evidence and citations.
- [ ] Centralize configurable API, model, chunking, retrieval, and threshold settings.
- [ ] Add better filename/content-type/file-size validation and safer error messages.
- [ ] Define behavior for scanned/image-only PDFs clearly.
- [ ] Keep rate limiting and API-key protection, but make configuration explicit.

#### Frontend

- [ ] Restore indexed documents from the backend after refresh.
- [ ] Fix the query counter so successful queries increment it.
- [ ] Make the Documents table's `Query →` action select the document and navigate to Query.
- [ ] Prevent duplicate document rows after re-upload.
- [ ] Validate the API key against the backend rather than treating any non-empty value as connected.
- [ ] Display processing, failure, and indexed states accurately.
- [ ] Display page-aware citations and selected evidence.
- [ ] Add an optional learning/debug panel showing dense rank, lexical rank, fused rank, reranker score, relevance decision, and query expansions.
- [ ] Show the active retrieval mode/configuration in debug mode.
- [ ] Avoid exposing secrets in logs, state diagnostics, or error messages.

### Acceptance criteria

- [ ] Refreshing the frontend does not lose the list of indexed documents.
- [ ] Re-upload, deletion, navigation, query count, and API-key state behave consistently with backend state.
- [ ] Users can trace a displayed answer to page-aware evidence.
- [ ] Debug mode explains the retrieval path without being required for normal use.
- [ ] API validation and errors are safe and actionable.

### Dependencies

- Document lifecycle endpoints depend on the Phase 4 data model.
- Debug output depends on diagnostics produced in Phases 5–9.
- Citation UI depends on Phase 10.
- This phase follows core RAG work so UI changes do not obscure retrieval experiments.

## Phase 13 — Final cleanup, regression testing, and documentation

### Purpose

Make Lexis reproducible, understandable, and complete as a learning artifact, then stop expanding it and move to the larger project.

### Tasks

- [ ] Add a pinned or suitably constrained Python dependency manifest.
- [ ] Add an `.env.example` containing variable names and safe placeholders only.
- [ ] Centralize settings and remove unnecessary hard-coded URLs, model names, chunk sizes, top-k values, and thresholds.
- [ ] Document local database, backend, frontend, migration, ingestion, re-indexing, evaluation, and test commands.
- [ ] Add focused unit tests for cleaning, chunking, metrics, RRF, deduplication, thresholds, and citation mapping.
- [ ] Add integration tests for document upload/indexing, listing, retrieval, querying, and deletion.
- [ ] Add regression tests for the labelled evaluation corpus.
- [ ] Review transaction handling and failure cleanup.
- [ ] Review CORS, API-key behavior, file limits, rate limits, and secret handling.
- [ ] Remove or archive obsolete code such as the unused `Sidebar.jsx` after confirming it has no required behavior.
- [ ] Remove unused Vite/React template assets and styles after confirming they are unreferenced.
- [ ] Replace placeholder documentation with architecture and learning documentation.
- [ ] Document the final end-to-end data flow and each retrieval stage.
- [ ] Produce an ablation comparison at minimum for:
  - [ ] Original baseline.
  - [ ] Improved extraction/chunking plus dense retrieval.
  - [ ] Dense-only final configuration.
  - [ ] Lexical-only retrieval.
  - [ ] Hybrid RRF.
  - [ ] Hybrid plus reranking.
  - [ ] Hybrid plus reranking plus query expansion.
- [ ] Report accuracy, category-level behavior, latency, and cost where applicable.
- [ ] Record remaining known limitations and explicitly close the project scope.

### Acceptance criteria

- [ ] A new developer can set up, index, query, evaluate, and understand Lexis from repository documentation.
- [ ] Tests and evaluation run through documented commands.
- [ ] Final metrics are compared with the reproducible 91-page Phase 1 baseline; the historical two-page figures may be mentioned for context but are not a required regression target because their labelled inputs are unavailable and the developer waived reproducing that test.
- [ ] The final system includes dense retrieval, lexical retrieval, RRF, reranking, safe query expansion, relevance/refusal handling, grounded answers, and page-aware citations.
- [ ] Retrieval and answer evaluation remain separate and reproducible.
- [ ] Dead code and template leftovers are removed without changing required behavior.
- [ ] Known limitations, deferred items, and the stopping point are documented.

### Dependencies

- Depends on the completed behavior and decisions of Phases 1–12.
- Regression evaluation uses the Phase 1 harness and Phase 11 answer evaluation.

## Dependencies between phases

The intended implementation order is sequential because each stage provides measurements or data required by the next:

```text
Phase 1: Retrieval evaluation baseline
   ↓
Phase 2: PDF extraction and cleaning
   ↓
Phase 3: Token/structure-aware chunking
   ↓
Phase 4: Persistent schema and versioning
   ↓
Phase 5: Strong dense-only baseline
   ↓
Phase 6: Independent lexical baseline
   ↓
Phase 7: Hybrid RRF
   ↓
Phase 8: Reranking
   ↓
Phase 9: Query expansion
   ↓
Phase 10: Grounded generation and citations
   ↓
Phase 11: Answer evaluation
   ↓
Phase 12: API/frontend integration
   ↓
Phase 13: Cleanup, regression, documentation, and closure
```

Important dependency rules:

- [ ] Do not change retrieval before capturing the Phase 1 baseline.
- [ ] Do not judge retrieval improvements without rerunning the same versioned evaluation set.
- [ ] Do not add hybrid fusion until dense and lexical retrieval can be inspected independently.
- [ ] Do not judge reranking without first measuring whether relevant evidence exists in its candidate pool.
- [ ] Do not add query expansion before hybrid/reranked retrieval is stable enough for an ablation comparison.
- [ ] Do not evaluate answer quality as a proxy for retrieval quality.
- [ ] Do not build citation UI until page-aware citation data exists end to end.
- [ ] Schema work may be designed alongside extraction/chunking but must reflect their settled metadata requirements.

## Deferred/stretch items

The following are deliberately outside the required 13-phase completion scope. They may be considered in a larger future project or only if empirical evaluation proves one is essential.

### Deferred document capabilities

- [ ] OCR for scanned/image-only PDFs.
- [ ] Advanced multi-column/layout reconstruction.
- [ ] Multimodal retrieval over figures, charts, and images.
- [ ] Specialized table extraction and table-aware retrieval.
- [ ] Support for DOCX, HTML, email, audio, or other document types.

### Deferred retrieval capabilities

- [ ] Knowledge-graph or graph-based RAG.
- [ ] Agentic/multi-step retrieval.
- [ ] Iterative self-querying or autonomous retrieval loops.
- [ ] Fine-tuned embedding or reranking models.
- [ ] Multiple vector databases or a separate search cluster.
- [ ] Learned sparse retrieval beyond the agreed PostgreSQL lexical baseline.
- [ ] HyDE or other generated-document retrieval methods.
- [ ] Full conversational question rewriting and persistent chat memory.

### Deferred infrastructure capabilities

- [ ] Microservice decomposition.
- [ ] Distributed task queues and worker fleets.
- [ ] Multi-tenant authorization and billing.
- [ ] Production cloud deployment, autoscaling, and high availability.
- [ ] Large-scale observability infrastructure.
- [ ] Comprehensive administration console.

### Conditional stretch decisions

- Parent-child chunking remains conditional on Phase 3 measurements.
- OCR/layout tooling remains conditional on demonstrated corpus requirements.
- Model-assisted answer judging remains secondary to validated human labels.
- A separate lexical/search service remains conditional on PostgreSQL proving inadequate.
- Conversational rewriting remains conditional on adding persistent backend conversation context.

## Definition of project completion

Lexis is complete when all required phase acceptance criteria are satisfied, the final ablation report demonstrates what each major retrieval addition changed, the system can explain and cite its selected evidence, and the repository teaches the entire advanced-RAG flow without relying on opaque orchestration. At that point, remaining stretch items are documented rather than pursued automatically, allowing work to move to the larger project.
