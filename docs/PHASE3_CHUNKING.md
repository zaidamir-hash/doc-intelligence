# Phase 3: token-aware, structure-aware chunking

## Outcome

Phase 3 replaces the old 1,000-character/200-character-overlap splitter with a
transparent token-aware chunker. The selected production configuration is:

- tokenizer encoding: `cl100k_base`
- maximum rendered chunk size: 200 tokens
- requested overlap: 30 tokens
- minimum-size target: 40 tokens
- chunker version: `structure-token-v1`

The 91-page evaluation PDF produces 424 chunks. Every one is non-empty, no
chunk exceeds 200 tokens, and all 424 stable content hashes are unique.

## Why tokens instead of characters

Embedding models receive tokens, not Python characters or words. A character
limit is only an indirect guess at model input size: punctuation, numbers,
whitespace, and different words can consume different token counts.

Lexis uses `tiktoken` 0.14.0 and explicitly requests `cl100k_base`. A local
check with `tiktoken.encoding_for_model("text-embedding-3-small")` resolves to
the same encoding. OpenAI documents that embedding inputs are measured in
tokens and that `text-embedding-3-small` accepts at most 8,192 input tokens.
The explicit encoding name keeps the chunker reproducible instead of depending
on an implicit model lookup at runtime.

For a new environment, install the tokenizer with:

```powershell
.venv\Scripts\python.exe -m pip install tiktoken==0.14.0
```

A complete dependency manifest is deliberately deferred to Phase 13, as
specified in the project plan.

## Chunking pipeline

The implementation is in `chunking.py` and uses small immutable data classes so
each intermediate value can be inspected.

1. `extract_structural_units` reads the clean, page-aware output from Phase 2.
2. `_looks_like_heading` recognizes conservative numbered, uppercase, and
   title-style headings and carries the current section title into later text.
3. Paragraphs and list items become `StructuralUnit` values with page and
   section metadata.
4. `_split_oversized_unit` splits an oversized paragraph at sentence boundaries.
   If one sentence is still too large, a binary search finds the largest safe
   token boundary.
5. `_pack_base_chunks` groups compatible units while keeping section boundaries
   intact and reserving room for overlap.
6. A small final chunk is rebalanced from its predecessor when that can be done
   without crossing a section or violating either size limit.
7. `_with_overlap` copies up to 30 tokens from the preceding chunk only within
   the same section. It reduces the overlap when necessary to stay within the
   200-token maximum.
8. `_render_chunk` creates embedding text containing only source-page context,
   optional section context, and the passage. Operational metadata is not added
   to embedding text because a random-looking hash would add semantic noise.
9. `chunk_extraction` enforces non-empty output and the maximum-token invariant,
   and removes exact duplicate identities.

The effective boundary priority is section, paragraph/list unit, sentence, and
finally a hard token boundary. Page boundaries remain metadata rather than an
absolute split: related text may form a cross-page chunk, whose `page_start` and
`page_end` preserve the complete provenance range.

## Chunk metadata and identity

Every `StructuredChunk` contains:

- `content` — the text sent to embeddings and later retrieval;
- `passage_text` — the source passage without the rendered provenance prefix;
- `page_start` and `page_end`;
- nullable `section_title`;
- exact rendered `token_count`;
- actual `overlap_token_count` (which can be below the requested maximum);
- SHA-256 `content_hash`;
- `chunking_version`; and
- `boundary_kinds` explaining how the text was formed.

The stable hash covers normalized serialization of passage text, page range,
section title, and chunker version. The same input and version therefore create
the same identity. It is metadata, not embedding content.

The current Phase 2 database table can store only filename, index, content, and
embedding. The full metadata exists throughout Phase 3 chunking and inspection,
but first-class persistence requires the schema work explicitly planned for
Phase 4. The page/section labels included in `content` preserve visible
provenance in the current schema until then.

## Inspection and evidence checks

Generate the selected configuration's complete JSON and Markdown reports with:

```powershell
.venv\Scripts\python.exe inspect_chunks.py "C:\Users\MA\Downloads\Gov-AR_1 test.pdf"
```

The generated reports are:

- `evaluation/reports/chunks-structure-token-v1-200-30.json`
- `evaluation/reports/chunks-structure-token-v1-200-30.md`

They show every chunk's full text, page range, heading, token count, actual
overlap, boundary types, version, and hash.

`evaluation/datasets/phase3_evidence.json` fixes page-and-pattern evidence
requirements for the 13 answerable Phase 1 questions. These requirements were
defined before seeing retrieval ranks. A chunk counts as preserving evidence
only when its page range intersects an accepted page and one complete regex
pattern group is present in the same passage.

All three bounded configurations preserved all 13 evidence cases:

| Configuration | Max/overlap | Chunks | Mean tokens | Evidence preserved | Adjacent near-duplicates | Cross-page chunks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 200/30 | 424 | 156.60 | 13/13 | 1 | 43 |
| balanced | 300/50 | 307 | 213.77 | 13/13 | 1 | 42 |
| large | 400/70 | 240 | 269.07 | 13/13 | 1 | 44 |

Exact duplicate hashes were zero for every configuration. A near-duplicate is
defined here as an adjacent chunk pair whose passage token-set Jaccard similarity
is at least 0.8. This is a simple, reproducible overlap diagnostic—not a claim
that non-adjacent semantic duplicates cannot exist. The selected 200/30 output
has one such adjacent pair, mean actual overlap 19.48 tokens, and maximum actual
overlap 30 tokens.

## Retrieval comparison and selection

Run the exact dense comparison with:

```powershell
.venv\Scripts\python.exe evaluate_chunk_retrieval.py "C:\Users\MA\Downloads\Gov-AR_1 test.pdf"
```

It embeds every generated chunk, performs exact in-memory L2 ranking, generates
configuration-specific labels from the fixed evidence rules before ranking, and
writes `evaluation/datasets/phase3.json` for the selected index. The selection
rule was declared in code before the run: maximize Hit@5, then MRR, then nDCG@5,
and prefer fewer maximum tokens only if those retrieval metrics tie.

| Configuration | Hit@1 | Hit@5 | Hit@10 | MRR | Recall@10 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 200/30 | 0.2308 | **1.0000** | **1.0000** | 0.5179 | **0.9615** | 0.6246 |
| 300/50 | 0.3077 | 0.9231 | 0.9231 | **0.5962** | 0.8846 | **0.6441** |
| 400/70 | 0.3077 | 0.9231 | 0.9231 | 0.5795 | 0.8590 | 0.5888 |

The selected 200/30 configuration prioritizes reliable evidence coverage in the
five chunks supplied to a later RAG stage. It does not dominate every metric:
300/50 ranks the first relevant result earlier on average. That distinction is
why both Hit@5 and MRR remain visible.

The database-backed pgvector verification run `9475274a22ba` exactly matched
the selected in-memory result: Hit@5 1.0000 and MRR 0.5179.

Compared with the Phase 2 character-chunk run:

| Index | Hit@1 | Hit@5 | Hit@10 | MRR | Recall@10 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 2 character chunks | 0.5385 | 0.9231 | 0.9231 | 0.7179 | 0.8526 | 0.6480 |
| Phase 3 token/structure chunks | 0.2308 | 1.0000 | 1.0000 | 0.5179 | 0.9615 | 0.6246 |

Phase 3 improved top-five coverage from 12/13 to 13/13 and recovered a larger
share of all labelled evidence by rank 10. It reduced first-result precision and
MRR. This is not hidden: chunking has made evidence more retrievable within the
candidate set, while later dense-search tuning, hybrid retrieval, and reranking
are still needed to order the best evidence first.

The comparison uses the fixed Phase 1 test questions because the project plan
explicitly requires it. Repeatedly tuning future retrieval decisions on this
same small test set would overfit it; future parameter work should add a separate
development/validation set and retain these 13 questions as a regression check.

## Parent-child decision

Parent-child chunking is explicitly deferred, not forgotten.

- Every tested child-size configuration preserved all 13 labelled evidence cases.
- The selected child chunks retrieved evidence for all 13 cases within five results.
- Parent expansion mainly changes how much surrounding text is sent to answer
  generation; Phase 3 evaluates retrieval, not generated-answer quality.
- Adding parents now would introduce additional identities, storage, expansion
  policy, and evaluation complexity without measured evidence that it solves a
  current Phase 3 failure.

If later answer evaluation shows that a precisely retrieved 200-token child lacks
enough surrounding context, parent expansion can be reconsidered with a measured
before/after experiment. If adopted, it must remain explicit and inspectable.

## Known limitations before Phase 4

- Heading detection is intentionally heuristic. It can misclassify unusual
  front matter, table labels, or title-like prose; the inspection report exposes
  these cases.
- `min_chunk_tokens` is a target, not a destructive rule. An unavoidable final
  section or tiny document is retained rather than discarded; the evaluation
  PDF's smallest rendered chunk is 28 tokens.
- Full structured metadata is not yet persisted in separate database columns.
- The 13-question test set is small and from one long PDF.
- Remote embedding calls can show slight numeric/ranking variation over time.
  Two final runs selected the same configuration and produced identical metrics
  for 200/30, while one non-selected configuration's MRR varied slightly.

These limitations are inputs to later planned phases, not reasons to start them
inside Phase 3.
