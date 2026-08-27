# Phase 2 PDF extraction and cleaning

## Scope

Phase 2 changes document representation before retrieval. It does not introduce
token-aware chunking, database schema migrations, lexical search, reranking, or
query expansion.

The two-page contract was explicitly waived before Phase 2. Inspection and
evaluation therefore use only the 91-page `Gov-AR_1 test.pdf` report.

## Before and after

The original upload path concatenated every extracted page into one string and
then character-chunked that combined string. A chunk could cross a PDF page,
contained no page metadata, and could include control characters plus repeated
page furniture.

The Phase 2 path is:

```text
PDF bytes
  -> extract every page independently
  -> detect repeated header/footer candidates across page edges
  -> clean each page independently
  -> retain page-level warnings and removed-furniture diagnostics
  -> run the existing 1,000-character/200-overlap chunker per page
  -> prefix every stored chunk with [Source page N]
  -> embed and store using the existing dense index
```

Page labels are currently visible in the stored content because the Phase 4
schema migration has not happened yet. `PageChunk` already carries
`page_start`/`page_end` in memory; Phase 4 will persist those as proper columns.
No Phase 4 schema work was pulled into this phase.

## Cleaning rules

Cleaning is intentionally conservative:

- Preserve page order and one-based PDF page numbers.
- Normalize CR/LF variants, non-breaking spaces, horizontal whitespace, and
  PDF control characters while preserving paragraph and list boundaries.
- Join a wrapped line only when the previous line is sufficiently long and the
  continuation begins with a lower-case character.
- Rejoin a line-ending hyphen only when the continuation begins lower-case.
- Protect list items, labels, and table-like rows from line joining.
- Treat a header/footer as repeated furniture only when the exact normalized
  line occurs at the same page edge on at least three pages and at least 25
  percent of text-bearing pages.
- Generalize only standalone page numbers. Numbers inside headings are not
  generalized, preventing headings such as `Chapter 1` and `Chapter 2` from
  becoming false duplicates.
- Preserve the first page's edge text because cover-page content may resemble
  later headers.
- Remove exact repeated long paragraphs within one page, while preserving short
  repeated values that may be legitimate table data.
- Emit `empty-page`, `low-text-page`, or `extraction-error` diagnostics rather
  than silently ignoring a page.

## Corpus inspection

The source has 91 PDF pages. Phase 2 produced 404 page-bounded chunks, compared
with 340 cross-page chunks in the Phase 1 index.

The extractor reported 13 warnings:

- Empty pages: 3, 6, 7, 9, 11, 13, 21, 53, 71, and 85.
- Low-text pages: 2, 4, and 15.

Rendered samples of the flagged empty pages confirmed that they are intentional
blank pages, not scanned pages whose text was missed. OCR is therefore not
justified for this corpus.

Observed improvements:

- Removed the recurring PDF control character `\x03`.
- Detected recurring `State Bank of Pakistan` and
  `Governor's Annual Report 2024-25` header variants.
- Removed recurring page-number footers while preserving the title-page text.
- Reduced PDF page 45's extracted text from 11,690 to 4,936 characters by
  removing exact repeated long paragraphs from overlapping text layers.
- Every indexed chunk now begins with a visible source-page label.

Observed layout limitations:

- Two-column pages, such as PDF page 50, are flattened into plain text. The
  extractor generally emits one column followed by the other, but paragraph
  continuity across columns cannot be guaranteed by `pypdf`.
- Tables and charts are flattened into lines. Values can remain searchable, but
  row/column relationships are not guaranteed.
- PDF page 45 contains overlapping text layers. Exact long-paragraph duplicates
  are removed, but some repeated table fragments remain because their extracted
  boundaries differ.
- Some PDF glyphs are decoded as replacement or unusual Unicode characters.

These failures are visible and bounded. Advanced layout reconstruction,
specialized table extraction, and OCR remain deferred according to the project
plan.

## Representative differences

### Document start

The Phase 1 first chunk combined cover content, later front-matter pages, and
control characters because every page had been concatenated. The Phase 2 first
chunk contains only PDF page 1 and starts with:

```text
[Source page 1]
State Bank of Pakistan
GOVERNOR'S ANNUAL REPORT
2024 - 2025
```

### Public debt and M2

The old evidence crossed overlapping global chunks 52 and 53. In Phase 2,
chunk 74 is explicitly tied to PDF page 28 and contains both the public-debt
comparison and the following M2 statement. Chunk 75 is an overlapping page-28
chunk containing the full M2 statement. Neither chunk can silently cross into a
different PDF page.

### Projection table

The January 2025 projection evidence remains available in page-45 chunks
174-178. The prose evidence is cleaner, but duplicated table-layer fragments
remain and are called out rather than hidden.

## Reproducible inspection

From the repository root:

```powershell
.venv\Scripts\python.exe inspect_pdf_extraction.py `
  "C:\Users\MA\Downloads\Gov-AR_1 test.pdf" `
  --sample-pages 1 20 45 50 91
```

This writes JSON and Markdown diagnostics under `evaluation/reports/` without
calling an embedding or answer model.

Run the retrieval experiment that matches the current index with:

```powershell
.venv\Scripts\python.exe eval_retrieval.py `
  --dataset evaluation/datasets/phase2.json `
  --output-dir evaluation/reports `
  --top-k 10
```

## Retrieval experiment

The fixed 14 Phase 1 questions were retained. Since page-bounded processing
changes chunk indices, their evidence was manually relabelled against the
complete 404-chunk output before retrieval was run and saved as
`evaluation/datasets/phase2.json`. The original Phase 1 dataset and report were
not overwritten.

| Metric | Phase 1 | Phase 2 | Change |
| --- | ---: | ---: | ---: |
| Hit@1 | 0.3846 | 0.5385 | +0.1538 |
| Hit@3 | 0.6923 | 0.9231 | +0.2308 |
| Hit@5 | 0.8462 | 0.9231 | +0.0769 |
| Hit@10 | 1.0000 | 0.9231 | -0.0769 |
| MRR | 0.5955 | 0.7179 | +0.1224 |
| Recall@5 | 0.7564 | 0.7385 | -0.0179 |
| Recall@10 | 0.9744 | 0.8526 | -0.1218 |
| nDCG@5 | 0.5842 | 0.6480 | +0.0638 |
| nDCG@10 | 0.6711 | 0.7021 | +0.0309 |

Phase 2 run ID: `bebc552d57bd`.

The aggregate early-ranking result improved, but the policy-rate-reduction
question did not retrieve its relevant chunk in the top 10. This is a real
regression. The exact-period evidence is still present in chunk 19 on PDF page
18 and ranked 12 in a follow-up diagnostic. Chunk 38 ranked 6 and states the
same 1,100-basis-point reduction for June 2024 to May 2025, but it was not added
to the fixed label after seeing retrieval because its stated period differs from
the question. Later
dense-retrieval work must diagnose why it ranked poorly. Phase 2 does not tune
retrieval to hide this result.

Recall values are affected by the changed number of valid overlapping evidence
chunks, so they should be interpreted alongside the per-question diagnostics,
not as a pure retrieval-only delta.

## Current limitations before Phase 3

- Chunk size and overlap are still character-based and fixed at 1,000/200.
- Page-bounded chunks prevent cross-page chunks; Phase 3 must test whether
  important evidence spanning a page transition needs controlled neighboring or
  parent context.
- Page provenance is embedded in content until Phase 4 adds proper columns.
- The current database now contains the Phase 2 index. Run
  `evaluation/datasets/phase2.json` against it; the original
  `baseline.json` labels belong to the saved 340-chunk Phase 1 index.
- Layout, table, and glyph limitations remain visible and should inform Phase 3
  chunk inspection, but do not justify adding OCR or a layout service now.
