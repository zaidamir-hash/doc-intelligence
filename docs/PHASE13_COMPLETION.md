# Phase 13 completion record

This record closes the required Lexis roadmap without extending the project
into any deferred features.

## Final verification

The following checks were executed on 2026-09-01 from the repository root:

```powershell
$env:RUN_DATABASE_TESTS = "1"
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe migrate.py status

Set-Location doc-intelligence-frontend
npm run lint
npm run build
```

Results:

- 113 Python tests passed, including the PostgreSQL upload/index/list/retrieve/
  query/delete lifecycle and fixed-corpus regression tests;
- all four database migrations were applied with valid checksums;
- Python dependency consistency passed;
- frontend lint and production build passed;
- the local Vite page returned HTTP 200 with the expected React root;
- deterministic Phase 11 regrading preserved retrieval success `1.0000`,
  answer success `0.8333`, and unsupported-claim rate `0.0000`.

The in-app visual browser could not start because its host runtime failed to
write its own kernel assets. This is an external validation-tool limitation,
not an application error; the frontend was still verified through lint, build,
and local HTTP serving.

## Transaction and failure-cleanup review

- Ingestion writes chunks and promotes a document to `ready` in one committed
  transaction. A failure rolls back partial chunks and records the failed
  version separately, preserving an earlier ready index.
- Re-indexing only supersedes an older ready version after the replacement is
  complete.
- An embedding-count mismatch now fails explicitly instead of allowing Python
  `zip` to silently store only part of a document.
- Document deletion uses the stable document UUID and database cascade rules;
  processing documents cannot be deleted mid-ingestion.
- Migration execution uses a transaction lock and checksum verification.

The database integration tests exercise success, idempotency, supersession,
failed indexing, incomplete embedding output, and cascade deletion.

## Security review

- `.env` remains Git-ignored and `.env.example` contains placeholders only.
- PostgreSQL no longer has a committed password; Docker Compose requires
  `POSTGRES_PASSWORD` from the environment or `.env`.
- API-key comparison is constant-time and authentication errors do not echo the
  key.
- CORS origins, accepted PDF MIME types, maximum upload size, filename length,
  and upload/query rate limits are centrally configured.
- Uploads require a safe `.pdf` filename, accepted MIME type, and PDF signature.
- Internal indexing errors and failed-document details are redacted from API
  responses.
- Detailed evaluation JSON remains local because it may contain source passages
  and model outputs.

This is appropriate for a local learning project. Shared-key authentication,
synchronous ingestion, and local process rate limiting are not presented as
production multi-user security.

## Evaluation closure

The final comparison is in
`evaluation/experiments/FINAL_ABLATION.md`; its machine-checkable source is
`evaluation/experiments/final_ablation_report.json`. It compares every required
retrieval stage against the fixed 91-page Phase 1 baseline and reports category
behavior, observed latency, and cost without inventing missing measurements.
Retrieval quality and answer quality remain separate.

Historical model-backed configurations were consolidated from their versioned
experiment artifacts rather than rerun and potentially changed by current model
nondeterminism. The fixed dataset hashes, required stage identities, and metric
floors are enforced by `tests/test_final_regression.py`.

## Closed scope

OCR, layout/table/multimodal retrieval, conversational memory, agentic RAG,
multi-tenancy, distributed jobs, cloud deployment, and large-scale observability
remain deliberately deferred. The required 13-phase learning project stops
here.
