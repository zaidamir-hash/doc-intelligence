# Phase 12 API and frontend integration

Phase 12 connects the persistent document and RAG pipeline from Phases 1–10 to
the browser UI. PostgreSQL is authoritative for document state; the frontend
does not invent permanent document rows.

## Document lifecycle

The API exposes these protected endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/auth/validate` | Validate the supplied `X-API-Key` against the backend. |
| `GET` | `/configuration` | Return non-secret active model, chunking, and retrieval settings. |
| `GET` | `/documents` | List current `processing`, `ready`, and `failed` document records. |
| `GET` | `/documents/{document_id}` | Read one current document by stable UUID. |
| `DELETE` | `/documents/{document_id}` | Delete one non-processing document and its chunks. |
| `POST` | `/upload` | Validate and synchronously index one PDF. |
| `POST` | `/query` | Query one ready document by stable UUID. |

Superseded document versions remain available for reproducibility inside the
database but are not shown as current documents. Re-uploading identical bytes
under the same filename reuses the same index and UUID. Re-uploading changed
bytes under the same filename creates a new stable version; after it becomes
ready, the previous ready version becomes superseded. Deleting a document uses
its UUID, and PostgreSQL cascades deletion to its chunks. A document in the
`processing` state cannot be deleted.

The upload call remains synchronous in Phase 12. The frontend shows an active
processing state and refreshes the document endpoint while the request runs;
the backend lifecycle record remains the authoritative status.

## Query and evidence contract

Normal queries send:

```json
{
  "document_id": "stable-document-uuid",
  "question": "the user's question",
  "debug": false
}
```

The backend resolves the UUID to the exact filename and content hash before it
runs retrieval. This prevents an identically named document version from being
queried accidentally. Normal responses include the grounded answer/refusal,
claims, page-aware citations, and only the selected evidence sent to the answer
model.

When `debug` is `true`, the response additionally includes:

- active retrieval mode and non-secret configuration;
- original and expanded-query behavior;
- dense, lexical, fused, and reranked ranks;
- reranker scores and relevance decisions;
- evidence suppression and fallback diagnostics.

Debug mode is educational and optional. Raw provider exceptions are converted
to safe fallback descriptions before they leave the backend.

## Validation and security behavior

- Filenames are reduced to a display-safe basename, bounded in length, and
  required to end in `.pdf` (case-insensitive).
- The configured PDF MIME type, upload size, non-empty content, and `%PDF-`
  signature are validated.
- Scanned/image-only PDFs return HTTP 422 with an explanation that OCR is not
  enabled.
- API, embedding, and processing failures return structured, actionable errors
  without returning raw exception text.
- Authentication and upload/query rate limits remain enabled and are configured
  through environment variables.
- The frontend never logs the API key. After successful backend validation it
  stores the key in tab-scoped `sessionStorage` so a page refresh can revalidate
  and restore documents; Disconnect or editing the key removes it. Closing the
  browser tab also clears it.

## Central configuration

`settings.py` owns the environment-backed production settings. Model modules,
production chunking, the query pipeline, API policy, and the debug configuration
endpoint consume the same settings object. `.env.example` contains safe names
and placeholders for every new variable. `VITE_API_URL` configures the frontend
backend URL without hard-coding it in components. If the backend key-header
name is changed, `VITE_API_KEY_HEADER` must be set to the same name for the
frontend build.

The active production retrieval mode is
`hybrid_rrf_rerank_expansion`: original and safe expanded queries each use dense
and lexical retrieval, RRF fuses the candidate ranks, the reranker scores the
fused passages, and only evidence meeting the relevance rules can reach answer
generation. Standalone dense, lexical, and hybrid functions from earlier phases
remain callable for evaluation and ablation; Phase 12 does not duplicate them or
add a second retrieval implementation.
