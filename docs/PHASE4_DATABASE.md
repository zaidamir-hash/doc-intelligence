# Phase 4: persistent document model and controlled re-indexing

## Outcome

Phase 4 replaces the filename-only `document_chunks` table with a versioned
document lifecycle and metadata-rich chunks. A ready document can now be traced
from its raw PDF hash and UUID to every page-aware, versioned dense/lexical
passage stored for it.

The migration preserves legacy documents, removes only duplicate legacy rows
that share the same `(filename, chunk_index)`, and keeps those documents
queryable until their original PDFs are re-indexed.

## Identity policy

Lexis uses several identities because they answer different questions:

- `documents.id` is a generated UUID: the stable database identity of one
  filename/content version.
- `documents.content_hash` is SHA-256 of the exact uploaded PDF bytes: it tells
  whether two uploads contain the same bytes.
- `document_chunks.id` is a generated UUID: the stable identity of one stored
  row during that index's lifetime.
- `document_chunks.content_hash` is Phase 3's deterministic chunk identity:
  passage, page range, heading, and chunker version produce the same hash again.
- `chunk_index` preserves document order but is explicitly transient; it may
  change after re-chunking and is no longer the only evaluation identity.

The upload/re-index behavior is deterministic:

| Upload | Behavior |
| --- | --- |
| Same filename and same bytes, current versions | Reuse the existing UUID and index; do not call embeddings. |
| Same filename and same bytes, changed processing version | Atomically replace chunks on the existing document UUID. |
| Same filename and different bytes | Create a new document version; mark the old ready version `superseded` only after the new index commits. |
| Same bytes under a different filename | Create a separate document because user-visible filename scope is different. |
| New ingestion fails | Store a `failed` document record with no chunks; retain any earlier ready version. |
| Forced re-index of a ready document fails | Keep its previous ready chunks and status unchanged. |

## Document lifecycle

```text
new/retried upload
       ↓
  processing ──failure──→ failed
       │
       └──complete transaction──→ ready
                                    │
same filename, different bytes      │
       new version becomes ready ───┘
       previous ready version → superseded
```

Only `ready` documents participate in production or evaluation retrieval.
`processing`, `failed`, and `superseded` records therefore cannot expose a
partial or obsolete index as current evidence.

## Database schema

### `documents`

The table stores:

- UUID primary key;
- original filename;
- raw PDF SHA-256;
- page and chunk counts;
- processing status and optional error;
- extraction, chunking, and embedding versions;
- complete tokenizer/chunk-size/overlap/minimum configuration used to decide
  whether a same-content index is still current;
- creation and update timestamps.

Important constraints/indexes include:

- unique `(original_filename, content_hash)` for idempotent identity;
- one partial unique index allowing only one `ready` version per filename;
- status and non-negative count checks;
- status index for lifecycle queries.

### `document_chunks`

The table stores:

- UUID primary key and `document_id` foreign key with cascade delete;
- transient ordered `chunk_index`;
- rendered embedding content and clean `passage_text`;
- page start/end and nullable section title;
- deterministic chunk content hash;
- token and actual-overlap counts;
- chunker and embedding versions;
- boundary kinds from Phase 3;
- generated lexical `search_vector`;
- 1,536-dimensional pgvector embedding;
- creation timestamp.

Important constraints/indexes include:

- unique `(document_id, chunk_index)`;
- unique `(document_id, content_hash)`;
- page/token/index validity checks;
- document and page-range B-tree indexes;
- GIN index over `search_vector`.

`search_vector` was introduced in Phase 4 as a PostgreSQL generated column.
Phase 6 migration `0004` refined it to weight headings and passage text:

```sql
setweight(to_tsvector('english', coalesce(section_title, '')), 'A')
|| setweight(to_tsvector('english', coalesce(passage_text, '')), 'B')
```

PostgreSQL updates it automatically whenever the heading or passage changes.
Phase 4 prepared lexical storage; Phase 6 implements and evaluates ranking.
No approximate pgvector index was added because choosing and measuring dense
index/search behavior belongs to Phase 5.

`parent_chunk_id` was not added because Phase 3 explicitly deferred
parent-child chunking after all 13 labelled evidence cases were preserved and
retrieved within five results.

## Migrations

Run pending migrations:

```powershell
.venv\Scripts\python.exe migrate.py upgrade
```

Inspect status and checksums:

```powershell
.venv\Scripts\python.exe migrate.py status
```

`schema_migrations.py` discovers files named `NNNN_description.sql`, calculates
their SHA-256 checksums, acquires a PostgreSQL advisory transaction lock, and
applies pending files in order inside one database transaction. If an applied
file is later edited, execution stops because its checksum no longer matches.

Phase 4 contains:

- `0001_phase4_document_schema.sql`: creates the new schema, transforms legacy
  rows, validates migrated distinct-row counts, then removes the transformed
  legacy table inside the same transaction;
- `0002_normalize_chunk_pkey_name.sql`: normalizes an index name that PostgreSQL
  temporarily suffixed while the renamed legacy table still owned the old name.
- `0003_store_chunking_config.sql`: stores and validates the full chunking
  configuration so a settings change triggers controlled re-indexing even when
  an algorithm version name was not changed.

The project uses small forward-only SQL migrations instead of introducing
Alembic at this stage. This keeps the mechanics visible for learning and adds no
dependency. Production deployment would still require a database backup and a
separate reviewed recovery migration; applied migrations are never edited.

## Transactional ingestion

`ingestion.py` is the single implementation shared by HTTP upload and the
re-index CLI.

1. Hash the exact PDF bytes.
2. Find or create the target document lifecycle record.
3. Extract and chunk using Phase 2 and Phase 3 code.
4. Generate embeddings in batches of 64.
5. Start the final database transaction state:
   - supersede the previous ready version when necessary;
   - delete old target chunks only for a successful re-index;
   - insert every new metadata-rich chunk;
   - set counts, versions, and status to `ready`.
6. Commit once.

Embedding generation completes before ready chunks are changed. Any database
failure rolls back all chunk changes and the lifecycle transition together.

## Controlled re-indexing

Run:

```powershell
.venv\Scripts\python.exe reindex_document.py `
  "C:\Users\MA\Downloads\Gov-AR_1 test.pdf"
```

Use `--filename` only when the stored filename should differ from the source
basename. Use `--force` to recompute a current index intentionally.

The command runs migrations first and then calls the same `ingest_document`
function as the API. It prints the stable UUID, raw content hash, action, page
count, and chunk count. A second ordinary run on the same file reports `reused`
and makes no embedding calls.

## Stable evaluation identities

Phase 1 originally labelled transient chunk indices. Phase 4 keeps those
indices for readable document order but adds optional:

- `document_content_hash`;
- `relevant_chunk_hashes` aligned with the labelled indices.

When hashes exist, the evaluation runner scores retrieved chunk hashes rather
than indices and filters the database by the expected document hash. Thus, a
chunk can move from index 3 to index 999 without losing its label if its stable
identity is unchanged.

Bind an existing dataset to the current ready index with:

```powershell
.venv\Scripts\python.exe bind_evaluation_identities.py `
  evaluation\datasets\phase3.json
```

Future Phase 3 comparison runs write the source PDF and chunk hashes directly.
Historical datasets without hashes remain readable and explicitly use
`legacy_chunk_index` evaluation mode.

## Migration and regression results

The migration was first tested on a disposable legacy-shaped PostgreSQL
database. A three-row input with one duplicated chunk index became two chunks;
page ranges, heading context, lexical vectors, migration history, and idempotent
reruns were verified before touching the live database.

Live legacy migration results:

| Filename | Legacy rows before | Distinct chunks after |
| --- | ---: | ---: |
| `Agha Zaid Amir Agreement.pdf` | 6 | 6 |
| `A.Zaid_Amir_Resume.pdf` | 6 | 6 |
| `Doc intel test pdf.pdf` | 276 | 69 |
| `Gov-AR_1 test.pdf` | 424 | 424 |

The 91-page PDF was then re-indexed through the controlled command:

- document UUID: `ad8187b9-577a-40c5-a8fa-2c0072aa78a6`;
- raw PDF SHA-256:
  `dd784d0aeb42939384ec433cef08498bbfe6e30649a2963e360519ce0b291074`;
- 424 chunks, all with page ranges, positive token counts, unique hashes, and
  generated lexical vectors;
- second run reused the same UUID/index.

Post-migration pgvector evaluation run `f4f3d2874679` used stable hash labels
and reproduced Phase 3 exactly: Hit@5 `1.0000`, MRR `0.5179`.

## Legacy limitations

The three PDFs whose source files were not supplied remain queryable legacy
documents. Their document hashes are deterministic hashes of retained stored
passages—not raw PDF hashes—and metadata unavailable in the old table is marked
with `legacy-unversioned`, zero token count, or nullable page/section values.
Running their original PDFs through `reindex_document.py` upgrades them to full
Phase 4 provenance without deleting the ready legacy index before success.

The application still accepts filename-based query requests for frontend
compatibility. Retrieval internally resolves that filename to the single ready
document UUID. Changing the external query API and frontend selection to UUIDs
is explicitly Phase 12 work.
