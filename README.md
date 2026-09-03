# Lexis — inspectable advanced RAG

Lexis is a learning-focused Retrieval-Augmented Generation project for PDF
documents. It implements the important RAG stages explicitly—without hiding
retrieval, fusion, reranking, evidence selection, or citation validation behind
an orchestration framework.

The project is complete at Phase 13. Its purpose is to teach the full pipeline
and its failure modes, not to become a production multi-tenant document service.

## What is included

- page-aware PDF extraction and conservative cleaning;
- token- and structure-aware chunks with stable hashes and overlap metadata;
- versioned PostgreSQL storage with pgvector embeddings and lexical `tsvector`;
- independently callable dense and lexical retrieval;
- explicit Reciprocal Rank Fusion (RRF);
- question-passage reranking with a visible 0–3 relevance rubric;
- safe query expansion that always preserves the original query;
- relevance filtering, grounded refusals, verified claims, and page citations;
- separate retrieval and answer-quality evaluation;
- protected FastAPI endpoints and a React learning/debug interface.

See [the architecture guide](docs/ARCHITECTURE.md) for the complete data flow and
[the final ablation](evaluation/experiments/FINAL_ABLATION.md) for what each
retrieval addition actually changed.

## Prerequisites

- Python 3.12 (the final validation used Python 3.12 on Windows);
- Docker Desktop with Docker Compose;
- Node.js 20 or newer and npm;
- an OpenAI API key for embeddings, expansion, reranking, and answers.

## First-time setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and replace the API keys and database password. Keep
`POSTGRES_PASSWORD` and the password inside `DATABASE_URL` identical. If the
password contains URL-reserved characters, URL-encode it inside `DATABASE_URL`.
Never commit `.env`.

Hosted model requests use a 30-second per-attempt timeout and at most two SDK
retries by default. Adjust `LEXIS_MODEL_TIMEOUT_SECONDS` and
`LEXIS_MODEL_MAX_RETRIES` in `.env` when a different bounded policy is needed.

Start PostgreSQL/pgvector and apply the checksum-tracked migrations:

```powershell
docker compose up -d db
.venv\Scripts\python.exe migrate.py status
.venv\Scripts\python.exe migrate.py upgrade
```

Docker only applies `POSTGRES_*` initialization values when it creates a new
database volume. Changing them later does not change an existing volume's
database password.

## Run Lexis

Backend, from the repository root:

```powershell
.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Frontend, in a second terminal:

```powershell
Set-Location doc-intelligence-frontend
npm ci
npm run dev
```

Open `http://localhost:5173`, enter the `API_KEY` from `.env`, and press
**Connect**. Upload a text-based PDF, wait for `Indexed`, and select **Query →**.
Enable **Learning debug** only when you want rank and expansion diagnostics.

## Migrations, ingestion, and re-indexing

Inspect or upgrade schema history:

```powershell
.venv\Scripts\python.exe migrate.py status
.venv\Scripts\python.exe migrate.py upgrade
```

Index a local PDF without the browser:

```powershell
.venv\Scripts\python.exe reindex_document.py "C:\path\report.pdf"
```

Force re-extraction, re-chunking, and re-embedding when the stored versions are
already current:

```powershell
.venv\Scripts\python.exe reindex_document.py "C:\path\report.pdf" --force
```

The command prints the stable document UUID and content SHA-256. An identical
filename/content pair reuses its current index unless `--force` is supplied. A
successful changed version supersedes the previous ready version; failures do
not destroy a ready index.

## Tests

Fast unit tests, with no database/model calls:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Full PostgreSQL integration suite:

```powershell
$env:RUN_DATABASE_TESTS = "1"
.venv\Scripts\python.exe -m unittest discover -s tests -v
Remove-Item Env:RUN_DATABASE_TESTS
```

Frontend checks:

```powershell
Set-Location doc-intelligence-frontend
npm run lint
npm run build
```

## Evaluation

The evaluation commands require the matching version of the labelled 91-page
document to be indexed. Do not run old chunk labels against a newly chunked
index.

```powershell
# Phase 1/3 dense retrieval baseline
.venv\Scripts\python.exe eval_retrieval.py

# Final named retrieval stages
.venv\Scripts\python.exe evaluate_dense_retrieval.py
.venv\Scripts\python.exe evaluate_lexical_retrieval.py
.venv\Scripts\python.exe evaluate_hybrid_retrieval.py
.venv\Scripts\python.exe evaluate_reranking.py
.venv\Scripts\python.exe evaluate_query_expansion.py

# Separate grounded-answer evaluation
.venv\Scripts\python.exe evaluate_answers.py
```

`evaluation/reports/` is intentionally local because detailed retrieval reports
can contain uploaded-document text. The Phase 11 raw JSON is also ignored. The
tracked Markdown summaries and comparison JSONs contain the safe reproducibility
record.

Regrade the exact locally saved Phase 11 model outputs without another model
call:

```powershell
.venv\Scripts\python.exe evaluate_answers.py `
  --regrade-existing evaluation/experiments/phase11_answer_evaluation.json
```

The stable corpus/report regression can be checked independently:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_final_regression -v
```

Detailed evaluation methodology is in [docs/EVALUATION.md](docs/EVALUATION.md).

## Main code map

| Area | Important files |
| --- | --- |
| Extraction/chunking | `pdf_processing.py`, `chunking.py` |
| Versioned ingestion | `ingestion.py`, `models.py`, `migrations/` |
| Dense/lexical retrieval | `dense_retrieval.py`, `lexical_retrieval.py` |
| Fusion/expansion | `hybrid_retrieval.py`, `expanded_retrieval.py`, `query_expansion.py` |
| Reranking/generation | `reranking.py`, `grounded_generation.py`, `query.py` |
| API/configuration | `main.py`, `auth.py`, `settings.py` |
| Evaluation | `evaluation/`, `evaluate_*.py`, `eval_retrieval.py` |
| Frontend | `doc-intelligence-frontend/src/` |

## Security and operational boundaries

- API keys and database credentials come from `.env`; only safe placeholders
  belong in Git.
- The backend uses API-key protection, CORS allowlists, upload/query rate limits,
  bounded PDF size, MIME/signature checks, and redacted client errors.
- PDF text and model output are untrusted data. Grounded generation validates
  evidence IDs and supporting quotes before claims are displayed. If a model
  answer fails that validation, Lexis permits one constrained repair attempt
  and validates the result again; an invalid repair still fails closed.
- The frontend keeps a validated key in tab-scoped `sessionStorage` so refresh
  can restore state; Disconnect or closing the tab clears it.

## Known limitations and closed scope

- Scanned/image-only PDFs require OCR, which is not included.
- Complex multi-column layouts, tables, charts, and multimodal retrieval are not
  handled specially.
- Upload/indexing is synchronous and can be slow for large documents.
- Authentication is one shared API key, not user accounts or multi-tenancy.
- Hosted expansion, reranking, and answering add latency, cost, and some
  nondeterminism even with pinned model snapshots.
- The labelled corpus is one 91-page report plus bounded synthetic safety cases;
  metrics are learning evidence, not a broad production guarantee.
- There is no persistent conversation memory, agentic retrieval, distributed
  job queue, cloud deployment, or production observability stack.

These are deliberate stopping points. They are recorded instead of expanding
Lexis beyond its learning objective.
