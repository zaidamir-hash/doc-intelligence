"""Controlled command for ingesting or re-indexing one PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from chunking import PRODUCTION_CHUNK_CONFIG
from database import SessionLocal, init_db
from ingestion import ingest_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--filename",
        help="Stored filename (defaults to the source file's basename)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even when all versions are current",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")
    if args.pdf.suffix.casefold() != ".pdf":
        raise SystemExit("Source must be a PDF file")

    init_db()
    database = SessionLocal()
    try:
        result = ingest_document(
            database,
            args.filename or args.pdf.name,
            args.pdf.read_bytes(),
            PRODUCTION_CHUNK_CONFIG,
            force=args.force,
        )
    finally:
        database.close()

    action = "reused" if result.reused_existing_index else "indexed"
    print(f"Document ID: {result.document_id}")
    print(f"Content SHA-256: {result.document_content_hash}")
    print(f"Action: {action}")
    print(f"Pages: {result.extraction.page_count}")
    print(f"Chunks: {len(result.chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
