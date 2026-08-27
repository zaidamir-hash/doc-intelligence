"""Bind evaluation labels to stable ready-document and chunk content hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from database import SessionLocal, init_db
from evaluation.dataset import load_dataset
from models import DOCUMENT_STATUS_READY, Document, DocumentChunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def bind_identities(document: dict[str, object], database) -> dict[str, object]:
    """Attach stable hashes matching the supplied dataset's current indices."""

    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError("dataset cases must be a list")

    cache: dict[str, tuple[Document, dict[int, str]]] = {}
    for case in cases:
        filename = case["filename"]
        if filename not in cache:
            ready_document = (
                database.query(Document)
                .filter(
                    Document.original_filename == filename,
                    Document.status == DOCUMENT_STATUS_READY,
                )
                .one_or_none()
            )
            if ready_document is None:
                raise ValueError(f"No ready document found for {filename}")
            chunk_hashes = dict(
                database.query(DocumentChunk.chunk_index, DocumentChunk.content_hash)
                .filter(DocumentChunk.document_id == ready_document.id)
                .all()
            )
            cache[filename] = (ready_document, chunk_hashes)

        ready_document, chunk_hashes = cache[filename]
        indices = case["relevant_chunk_indices"]
        missing = [index for index in indices if index not in chunk_hashes]
        if missing:
            raise ValueError(
                f"Case {case['id']} references missing chunk indices {missing}"
            )
        case["document_content_hash"] = ready_document.content_hash
        case["relevant_chunk_hashes"] = [chunk_hashes[index] for index in indices]
    return document


def main() -> int:
    args = parse_args()
    load_dataset(args.dataset)
    raw_document = json.loads(args.dataset.read_text(encoding="utf-8"))
    init_db()
    database = SessionLocal()
    try:
        bound_document = bind_identities(raw_document, database)
    finally:
        database.close()

    output = args.output or args.dataset
    output.write_text(json.dumps(bound_document, indent=2), encoding="utf-8")
    load_dataset(output)
    print(f"Bound stable identities: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
