"""Create a human- and machine-readable Phase 2 PDF extraction report."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pdf_processing import (
    ExtractionResult,
    chunk_extracted_pages,
    extract_pdf,
    summarize_extraction,
)


DEFAULT_OUTPUT_DIRECTORY = Path("evaluation/reports")
DEFAULT_SAMPLE_PAGES = (1, 20, 50, 91)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect page-aware PDF extraction without embedding or indexing."
    )
    parser.add_argument("pdf", type=Path, help="PDF to inspect")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_PAGES),
        help="One-based pages whose raw and cleaned previews are included",
    )
    parser.add_argument("--preview-characters", type=int, default=800)
    return parser.parse_args()


def build_report(
    pdf_path: Path,
    contents: bytes,
    extraction: ExtractionResult,
    sample_pages: list[int],
    preview_characters: int,
) -> dict[str, object]:
    if preview_characters <= 0:
        raise ValueError("preview_characters must be positive")
    invalid_pages = [
        page for page in sample_pages if page < 1 or page > extraction.page_count
    ]
    if invalid_pages:
        raise ValueError(f"sample pages outside the PDF: {invalid_pages}")

    chunks = chunk_extracted_pages(extraction)
    return {
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_path": str(pdf_path.resolve()),
            "source_sha256": hashlib.sha256(contents).hexdigest(),
        },
        "extraction": summarize_extraction(extraction),
        "chunking": {
            "strategy": "existing character chunker applied independently per page",
            "chunk_size_characters": 1000,
            "overlap_characters": 200,
            "chunk_count": len(chunks),
            "page_provenance": "visible [Source page N] prefix",
        },
        "sample_pages": [
            {
                "page_number": page_number,
                "raw_preview": extraction.pages[page_number - 1].raw_text[
                    :preview_characters
                ],
                "cleaned_preview": extraction.pages[page_number - 1].cleaned_text[
                    :preview_characters
                ],
                "removed_headers": list(
                    extraction.pages[page_number - 1].removed_headers
                ),
                "removed_footers": list(
                    extraction.pages[page_number - 1].removed_footers
                ),
            }
            for page_number in sample_pages
        ],
        "chunk_samples": [
            {
                "chunk_index": index,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "preview": chunk.content[:preview_characters],
            }
            for index, chunk in enumerate(chunks)
            if chunk.page_start in sample_pages
        ],
    }


def render_markdown(report: dict[str, object]) -> str:
    run = report["run"]
    extraction = report["extraction"]
    chunking = report["chunking"]
    lines = [
        "# Phase 2 PDF extraction inspection",
        "",
        f"- Source: `{run['source_path']}`",
        f"- SHA-256: `{run['source_sha256']}`",
        f"- Pages: `{extraction['page_count']}`",
        f"- Chunks: `{chunking['chunk_count']}`",
        f"- Extraction warnings: `{extraction['warning_count']}`",
        f"- Repeated headers: `{extraction['repeated_headers']}`",
        f"- Repeated footers: `{extraction['repeated_footers']}`",
        "",
        "## Pages with extraction warnings",
        "",
    ]
    warning_pages = extraction["pages_with_warnings"]
    if not warning_pages:
        lines.append("None.")
    for page in warning_pages:
        messages = "; ".join(
            f"{warning['code']}: {warning['message']}"
            for warning in page["warnings"]
        )
        lines.append(f"- Page {page['page_number']}: {messages}")

    lines.extend(["", "## Representative page diffs", ""])
    for page in report["sample_pages"]:
        lines.extend(
            [
                f"### PDF page {page['page_number']}",
                "",
                f"Removed headers: `{page['removed_headers']}`  ",
                f"Removed footers: `{page['removed_footers']}`",
                "",
                "Raw extraction:",
                "",
                "```text",
                page["raw_preview"],
                "```",
                "",
                "Cleaned extraction:",
                "",
                "```text",
                page["cleaned_preview"],
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    contents = args.pdf.read_bytes()
    extraction = extract_pdf(contents)
    report = build_report(
        args.pdf,
        contents,
        extraction,
        args.sample_pages,
        args.preview_characters,
    )
    run_id = report["run"]["source_sha256"][:12]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"pdf-extraction-{run_id}.json"
    markdown_path = args.output_dir / f"pdf-extraction-{run_id}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
