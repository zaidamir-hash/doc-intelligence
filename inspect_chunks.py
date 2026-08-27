"""Inspect every Phase 3 chunk and its provenance/identity metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chunking import ChunkingConfig, chunk_extraction, summarize_chunks
from pdf_processing import extract_pdf


DEFAULT_OUTPUT_DIRECTORY = Path("evaluation/reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Phase 3 token/structure-aware chunk diagnostics."
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--min-chunk-tokens", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def render_markdown(report: dict[str, object]) -> str:
    config = report["configuration"]
    summary = report["summary"]
    lines = [
        "# Phase 3 chunk inspection",
        "",
        f"- Source: `{report['source']}`",
        f"- Version: `{config['version']}`",
        f"- Tokenizer: `{config['encoding_name']}`",
        f"- Maximum tokens: `{config['max_tokens']}`",
        f"- Requested overlap: `{config['overlap_tokens']}`",
        f"- Minimum target: `{config['min_chunk_tokens']}`",
        f"- Chunks: `{summary['chunk_count']}`",
        f"- Token range: `{summary['token_count']['minimum']}` to "
        f"`{summary['token_count']['maximum']}`",
        f"- Cross-page chunks: `{summary['cross_page_chunks']}`",
        f"- Chunks with heading context: `{summary['chunks_with_heading']}`",
        f"- Exact duplicate hashes: `{summary['exact_duplicate_hashes']}`",
        f"- Adjacent near-duplicate pairs (Jaccard >= 0.8): "
        f"`{summary['adjacent_token_jaccard']['near_duplicate_pairs_at_0_8']}`",
        "",
        "## Chunks",
        "",
    ]
    for chunk in report["chunks"]:
        lines.extend(
            [
                f"### Chunk {chunk['chunk_index']}",
                "",
                f"- Pages: `{chunk['page_start']}-{chunk['page_end']}`",
                f"- Section: `{chunk['section_title']}`",
                f"- Tokens: `{chunk['token_count']}`",
                f"- Actual overlap tokens: `{chunk['overlap_token_count']}`",
                f"- Boundary kinds: `{chunk['boundary_kinds']}`",
                f"- SHA-256: `{chunk['content_hash']}`",
                "",
                "```text",
                chunk["content"],
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    config = ChunkingConfig(
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        min_chunk_tokens=args.min_chunk_tokens,
    )
    extraction = extract_pdf(args.pdf.read_bytes())
    chunks = chunk_extraction(extraction, config)
    report = {
        "source": str(args.pdf.resolve()),
        "configuration": {
            "max_tokens": config.max_tokens,
            "overlap_tokens": config.overlap_tokens,
            "min_chunk_tokens": config.min_chunk_tokens,
            "encoding_name": config.encoding_name,
            "version": config.version,
        },
        "summary": summarize_chunks(chunks),
        "chunks": [
            {
                "chunk_index": index,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_title": chunk.section_title,
                "token_count": chunk.token_count,
                "overlap_token_count": chunk.overlap_token_count,
                "content_hash": chunk.content_hash,
                "chunking_version": chunk.chunking_version,
                "boundary_kinds": list(chunk.boundary_kinds),
                "content": chunk.content,
            }
            for index, chunk in enumerate(chunks)
        ],
    }
    run_name = (
        f"chunks-{config.version}-{config.max_tokens}-{config.overlap_tokens}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{run_name}.json"
    markdown_path = args.output_dir / f"{run_name}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
