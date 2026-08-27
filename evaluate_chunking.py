"""Compare bounded Phase 3 chunk configurations without retrieval tuning."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from chunking import ChunkingConfig, StructuredChunk, chunk_extraction, summarize_chunks
from pdf_processing import extract_pdf


DEFAULT_EVIDENCE = Path("evaluation/datasets/phase3_evidence.json")
DEFAULT_OUTPUT_DIRECTORY = Path("evaluation/reports")
CONFIGURATIONS = {
    "small": ChunkingConfig(max_tokens=200, overlap_tokens=30),
    "balanced": ChunkingConfig(max_tokens=300, overlap_tokens=50),
    "large": ChunkingConfig(max_tokens=400, overlap_tokens=70),
}


def _normalized(text: str) -> str:
    return (
        re.sub(r"\s+", " ", text)
        .replace("–", "-")
        .replace("—", "-")
        .casefold()
        .strip()
    )


def matching_chunks(
    chunks: list[StructuredChunk], evidence_case: dict[str, object]
) -> list[int]:
    pages = set(evidence_case["pages"])
    pattern_groups = evidence_case["pattern_groups"]
    matches: list[int] = []
    for index, chunk in enumerate(chunks):
        chunk_pages = set(range(chunk.page_start, chunk.page_end + 1))
        if not pages.intersection(chunk_pages):
            continue
        normalized_text = _normalized(chunk.passage_text)
        if any(
            all(re.search(pattern, normalized_text) for pattern in group)
            for group in pattern_groups
        ):
            matches.append(index)
    return matches


def evaluate_configuration(
    chunks: list[StructuredChunk], evidence_cases: list[dict[str, object]]
) -> dict[str, object]:
    evidence_results = []
    for case in evidence_cases:
        matches = matching_chunks(chunks, case)
        evidence_results.append(
            {
                "id": case["id"],
                "preserved": bool(matches),
                "matching_chunk_indices": matches,
            }
        )
    return {
        "summary": summarize_chunks(chunks),
        "evidence": evidence_results,
        "evidence_cases_preserved": sum(
            result["preserved"] for result in evidence_results
        ),
        "evidence_case_count": len(evidence_results),
    }


def select_configuration(results: dict[str, dict[str, object]]) -> str:
    """Require full evidence, then minimize duplicates and oversized context."""

    eligible = [
        name
        for name, result in results.items()
        if result["evidence_cases_preserved"] == result["evidence_case_count"]
        and result["summary"]["exact_duplicate_hashes"] == 0
    ]
    if not eligible:
        raise ValueError("No configuration preserved every labelled evidence case")
    return min(
        eligible,
        key=lambda name: (
            results[name]["summary"]["adjacent_token_jaccard"][
                "near_duplicate_pairs_at_0_8"
            ],
            abs(CONFIGURATIONS[name].max_tokens - 300),
            CONFIGURATIONS[name].max_tokens,
        ),
    )


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Phase 3 bounded chunking experiment",
        "",
        f"- Source: `{report['source']}`",
        f"- Evidence cases: `{report['evidence_case_count']}`",
        f"- Structure-only recommendation: `{report['selected_configuration']}`",
        f"- Parent-child decision: `{report['parent_child_decision']}`",
        "",
        "| Configuration | Max/overlap | Chunks | Mean tokens | Evidence | "
        "Near-duplicate pairs | Cross-page |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in report["configurations"].items():
        config = result["configuration"]
        summary = result["summary"]
        lines.append(
            f"| {name} | {config['max_tokens']}/{config['overlap_tokens']} | "
            f"{summary['chunk_count']} | {summary['token_count']['mean']:.2f} | "
            f"{result['evidence_cases_preserved']}/{result['evidence_case_count']} | "
            f"{summary['adjacent_token_jaccard']['near_duplicate_pairs_at_0_8']} | "
            f"{summary['cross_page_chunks']} |"
        )
    for name, result in report["configurations"].items():
        lines.extend(["", f"## {name}", ""])
        for evidence in result["evidence"]:
            lines.append(
                f"- `{evidence['id']}`: preserved=`{str(evidence['preserved']).lower()}`, "
                f"chunks=`{evidence['matching_chunk_indices']}`"
            )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evidence_document = json.loads(args.evidence.read_text(encoding="utf-8"))
    evidence_cases = evidence_document["cases"]
    extraction = extract_pdf(args.pdf.read_bytes())
    results: dict[str, dict[str, object]] = {}
    for name, config in CONFIGURATIONS.items():
        chunks = chunk_extraction(extraction, config)
        result = evaluate_configuration(chunks, evidence_cases)
        result["configuration"] = {
            "max_tokens": config.max_tokens,
            "overlap_tokens": config.overlap_tokens,
            "min_chunk_tokens": config.min_chunk_tokens,
            "encoding_name": config.encoding_name,
            "version": config.version,
        }
        results[name] = result

    selected = select_configuration(results)
    report = {
        "source": str(args.pdf.resolve()),
        "evidence_source": str(args.evidence.resolve()),
        "evidence_case_count": len(evidence_cases),
        "selection_rule": (
            "Preserve every evidence case, reject exact duplicate identities, "
            "then minimize adjacent near-duplicates and prefer the bounded "
            "300-token center configuration."
        ),
        "structure_only_recommendation": selected,
        "selected_configuration": selected,
        "parent_child_decision": (
            "defer unless all bounded child configurations fail to preserve "
            "evidence or the selected retrieval experiment demonstrates a "
            "need for larger generation context"
        ),
        "configurations": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "chunking-experiment.json"
    markdown_path = args.output_dir / "chunking-experiment.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Selected configuration: {selected}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
