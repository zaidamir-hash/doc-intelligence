"""Write machine-readable and human-readable evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a report that traces aggregate metrics to individual cases."""

    run = report["run"]
    dataset = report["dataset"]
    configuration = report["configuration"]
    aggregate = report["aggregate_metrics"]

    lines = [
        f"# Retrieval Evaluation — {dataset['name']}",
        "",
        "## Run",
        "",
        f"- Run ID: `{run['run_id']}`",
        f"- Started: `{run['started_at']}`",
        f"- Duration: `{run['duration_ms']} ms`",
        f"- Dataset split: `{dataset['split']}`",
        f"- Cases: {dataset['total_cases']} total, "
        f"{dataset['answerable_cases']} answerable, "
        f"{dataset['unanswerable_cases']} unanswerable",
        f"- Retrieval: `{configuration['retrieval_method']}`",
        f"- Embedding model: `{configuration['embedding_model']}`",
        f"- Final top-k: `{configuration['top_k']}`",
        f"- Candidate limit: `{configuration.get('candidate_limit', 'not recorded')}`",
        "",
        "## Aggregate retrieval metrics",
        "",
        "Only answerable cases are included. Unanswerable cases remain in the "
        "diagnostics but cannot be scored as successful refusals until a "
        "retrieval relevance threshold exists.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric_name, value in aggregate.items():
        lines.append(f"| {metric_name} | {_format_metric(value)} |")

    lines.extend(["", "## Per-question diagnostics", ""])
    for case in report["cases"]:
        lines.extend(
            [
                f"### {case['id']}",
                "",
                f"- Document: `{case['filename']}`",
                f"- Question: {case['question']}",
                f"- Answerable: `{str(case['answerable']).lower()}`",
                f"- Evaluation identity: `{case['evaluation_identity_mode']}`",
                f"- Expected document hash: `{case['expected_document_content_hash']}`",
                f"- Expected chunks: `{case['expected_chunk_indices']}`",
                f"- Expected chunk hashes: `{case['expected_chunk_hashes']}`",
                f"- Tags: `{case['tags']}`",
                f"- Retrieval latency: `{case['elapsed_ms']} ms`",
            ]
        )
        if case["metrics"] is not None:
            metrics = case["metrics"]
            lines.extend(
                [
                    f"- First relevant rank: `{metrics['first_relevant_rank']}`",
                    f"- Reciprocal rank: `{_format_metric(metrics['reciprocal_rank'])}`",
                ]
            )
        else:
            lines.append(f"- Metric status: `{case['metric_status']}`")

        lines.extend(
            [
                "",
                "| Rank | Candidate rank | Chunk | Stable hash | Pages | Metric | Raw distance | Similarity | Preview |",
                "| ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |",
            ]
        )
        for retrieved in case["retrieved"]:
            preview = retrieved["preview"].replace("\n", " ").replace("|", "\\|")
            pages = (
                str(retrieved["page_start"])
                if retrieved["page_start"] == retrieved["page_end"]
                else f"{retrieved['page_start']}-{retrieved['page_end']}"
            )
            lines.append(
                f"| {retrieved['rank']} | {retrieved.get('candidate_rank')} | "
                f"{retrieved['chunk_index']} | "
                f"`{retrieved['chunk_content_hash'][:12]}` | {pages} | "
                f"{retrieved.get('distance_metric', 'unknown')} | "
                f"{retrieved['distance']:.6f} | "
                f"{retrieved.get('similarity', 0.0) or 0.0:.6f} | {preview} |"
            )
        if not case["retrieved"]:
            lines.append("| — | — | — | — | — | — | — | — | No chunks retrieved |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_reports(report: dict[str, Any], output_directory: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown reports using the same run ID."""

    output_directory.mkdir(parents=True, exist_ok=True)
    run_id = report["run"]["run_id"]
    json_path = output_directory / f"retrieval-{run_id}.json"
    markdown_path = output_directory / f"retrieval-{run_id}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path
