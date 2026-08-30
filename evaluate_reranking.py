"""Analyze the Phase 8 named run without making additional model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.metrics import average_metrics, calculate_case_metrics


DEFAULT_RERANKED_REPORT_DIRECTORY = Path("evaluation/reports/phase8-selected")
DEFAULT_HYBRID_REPORT_DIRECTORY = Path("evaluation/reports/phase7-selected")
DEFAULT_JSON_OUTPUT = Path("evaluation/experiments/phase8_reranking_comparison.json")
DEFAULT_MARKDOWN_OUTPUT = Path("evaluation/experiments/phase8_reranking_comparison.md")
CALIBRATION_CASE_IDS = {
    "gov-ncpi-fy25-vs-fy24",
    "gov-policy-rate-reduction",
    "gov-climate-risk-fund-purpose",
    "gov-prism-plus-expansion",
    "gov-bitcoin-reserve-target-unanswerable",
}


def _latest_report(directory: Path) -> dict[str, Any]:
    paths = list(directory.glob("retrieval-*.json"))
    if not paths:
        raise ValueError(f"No retrieval reports found in {directory}")
    return json.loads(max(paths, key=lambda path: path.stat().st_mtime).read_text())


def _grades(case: dict[str, Any]) -> dict[str, int]:
    return {
        content_hash: case["relevance_grades"][str(index)]
        for index, content_hash in zip(
            case["expected_chunk_indices"], case["expected_chunk_hashes"]
        )
    }


def _cutoff_metrics(
    report: dict[str, Any], case_ids: set[str], cutoff: int, final_k: int
) -> dict[str, Any]:
    scored = []
    unanswerable_total = 0
    unanswerable_empty = 0
    selected_counts = []
    for case in report["cases"]:
        if case["id"] not in case_ids:
            continue
        selected = [
            item
            for item in case["retrieved"]
            if item["reranker_score"] >= cutoff
        ][:final_k]
        selected_counts.append(len(selected))
        if case["answerable"]:
            scored.append(
                calculate_case_metrics(
                    [item["chunk_content_hash"] for item in selected],
                    _grades(case),
                    (1, 3, 5),
                )
            )
        else:
            unanswerable_total += 1
            unanswerable_empty += int(not selected)
    return {
        "aggregate_metrics": average_metrics(scored, (1, 3, 5)),
        "unanswerable_empty_rate": (
            unanswerable_empty / unanswerable_total
            if unanswerable_total
            else None
        ),
        "mean_selected_evidence": sum(selected_counts) / len(selected_counts),
    }


def _first_rank(case: dict[str, Any]) -> int | None:
    return case["metrics"]["first_relevant_rank"] if case["metrics"] else None


def _rank_analysis(
    hybrid: dict[str, Any], reranked: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    hybrid_by_id = {case["id"]: case for case in hybrid["cases"]}
    groups: dict[str, list[dict[str, Any]]] = {
        "promoted": [],
        "unchanged": [],
        "demoted": [],
    }
    for case in reranked["cases"]:
        if not case["answerable"]:
            continue
        before = _first_rank(hybrid_by_id[case["id"]])
        after = _first_rank(case)
        before_value = before if before is not None else float("inf")
        after_value = after if after is not None else float("inf")
        group = (
            "promoted"
            if after_value < before_value
            else "demoted"
            if after_value > before_value
            else "unchanged"
        )
        groups[group].append(
            {
                "case_id": case["id"],
                "hybrid_rank": before,
                "reranked_rank": after,
                "top_output_original_fused_rank": case["retrieved"][0][
                    "original_fused_rank"
                ],
                "top_output_score": case["retrieved"][0]["reranker_score"],
            }
        )
    return groups


def render_markdown(result: dict[str, Any]) -> str:
    baseline = result["named_baselines"]
    ranks = result["rank_analysis"]
    lines = [
        "# Phase 8 reranking comparison",
        "",
        f"- Selected cutoff: `{result['selected_relevance_cutoff']}`",
        f"- Named reranked run: `{result['named_reranked_run_id']}`",
        f"- Candidate recall before reranking: `{result['candidate_pool']['mean_recall']:.4f}`",
        f"- Measured reranker latency: `{result['cost_and_latency']['total_reranker_latency_ms']:.1f} ms`",
        f"- Estimated API cost: `${result['cost_and_latency']['estimated_cost_usd']:.6f}`",
        "",
        "| Baseline | Hit@1 | Hit@5 | MRR | Recall@5 | nDCG@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in baseline.items():
        lines.append(
            f"| {name} | {metrics['hit_at_1']:.4f} | {metrics['hit_at_5']:.4f} | "
            f"{metrics['mrr']:.4f} | {metrics['recall_at_5']:.4f} | "
            f"{metrics['ndcg_at_5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Rank changes",
            "",
            f"- Promoted: {len(ranks['promoted'])}",
            f"- Unchanged: {len(ranks['unchanged'])}",
            f"- Demoted: {len(ranks['demoted'])}",
            "",
            "## Cutoff calibration",
            "",
            "| Cutoff | Dev Hit@5 | Dev Recall@5 | Dev nDCG@5 | Unanswerable empty |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cutoff, summary in result["cutoff_experiments"].items():
        metrics = summary["development"]["aggregate_metrics"]
        empty = summary["development"]["unanswerable_empty_rate"]
        lines.append(
            f"| {cutoff} | {metrics['hit_at_5']:.4f} | "
            f"{metrics['recall_at_5']:.4f} | {metrics['ndcg_at_5']:.4f} | "
            f"{empty:.4f} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reranked-report-dir", type=Path, default=DEFAULT_RERANKED_REPORT_DIRECTORY
    )
    parser.add_argument(
        "--hybrid-report-dir", type=Path, default=DEFAULT_HYBRID_REPORT_DIRECTORY
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reranked = _latest_report(args.reranked_report_dir)
    hybrid = _latest_report(args.hybrid_report_dir)
    all_ids = {case["id"] for case in reranked["cases"]}
    held_out_ids = all_ids - CALIBRATION_CASE_IDS
    cutoff_experiments = {
        str(cutoff): {
            "development": _cutoff_metrics(
                reranked, CALIBRATION_CASE_IDS, cutoff, 5
            ),
            "held_out": _cutoff_metrics(reranked, held_out_ids, cutoff, 5),
        }
        for cutoff in range(4)
    }
    selected_cutoff = max(
        range(4),
        key=lambda cutoff: (
            cutoff_experiments[str(cutoff)]["development"][
                "unanswerable_empty_rate"
            ],
            cutoff_experiments[str(cutoff)]["development"]["aggregate_metrics"][
                "hit_at_5"
            ],
            cutoff_experiments[str(cutoff)]["development"]["aggregate_metrics"][
                "recall_at_5"
            ],
            cutoff_experiments[str(cutoff)]["development"]["aggregate_metrics"][
                "ndcg_at_5"
            ],
            -abs(cutoff - 2),
        ),
    )
    details = reranked["reranker_summary"]["case_details"]
    answerable_details = [
        details[case["id"]] for case in reranked["cases"] if case["answerable"]
    ]
    result = {
        "selection_rule": (
            "On the fixed development subset, maximize unanswerable empty rate, "
            "then Hit@5, Recall@5, and nDCG@5; on a complete tie prefer score "
            "2 because the fixed rubric defines it as useful supporting evidence."
        ),
        "selected_relevance_cutoff": selected_cutoff,
        "named_reranked_run_id": reranked["run"]["run_id"],
        "reranker_model": reranked["configuration"]["reranker_model"],
        "candidate_pool": {
            "size": reranked["configuration"]["rerank_k"],
            "hit_rate": sum(detail["candidate_hit"] for detail in answerable_details)
            / len(answerable_details),
            "mean_recall": sum(
                detail["candidate_recall"] for detail in answerable_details
            )
            / len(answerable_details),
        },
        "named_baselines": {
            "hybrid_phase7": hybrid["aggregate_metrics"],
            "reranked_phase8": reranked["aggregate_metrics"],
        },
        "cutoff_experiments": cutoff_experiments,
        "rank_analysis": _rank_analysis(hybrid, reranked),
        "cost_and_latency": {
            "requests": reranked["reranker_summary"]["requests"],
            "input_tokens": reranked["reranker_summary"]["input_tokens"],
            "output_tokens": reranked["reranker_summary"]["output_tokens"],
            "estimated_cost_usd": reranked["reranker_summary"][
                "estimated_cost_usd"
            ],
            "total_run_duration_ms": reranked["run"]["duration_ms"],
            "total_reranker_latency_ms": sum(
                detail["reranker_latency_ms"] for detail in details.values()
            ),
            "mean_reranker_latency_ms": sum(
                detail["reranker_latency_ms"] for detail in details.values()
            )
            / len(details),
            "fallback_cases": reranked["reranker_summary"]["fallback_cases"],
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown_output.write_text(render_markdown(result), encoding="utf-8")
    print(f"Selected relevance cutoff: {selected_cutoff}")
    print(f"JSON comparison: {args.json_output}")
    print(f"Markdown comparison: {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
