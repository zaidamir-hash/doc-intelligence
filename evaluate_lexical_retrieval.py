"""Compare named dense, pure-FTS, and supplemented lexical baseline reports."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DENSE_REPORT = Path("evaluation/reports/phase5-selected")
DEFAULT_PURE_REPORT = Path("evaluation/reports/phase6-pure-fts-v2")
DEFAULT_SUPPLEMENT_REPORT = Path("evaluation/reports/phase6-supplement-v2")
DEFAULT_JSON_OUTPUT = Path(
    "evaluation/experiments/phase6_lexical_comparison.json"
)
DEFAULT_MARKDOWN_OUTPUT = Path(
    "evaluation/experiments/phase6_lexical_comparison.md"
)


def _load(path: Path) -> dict[str, Any]:
    if path.is_dir():
        candidates = list(path.glob("retrieval-*.json"))
        if not candidates:
            raise ValueError(f"No retrieval JSON reports found in {path}")
        path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"Missing report {path}. Run the documented baseline command first."
        ) from error


def _category_metrics(report: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in report["cases"]:
        if case["metrics"] is None:
            continue
        for tag in case["tags"]:
            values[tag].append(case["metrics"])

    return {
        tag: {
            "cases": len(metrics),
            "mrr": sum(item["reciprocal_rank"] for item in metrics) / len(metrics),
            "hit_at_5": sum(item["hit_at_k"]["5"] for item in metrics)
            / len(metrics),
            "recall_at_5": sum(item["recall_at_k"]["5"] for item in metrics)
            / len(metrics),
            "ndcg_at_5": sum(item["ndcg_at_k"]["5"] for item in metrics)
            / len(metrics),
        }
        for tag, metrics in sorted(values.items())
    }


def _case_summary(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        case["id"]: {
            "answerable": case["answerable"],
            "tags": case["tags"],
            "first_relevant_rank": (
                case["metrics"]["first_relevant_rank"] if case["metrics"] else None
            ),
            "reciprocal_rank": (
                case["metrics"]["reciprocal_rank"] if case["metrics"] else None
            ),
            "returned_chunks": len(case["retrieved"]),
            "top_score": (
                case["retrieved"][0].get("lexical_score")
                if case["retrieved"]
                else None
            ),
        }
        for case in report["cases"]
    }


def _baseline_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": report["run"]["run_id"],
        "configuration": report["configuration"],
        "aggregate_metrics": report["aggregate_metrics"],
        "unanswerable_metrics": report["unanswerable_metrics"],
        "category_metrics": _category_metrics(report),
        "cases": _case_summary(report),
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Phase 6 lexical retrieval comparison",
        "",
        "| Baseline | Hit@1 | Hit@5 | MRR | Recall@5 | nDCG@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, baseline in comparison["baselines"].items():
        metrics = baseline["aggregate_metrics"]
        lines.append(
            f"| {name} | {metrics['hit_at_1']:.4f} | "
            f"{metrics['hit_at_5']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['recall_at_5']:.4f} | {metrics['ndcg_at_5']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Dense versus selected lexical by category",
            "",
            "| Category | Cases | Dense Hit@5 | Lexical Hit@5 | Dense MRR | Lexical MRR |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    dense = comparison["baselines"]["dense_phase5"]["category_metrics"]
    lexical = comparison["baselines"]["lexical_supplemented"]["category_metrics"]
    for category in sorted(set(dense) & set(lexical)):
        lines.append(
            f"| {category} | {dense[category]['cases']} | "
            f"{dense[category]['hit_at_5']:.4f} | "
            f"{lexical[category]['hit_at_5']:.4f} | "
            f"{dense[category]['mrr']:.4f} | {lexical[category]['mrr']:.4f} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-report", type=Path, default=DEFAULT_DENSE_REPORT)
    parser.add_argument("--pure-report", type=Path, default=DEFAULT_PURE_REPORT)
    parser.add_argument(
        "--supplement-report", type=Path, default=DEFAULT_SUPPLEMENT_REPORT
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = {
        "dense_phase5": _load(args.dense_report),
        "lexical_pure_fts": _load(args.pure_report),
        "lexical_supplemented": _load(args.supplement_report),
    }
    dataset_names = {report["dataset"]["name"] for report in reports.values()}
    if len(dataset_names) != 1:
        raise ValueError("All compared reports must use the same dataset")

    comparison = {
        "dataset": next(iter(dataset_names)),
        "selection_decision": (
            "Keep exact identifier supplementation because it improves Hit@5 "
            "and MRR over pure FTS on the fixed dataset."
        ),
        "baselines": {
            name: _baseline_summary(report) for name, report in reports.items()
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    args.markdown_output.write_text(render_markdown(comparison), encoding="utf-8")
    print(f"JSON comparison: {args.json_output}")
    print(f"Markdown comparison: {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
