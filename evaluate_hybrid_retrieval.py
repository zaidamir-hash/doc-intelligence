"""Run the bounded Phase 7 RRF ablation and save its selected baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from database import SessionLocal
from embeddings import get_embeddings
from evaluation.dataset import EvaluationCase, EvaluationDataset, load_dataset
from evaluation.reporting import write_reports
from evaluation.runner import hybrid_candidate_to_retrieved, run_evaluation
from hybrid_retrieval import retrieve_hybrid_candidates


DEFAULT_DATASET = Path("evaluation/datasets/phase3.json")
DEFAULT_JSON_OUTPUT = Path("evaluation/experiments/phase7_hybrid_comparison.json")
DEFAULT_MARKDOWN_OUTPUT = Path("evaluation/experiments/phase7_hybrid_comparison.md")
DEFAULT_SELECTED_REPORT_DIRECTORY = Path("evaluation/reports/phase7-selected")
DEFAULT_DENSE_REPORT_DIRECTORY = Path("evaluation/reports/phase5-selected")
DEFAULT_LEXICAL_REPORT_DIRECTORY = Path("evaluation/reports/phase6-supplement-v2")
CALIBRATION_CASE_IDS = {
    "gov-ncpi-fy25-vs-fy24",
    "gov-policy-rate-reduction",
    "gov-climate-risk-fund-purpose",
    "gov-prism-plus-expansion",
    "gov-bitcoin-reserve-target-unanswerable",
}


@dataclass(frozen=True)
class FusionExperiment:
    name: str
    dense_candidate_k: int
    lexical_candidate_k: int
    rrf_k: int
    top_k: int = 10


EXPERIMENTS = (
    FusionExperiment("d10-l10-rrf10", 10, 10, 10),
    FusionExperiment("d10-l10-rrf30", 10, 10, 30),
    FusionExperiment("d10-l10-rrf60", 10, 10, 60),
    FusionExperiment("d10-l10-rrf100", 10, 10, 100),
    FusionExperiment("d20-l20-rrf10", 20, 20, 10),
    FusionExperiment("d20-l20-rrf60", 20, 20, 60),
    FusionExperiment("d30-l30-rrf10", 30, 30, 10),
    FusionExperiment("d30-l30-rrf30", 30, 30, 30),
    FusionExperiment("d30-l30-rrf60", 30, 30, 60),
    FusionExperiment("d30-l30-rrf100", 30, 30, 100),
    FusionExperiment("d30-l10-rrf60", 30, 10, 60),
    FusionExperiment("d10-l30-rrf60", 10, 30, 60),
)


def _subset(
    dataset: EvaluationDataset, case_ids: set[str], name: str, split: str
) -> EvaluationDataset:
    return replace(
        dataset,
        name=name,
        split=split,
        cases=tuple(case for case in dataset.cases if case.case_id in case_ids),
    )


def _run_experiment(
    experiment: FusionExperiment,
    dataset: EvaluationDataset,
    database,
    query_embeddings: dict[str, list[float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    union_by_case: dict[str, dict[str, float | int | bool]] = {}

    def retrieve(case: EvaluationCase, _top_k: int):
        result = retrieve_hybrid_candidates(
            case.question,
            database,
            case.filename,
            dense_candidate_k=experiment.dense_candidate_k,
            lexical_candidate_k=experiment.lexical_candidate_k,
            fused_top_k=experiment.top_k,
            rrf_k=experiment.rrf_k,
            document_content_hash=case.document_content_hash,
            embedding_function=lambda _text: query_embeddings[case.case_id],
        )
        union_hashes = {
            candidate.chunk_content_hash for candidate in result.dense_candidates
        } | {
            candidate.chunk_content_hash for candidate in result.lexical_candidates
        }
        expected = set(case.relevant_chunk_hashes)
        union_by_case[case.case_id] = {
            "answerable": case.answerable,
            "dense_candidates": len(result.dense_candidates),
            "lexical_candidates": len(result.lexical_candidates),
            "union_candidates": len(union_hashes),
            "union_hit": bool(expected & union_hashes) if case.answerable else False,
            "union_recall": (
                len(expected & union_hashes) / len(expected)
                if case.answerable and expected
                else 0.0
            ),
        }
        return [
            hybrid_candidate_to_retrieved(candidate)
            for candidate in result.fused_candidates
        ]

    report = run_evaluation(
        dataset,
        retrieve,
        top_k=experiment.top_k,
        configuration={
            "retrieval_method": "dense_lexical_rrf",
            **asdict(experiment),
            "candidate_limit": experiment.dense_candidate_k
            + experiment.lexical_candidate_k,
            "rrf_formula": "sum(1 / (rrf_k + source_rank))",
        },
    )
    answerable_union = [
        value for value in union_by_case.values() if value["answerable"]
    ]
    union_summary = {
        "hit_rate": sum(bool(value["union_hit"]) for value in answerable_union)
        / len(answerable_union),
        "mean_recall": sum(float(value["union_recall"]) for value in answerable_union)
        / len(answerable_union),
        "mean_unique_candidates": sum(
            int(value["union_candidates"]) for value in union_by_case.values()
        )
        / len(union_by_case),
        "cases": union_by_case,
    }
    return report, union_summary


def _summary(report: dict[str, Any], union: dict[str, Any]) -> dict[str, Any]:
    return {
        "aggregate_metrics": report["aggregate_metrics"],
        "union_metrics": {
            key: value for key, value in union.items() if key != "cases"
        },
        "duration_ms": report["run"]["duration_ms"],
    }


def _latest_report(directory: Path) -> dict[str, Any]:
    candidates = list(directory.glob("retrieval-*.json"))
    if not candidates:
        raise ValueError(f"No retrieval JSON reports found in {directory}")
    path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    return json.loads(path.read_text(encoding="utf-8"))


def _first_ranks(report: dict[str, Any]) -> dict[str, int | None]:
    return {
        case["id"]: (
            case["metrics"]["first_relevant_rank"] if case["metrics"] else None
        )
        for case in report["cases"]
    }


def _rank_comparison(
    hybrid_report: dict[str, Any], parent_report: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    parent_ranks = _first_ranks(parent_report)
    categories: dict[str, list[dict[str, Any]]] = {
        "helped": [],
        "unchanged": [],
        "harmed": [],
    }
    for case in hybrid_report["cases"]:
        if not case["answerable"]:
            continue
        hybrid_rank = case["metrics"]["first_relevant_rank"]
        parent_rank = parent_ranks[case["id"]]
        hybrid_value = hybrid_rank if hybrid_rank is not None else float("inf")
        parent_value = parent_rank if parent_rank is not None else float("inf")
        category = (
            "helped"
            if hybrid_value < parent_value
            else "harmed"
            if hybrid_value > parent_value
            else "unchanged"
        )
        categories[category].append(
            {
                "case_id": case["id"],
                "parent_rank": parent_rank,
                "hybrid_rank": hybrid_rank,
                "tags": case["tags"],
            }
        )
    return categories


def _category_metrics(report: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    tags: dict[str, list[dict[str, Any]]] = {}
    for case in report["cases"]:
        if case["metrics"] is None:
            continue
        for tag in case["tags"]:
            tags.setdefault(tag, []).append(case["metrics"])

    def at_five(metric: dict[Any, float]) -> float:
        return metric[5] if 5 in metric else metric["5"]

    return {
        tag: {
            "cases": len(values),
            "mrr": sum(value["reciprocal_rank"] for value in values) / len(values),
            "hit_at_5": sum(at_five(value["hit_at_k"]) for value in values)
            / len(values),
        }
        for tag, values in sorted(tags.items())
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Phase 7 hybrid RRF comparison",
        "",
        f"- Selected configuration: `{comparison['selected_configuration']}`",
        f"- Named hybrid run: `{comparison['named_hybrid_run_id']}`",
        "",
        "| Baseline | Hit@1 | Hit@5 | MRR | Recall@5 | nDCG@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in comparison["named_baselines"].items():
        lines.append(
            f"| {name} | {metrics['hit_at_1']:.4f} | "
            f"{metrics['hit_at_5']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['recall_at_5']:.4f} | {metrics['ndcg_at_5']:.4f} |"
        )
    selected = comparison["experiments"][comparison["selected_configuration"]]
    lines.extend(
        [
            "",
            "## Candidate union versus final fused ranking",
            "",
            f"- Held-out union Hit: `{selected['held_out']['union_metrics']['hit_rate']:.4f}`",
            f"- Held-out union mean recall: `{selected['held_out']['union_metrics']['mean_recall']:.4f}`",
            f"- Held-out final Hit@5: `{selected['held_out']['aggregate_metrics']['hit_at_5']:.4f}`",
            f"- Held-out final Recall@5: `{selected['held_out']['aggregate_metrics']['recall_at_5']:.4f}`",
            "",
            "## Rank changes",
            "",
        ]
    )
    for parent, categories in comparison["rank_analysis"].items():
        lines.append(
            f"- Versus {parent}: {len(categories['helped'])} helped, "
            f"{len(categories['unchanged'])} unchanged, "
            f"{len(categories['harmed'])} harmed."
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument(
        "--selected-report-dir",
        type=Path,
        default=DEFAULT_SELECTED_REPORT_DIRECTORY,
    )
    parser.add_argument(
        "--dense-report-dir", type=Path, default=DEFAULT_DENSE_REPORT_DIRECTORY
    )
    parser.add_argument(
        "--lexical-report-dir", type=Path, default=DEFAULT_LEXICAL_REPORT_DIRECTORY
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    all_ids = {case.case_id for case in dataset.cases}
    if not CALIBRATION_CASE_IDS < all_ids:
        raise ValueError("Phase 7 calibration IDs do not match the dataset")
    calibration = _subset(
        dataset, CALIBRATION_CASE_IDS, "lexis-phase7-calibration", "development"
    )
    held_out = _subset(
        dataset,
        all_ids - CALIBRATION_CASE_IDS,
        "lexis-phase7-held-out",
        "test",
    )
    query_embeddings = dict(
        zip(
            [case.case_id for case in dataset.cases],
            get_embeddings([case.question for case in dataset.cases]),
        )
    )

    database = SessionLocal()
    try:
        experiments: dict[str, dict[str, Any]] = {}
        for experiment in EXPERIMENTS:
            calibration_report, calibration_union = _run_experiment(
                experiment, calibration, database, query_embeddings
            )
            held_out_report, held_out_union = _run_experiment(
                experiment, held_out, database, query_embeddings
            )
            experiments[experiment.name] = {
                "configuration": asdict(experiment),
                "calibration": _summary(calibration_report, calibration_union),
                "held_out": _summary(held_out_report, held_out_union),
            }

        selected_name = max(
            experiments,
            key=lambda name: (
                experiments[name]["calibration"]["aggregate_metrics"]["hit_at_5"],
                experiments[name]["calibration"]["aggregate_metrics"]["mrr"],
                experiments[name]["calibration"]["aggregate_metrics"]["recall_at_5"],
                experiments[name]["calibration"]["aggregate_metrics"]["ndcg_at_5"],
                -(
                    experiments[name]["configuration"]["dense_candidate_k"]
                    + experiments[name]["configuration"]["lexical_candidate_k"]
                ),
            ),
        )
        selected = FusionExperiment(**experiments[selected_name]["configuration"])
        full_report, full_union = _run_experiment(
            selected, dataset, database, query_embeddings
        )
    finally:
        database.close()

    write_reports(full_report, args.selected_report_dir)
    dense_report = _latest_report(args.dense_report_dir)
    lexical_report = _latest_report(args.lexical_report_dir)
    comparison = {
        "dataset": dataset.name,
        "selection_rule": (
            "Use development Hit@5, MRR, Recall@5, then nDCG@5; prefer the "
            "smaller total candidate pool only on a complete metric tie."
        ),
        "selected_configuration": selected_name,
        "named_hybrid_run_id": full_report["run"]["run_id"],
        "experiments": experiments,
        "selected_full_union": full_union,
        "named_baselines": {
            "dense_phase5": dense_report["aggregate_metrics"],
            "lexical_phase6": lexical_report["aggregate_metrics"],
            "hybrid_phase7": full_report["aggregate_metrics"],
        },
        "category_metrics": {
            "dense_phase5": _category_metrics(dense_report),
            "lexical_phase6": _category_metrics(lexical_report),
            "hybrid_phase7": _category_metrics(full_report),
        },
        "rank_analysis": {
            "dense_phase5": _rank_comparison(full_report, dense_report),
            "lexical_phase6": _rank_comparison(full_report, lexical_report),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    args.markdown_output.write_text(render_markdown(comparison), encoding="utf-8")
    print(f"Selected configuration: {selected_name}")
    print(f"Named hybrid run: {full_report['run']['run_id']}")
    print(f"JSON comparison: {args.json_output}")
    print(f"Markdown comparison: {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
