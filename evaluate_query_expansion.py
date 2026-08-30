"""Compare Phase 9 expansion off/on for precise and vocabulary-mismatch queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PRECISE_DISABLED = Path("evaluation/reports/phase9-precise-disabled-batch5")
DEFAULT_PRECISE_ENABLED = Path("evaluation/reports/phase9-precise-enabled-batch5")
DEFAULT_VOCAB_DISABLED = Path("evaluation/reports/phase9-vocab-disabled-batch5")
DEFAULT_VOCAB_ENABLED = Path("evaluation/reports/phase9-vocab-enabled-batch5")
DEFAULT_JSON_OUTPUT = Path("evaluation/experiments/phase9_query_expansion_comparison.json")
DEFAULT_MARKDOWN_OUTPUT = Path("evaluation/experiments/phase9_query_expansion_comparison.md")


def _latest_report(directory: Path) -> dict[str, Any]:
    reports = list(directory.glob("retrieval-*.json"))
    if not reports:
        raise ValueError(f"No retrieval reports found in {directory}")
    path = max(reports, key=lambda item: item.stat().st_mtime)
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_rows(report: dict[str, Any], expanded: bool) -> dict[str, dict[str, Any]]:
    key = "expanded_candidates" if expanded else "original_candidates"
    details = report["query_expansion_summary"]["case_details"]
    return {case_id: value[key] for case_id, value in details.items()}


def _candidate_summary(report: dict[str, Any], expanded: bool) -> dict[str, float]:
    answerable_ids = {case["id"] for case in report["cases"] if case["answerable"]}
    rows = _candidate_rows(report, expanded)
    ranks = [rows[case_id]["first_relevant_rank"] for case_id in answerable_ids]
    return {
        "hit_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / len(ranks),
        "hit_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
        "hit_at_10": sum(rank is not None and rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(0.0 if rank is None else 1 / rank for rank in ranks) / len(ranks),
        "mean_recall_at_10": sum(rows[case_id]["recall"] for case_id in answerable_ids)
        / len(answerable_ids),
    }


def _candidate_rank_changes(
    disabled: dict[str, Any], enabled: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    before = _candidate_rows(disabled, expanded=False)
    after = _candidate_rows(enabled, expanded=True)
    groups: dict[str, list[dict[str, Any]]] = {
        "helped": [],
        "unchanged": [],
        "harmed": [],
    }
    for case in enabled["cases"]:
        if not case["answerable"]:
            continue
        old_rank = before[case["id"]]["first_relevant_rank"]
        new_rank = after[case["id"]]["first_relevant_rank"]
        old_value = old_rank if old_rank is not None else float("inf")
        new_value = new_rank if new_rank is not None else float("inf")
        group = (
            "helped"
            if new_value < old_value
            else "harmed"
            if new_value > old_value
            else "unchanged"
        )
        expansion = enabled["query_expansion_summary"]["case_details"][case["id"]][
            "query_expansion"
        ]
        groups[group].append(
            {
                "case_id": case["id"],
                "original_rank": old_rank,
                "expanded_rank": new_rank,
                "original_query": expansion["original_query"],
                "generated_query": expansion.get("generated_query"),
                "expanded_query": expansion["expanded_query"],
                "used_expansion": expansion["used_expansion"],
                "fallback_reason": expansion["fallback_reason"],
            }
        )
    return groups


def _cost(report: dict[str, Any]) -> dict[str, float | int]:
    expansion = report["query_expansion_summary"]
    reranker = report["reranker_summary"]
    return {
        "duration_ms": report["run"]["duration_ms"],
        "expansion_requests": expansion["requests"],
        "expansion_input_tokens": expansion["input_tokens"],
        "expansion_output_tokens": expansion["output_tokens"],
        "expansion_cost_usd": expansion["estimated_cost_usd"],
        "reranker_requests": reranker["requests"],
        "reranker_cost_usd": reranker["estimated_cost_usd"],
        "total_model_cost_usd": (
            expansion["estimated_cost_usd"] + reranker["estimated_cost_usd"]
        ),
        "expansion_used_cases": expansion["used_cases"],
        "expansion_fallback_cases": expansion["fallback_cases"],
        "reranker_fallback_cases": reranker["fallback_cases"],
    }


def render_markdown(result: dict[str, Any]) -> str:
    precise = result["precise_queries"]
    vocab = result["vocabulary_mismatch"]
    lines = [
        "# Phase 9 query expansion comparison",
        "",
        "## Fixed precise-query benchmark",
        "",
        "| Mode | Candidate Hit@5 | Candidate MRR | Final Hit@5 | Final MRR | Final Recall@5 | Final nDCG@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("disabled", "enabled"):
        candidate = precise[mode]["candidate_metrics"]
        final = precise[mode]["final_metrics"]
        lines.append(
            f"| {mode} | {candidate['hit_at_5']:.4f} | {candidate['mrr']:.4f} | "
            f"{final['hit_at_5']:.4f} | {final['mrr']:.4f} | "
            f"{final['recall_at_5']:.4f} | {final['ndcg_at_5']:.4f} |"
        )
    precise_changes = precise["candidate_rank_changes"]
    lines.extend(
        [
            "",
            f"Candidate changes: {len(precise_changes['helped'])} helped, "
            f"{len(precise_changes['unchanged'])} unchanged, "
            f"{len(precise_changes['harmed'])} harmed.",
            "",
            "## Development vocabulary-mismatch stress set",
            "",
            "| Mode | Candidate Hit@5 | Candidate MRR | Final Hit@5 | Final MRR |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in ("disabled", "enabled"):
        candidate = vocab[mode]["candidate_metrics"]
        final = vocab[mode]["final_metrics"]
        lines.append(
            f"| {mode} | {candidate['hit_at_5']:.4f} | {candidate['mrr']:.4f} | "
            f"{final['hit_at_5']:.4f} | {final['mrr']:.4f} |"
        )
    vocab_changes = vocab["candidate_rank_changes"]
    lines.extend(
        [
            "",
            f"Candidate changes: {len(vocab_changes['helped'])} helped, "
            f"{len(vocab_changes['unchanged'])} unchanged, "
            f"{len(vocab_changes['harmed'])} harmed.",
            "",
            "## Expansion overhead on the fixed benchmark",
            "",
            f"- Accepted expansions: `{precise['enabled']['cost']['expansion_used_cases']}/14`",
            f"- Safe fallbacks: `{precise['enabled']['cost']['expansion_fallback_cases']}`",
            f"- Expansion-only cost: `${precise['enabled']['cost']['expansion_cost_usd']:.6f}`",
            f"- Disabled duration: `{precise['disabled']['cost']['duration_ms']:.1f} ms`",
            f"- Enabled duration: `{precise['enabled']['cost']['duration_ms']:.1f} ms`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precise-disabled", type=Path, default=DEFAULT_PRECISE_DISABLED)
    parser.add_argument("--precise-enabled", type=Path, default=DEFAULT_PRECISE_ENABLED)
    parser.add_argument("--vocab-disabled", type=Path, default=DEFAULT_VOCAB_DISABLED)
    parser.add_argument("--vocab-enabled", type=Path, default=DEFAULT_VOCAB_ENABLED)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args()


def _comparison(disabled: dict[str, Any], enabled: dict[str, Any]) -> dict[str, Any]:
    return {
        "disabled": {
            "run_id": disabled["run"]["run_id"],
            "candidate_metrics": _candidate_summary(disabled, expanded=False),
            "final_metrics": disabled["aggregate_metrics"],
            "cost": _cost(disabled),
        },
        "enabled": {
            "run_id": enabled["run"]["run_id"],
            "candidate_metrics": _candidate_summary(enabled, expanded=True),
            "final_metrics": enabled["aggregate_metrics"],
            "cost": _cost(enabled),
        },
        "candidate_rank_changes": _candidate_rank_changes(disabled, enabled),
    }


def main() -> int:
    args = parse_args()
    precise_disabled = _latest_report(args.precise_disabled)
    precise_enabled = _latest_report(args.precise_enabled)
    vocab_disabled = _latest_report(args.vocab_disabled)
    vocab_enabled = _latest_report(args.vocab_enabled)
    result = {
        "method": (
            "Compare expansion off/on with identical dense=20, lexical=20, "
            "RRF k=10, rerank k=10, reranker batch=5 settings. Candidate metrics "
            "isolate retrieval/fusion; final metrics include hosted reranking."
        ),
        "precise_queries": _comparison(precise_disabled, precise_enabled),
        "vocabulary_mismatch": _comparison(vocab_disabled, vocab_enabled),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown_output.write_text(render_markdown(result), encoding="utf-8")
    print(f"JSON comparison: {args.json_output}")
    print(f"Markdown comparison: {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
