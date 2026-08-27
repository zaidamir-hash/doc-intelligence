"""Pure retrieval metric calculations with no database or model calls."""

from __future__ import annotations

import math
from dataclasses import dataclass


DEFAULT_CUTOFFS = (1, 3, 5, 10)


@dataclass(frozen=True)
class CaseMetrics:
    """Metrics for one answerable evaluation question."""

    first_relevant_rank: int | None
    reciprocal_rank: float
    hit_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]


def _validate_cutoffs(cutoffs: tuple[int, ...]) -> None:
    if not cutoffs or any(k <= 0 for k in cutoffs):
        raise ValueError("cutoffs must contain positive integers")
    if len(cutoffs) != len(set(cutoffs)):
        raise ValueError("cutoffs must not contain duplicates")


def _discounted_cumulative_gain(grades: list[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def calculate_case_metrics(
    retrieved_chunk_indices: list[int],
    relevance_grades: dict[int, int],
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
) -> CaseMetrics:
    """Calculate rank metrics for one answerable question."""

    _validate_cutoffs(cutoffs)
    if not relevance_grades:
        raise ValueError("answerable cases require at least one relevant chunk")
    if any(grade <= 0 for grade in relevance_grades.values()):
        raise ValueError("relevance grades must be positive")

    relevant_indices = set(relevance_grades)
    first_relevant_rank = next(
        (
            rank
            for rank, chunk_index in enumerate(retrieved_chunk_indices, start=1)
            if chunk_index in relevant_indices
        ),
        None,
    )
    reciprocal_rank = 0.0 if first_relevant_rank is None else 1 / first_relevant_rank

    hit_at_k: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}
    ndcg_at_k: dict[int, float] = {}
    ideal_grades = sorted(relevance_grades.values(), reverse=True)

    for k in cutoffs:
        retrieved_at_k = retrieved_chunk_indices[:k]
        relevant_retrieved = relevant_indices.intersection(retrieved_at_k)
        hit_at_k[k] = float(bool(relevant_retrieved))
        recall_at_k[k] = len(relevant_retrieved) / len(relevant_indices)

        seen_relevant_indices: set[int] = set()
        retrieved_grades: list[int] = []
        for chunk_index in retrieved_at_k:
            if chunk_index in seen_relevant_indices:
                retrieved_grades.append(0)
                continue
            retrieved_grades.append(relevance_grades.get(chunk_index, 0))
            if chunk_index in relevant_indices:
                seen_relevant_indices.add(chunk_index)
        dcg = _discounted_cumulative_gain(retrieved_grades)
        ideal_dcg = _discounted_cumulative_gain(ideal_grades[:k])
        ndcg_at_k[k] = 0.0 if ideal_dcg == 0 else dcg / ideal_dcg

    return CaseMetrics(
        first_relevant_rank=first_relevant_rank,
        reciprocal_rank=reciprocal_rank,
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        ndcg_at_k=ndcg_at_k,
    )


def average_metrics(
    case_metrics: list[CaseMetrics],
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
) -> dict[str, float]:
    """Macro-average metrics so every answerable question has equal weight."""

    _validate_cutoffs(cutoffs)
    if not case_metrics:
        return {
            "mrr": 0.0,
            **{f"hit_at_{k}": 0.0 for k in cutoffs},
            **{f"recall_at_{k}": 0.0 for k in cutoffs},
            **{f"ndcg_at_{k}": 0.0 for k in cutoffs},
        }

    question_count = len(case_metrics)
    aggregate = {
        "mrr": sum(metrics.reciprocal_rank for metrics in case_metrics)
        / question_count
    }
    for k in cutoffs:
        aggregate[f"hit_at_{k}"] = (
            sum(metrics.hit_at_k[k] for metrics in case_metrics) / question_count
        )
        aggregate[f"recall_at_{k}"] = (
            sum(metrics.recall_at_k[k] for metrics in case_metrics) / question_count
        )
        aggregate[f"ndcg_at_{k}"] = (
            sum(metrics.ndcg_at_k[k] for metrics in case_metrics) / question_count
        )
    return aggregate
