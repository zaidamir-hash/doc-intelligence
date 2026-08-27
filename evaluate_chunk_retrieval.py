"""Compare eligible Phase 3 chunk configurations with exact dense retrieval."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from chunking import ChunkingConfig, StructuredChunk, chunk_extraction
from embeddings import client
from evaluate_chunking import CONFIGURATIONS, matching_chunks
from evaluation.dataset import EvaluationDataset, load_dataset
from evaluation.runner import RetrievedChunk, run_evaluation
from pdf_processing import extract_pdf


BASE_QUESTIONS = Path("evaluation/datasets/phase2.json")
EVIDENCE_REQUIREMENTS = Path("evaluation/datasets/phase3_evidence.json")
DEFAULT_OUTPUT_DIRECTORY = Path("evaluation/reports")
DEFAULT_DATASET_OUTPUT = Path("evaluation/datasets/phase3.json")
ELIGIBLE_CONFIGURATIONS = ("small", "balanced", "large")
EMBEDDING_MODEL = "text-embedding-3-small"


def _embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts[start : start + batch_size],
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        embeddings.extend(item.embedding for item in ordered)
        print(f"Embedded {min(start + batch_size, len(texts))}/{len(texts)}")
    if len(embeddings) != len(texts):
        raise RuntimeError("embedding response count did not match input count")
    return embeddings


def _label_dataset(
    base: EvaluationDataset,
    chunks: list[StructuredChunk],
    evidence_by_id: dict[str, dict[str, object]],
    name: str,
) -> EvaluationDataset:
    labelled_cases = []
    for case in base.cases:
        if not case.answerable:
            labelled_cases.append(case)
            continue
        indices = matching_chunks(chunks, evidence_by_id[case.case_id])
        if not indices:
            raise ValueError(f"No preserved evidence for {case.case_id} in {name}")
        labelled_cases.append(
            replace(
                case,
                relevant_chunk_indices=tuple(indices),
                relevance_grades={index: 1 for index in indices},
                notes=(
                    f"Phase 3 {name} labels generated from the fixed, "
                    "pre-retrieval page-and-pattern evidence requirements."
                ),
            )
        )
    return replace(
        base,
        name=f"lexis-phase3-{name}",
        description=(
            f"Fixed Phase 1 questions labelled against the Phase 3 {name} "
            "configuration before ranking."
        ),
        cases=tuple(labelled_cases),
    )


def _retriever(
    chunks: list[StructuredChunk],
    chunk_embeddings: list[list[float]],
    question_embeddings: dict[str, list[float]],
):
    def retrieve(case, top_k: int) -> list[RetrievedChunk]:
        question_embedding = question_embeddings[case.case_id]
        ranked = sorted(
            enumerate(chunk_embeddings),
            key=lambda item: sum(
                (left - right) ** 2
                for left, right in zip(question_embedding, item[1])
            ),
        )
        return [
            RetrievedChunk(
                chunk_id=index,
                chunk_index=index,
                filename=case.filename,
                content=chunks[index].content,
                distance=math.sqrt(
                    sum(
                        (left - right) ** 2
                        for left, right in zip(question_embedding, embedding)
                    )
                ),
            )
            for index, embedding in ranked[:top_k]
        ]

    return retrieve


def _select(results: dict[str, dict[str, object]]) -> str:
    return max(
        results,
        key=lambda name: (
            results[name]["aggregate_metrics"]["hit_at_5"],
            results[name]["aggregate_metrics"]["mrr"],
            results[name]["aggregate_metrics"]["ndcg_at_5"],
            -CONFIGURATIONS[name].max_tokens,
        ),
    )


def _dataset_document(dataset: EvaluationDataset) -> dict[str, object]:
    """Convert the selected immutable dataset to the public JSON schema."""

    cases = []
    for case in dataset.cases:
        document = asdict(case)
        document["id"] = document.pop("case_id")
        document["relevant_chunk_indices"] = list(case.relevant_chunk_indices)
        document["relevance_grades"] = {
            str(index): grade for index, grade in case.relevance_grades.items()
        }
        document["tags"] = list(case.tags)
        cases.append(document)
    return {
        "schema_version": dataset.schema_version,
        "name": dataset.name,
        "description": dataset.description,
        "split": dataset.split,
        "cases": cases,
    }


def render_markdown(comparison: dict[str, object]) -> str:
    lines = [
        "# Phase 3 dense retrieval comparison",
        "",
        f"- Selected configuration: `{comparison['selected_configuration']}`",
        f"- Embedding model: `{EMBEDDING_MODEL}`",
        "- Search: exact in-memory L2 over every chunk",
        "- Labels: fixed page-and-pattern requirements created before ranking",
        "",
        "| Configuration | Max/overlap | Hit@1 | Hit@5 | Hit@10 | MRR | "
        "Recall@10 | nDCG@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, report in comparison["configurations"].items():
        metrics = report["aggregate_metrics"]
        config = report["chunking_configuration"]
        lines.append(
            f"| {name} | {config['max_tokens']}/{config['overlap_tokens']} | "
            f"{metrics['hit_at_1']:.4f} | {metrics['hit_at_5']:.4f} | "
            f"{metrics['hit_at_10']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['recall_at_10']:.4f} | {metrics['ndcg_at_5']:.4f} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--dataset-output", type=Path, default=DEFAULT_DATASET_OUTPUT
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extraction = extract_pdf(args.pdf.read_bytes())
    base = load_dataset(BASE_QUESTIONS)
    evidence_document = json.loads(
        EVIDENCE_REQUIREMENTS.read_text(encoding="utf-8")
    )
    evidence_by_id = {case["id"]: case for case in evidence_document["cases"]}
    question_embeddings = dict(
        zip(
            [case.case_id for case in base.cases],
            _embed_texts([case.question for case in base.cases]),
        )
    )

    results: dict[str, dict[str, object]] = {}
    labelled_datasets: dict[str, EvaluationDataset] = {}
    for name in ELIGIBLE_CONFIGURATIONS:
        config = CONFIGURATIONS[name]
        chunks = chunk_extraction(extraction, config)
        dataset = _label_dataset(base, chunks, evidence_by_id, name)
        labelled_datasets[name] = dataset
        embeddings = _embed_texts([chunk.content for chunk in chunks])
        report = run_evaluation(
            dataset,
            _retriever(chunks, embeddings, question_embeddings),
            top_k=10,
        )
        report["configuration"]["retrieval_method"] = "exact_in_memory_l2"
        report["configuration"]["candidate_limit"] = len(chunks)
        report["chunking_configuration"] = {
            "max_tokens": config.max_tokens,
            "overlap_tokens": config.overlap_tokens,
            "min_chunk_tokens": config.min_chunk_tokens,
            "encoding_name": config.encoding_name,
            "version": config.version,
            "chunk_count": len(chunks),
        }
        results[name] = report

    selected = _select(results)
    comparison = {
        "selection_rule": (
            "Maximize Hit@5, then MRR, then nDCG@5; prefer fewer maximum "
            "tokens only if retrieval metrics tie."
        ),
        "selected_configuration": selected,
        "configurations": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "chunk-retrieval-comparison.json"
    markdown_path = args.output_dir / "chunk-retrieval-comparison.md"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(comparison), encoding="utf-8")
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_output.write_text(
        json.dumps(_dataset_document(labelled_datasets[selected]), indent=2),
        encoding="utf-8",
    )
    print(f"Selected configuration: {comparison['selected_configuration']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print(f"Selected dataset: {args.dataset_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
