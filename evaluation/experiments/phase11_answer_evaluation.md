# Answer Evaluation — lexis-phase11-answer-quality-v1

## Run

- Run ID: `01903932da6b`
- Dataset SHA-256: `153419c6e7661ae9020f7b541460d897d02e6227443c823033daff925e5a0613`
- Dataset split: `test`
- Git commit: `2baf5d43a9f5a3e6bda91152923699d6ecec8826`
- Deterministic grader: `phase11-deterministic-gold-v1`
- Model-assisted judge: `not used`

## Separate outcomes

- Retrieval success rate (answerable live-pipeline cases only): `1.0000`
- Answer success rate (all cases): `0.8333`

| Answer dimension | Score |
| --- | ---: |
| correctness | 0.8333 |
| faithfulness | 0.8333 |
| citation_accuracy | 0.8333 |
| completeness | 0.8000 |
| refusal_accuracy | 0.8333 |
| unsupported_claim_rate | 0.0000 |

The scores use fixed gold facts/evidence and deterministic checks. No model judge decides acceptance.

## Failure attribution

- `generation`: 1
- `none`: 5

## Per-case diagnostics

### gov-public-debt-ratio

- Type / mode: `answerable` / `pipeline`
- Question: How did Pakistan's public debt-to-GDP ratio change from FY24 to FY25?
- Reference: It rose from 67.7 percent in FY24 to 70.8 percent in FY25.
- Answer: Pakistan's public debt-to-GDP ratio rose to 70.8 percent during FY25. [S1] The public debt-to-GDP ratio was 67.7 percent in FY24. [S1] The increase in the public debt-to-GDP ratio in FY25 was due to slower growth in nominal GDP and a slight PKR depreciation. [S1]
- Status: `answered`
- Retrieval success: `True`
- Answer success: `true`
- Failure stage: `none`
- Evidence pages: `[(28, 28), (27, 27)]`
- Exact evidence records in JSON: `2`
- Correctness / faithfulness / citations: `1.0000` / `1.0000` / `1.0000`
- Completeness: `1.0000`
- Refusal accuracy / unsupported-claim rate: `1.0000` / `0.0000`

### gov-january-2025-projections

- Type / mode: `answerable` / `pipeline`
- Question: After the January 2025 revision, what inflation projection range did the MPC adopt, and what real GDP growth range did it maintain?
- Reference: The MPC revised inflation to 5.5-7.5 percent and maintained real GDP growth at 2.5-3.5 percent.
- Answer: The MPC revised down the inflation projection range to 5.5 - 7.5 percent in January 2025. [S1] The MPC maintained its real GDP growth projection in the range of 2.5 to 3.5 percent. [S1]
- Status: `answered`
- Retrieval success: `True`
- Answer success: `true`
- Failure stage: `none`
- Evidence pages: `[(45, 45), (45, 45), (45, 45)]`
- Exact evidence records in JSON: `3`
- Correctness / faithfulness / citations: `1.0000` / `1.0000` / `1.0000`
- Completeness: `1.0000`
- Refusal accuracy / unsupported-claim rate: `1.0000` / `0.0000`

### gov-statutory-submission

- Type / mode: `answerable` / `pipeline`
- Question: Under which statutory provision was the Governor's Annual Report submitted to Parliament?
- Reference: It was submitted under Section 39(1) of the State Bank of Pakistan Act, 1956.
- Answer: I could not produce a safely validated answer from the selected evidence.
- Status: `insufficient_evidence`
- Retrieval success: `True`
- Answer success: `false`
- Failure stage: `generation`
- Evidence pages: `[(8, 10), (10, 12), (64, 64)]`
- Exact evidence records in JSON: `3`
- Correctness / faithfulness / citations: `0.0000` / `0.0000` / `0.0000`
- Completeness: `0.0000`
- Refusal accuracy / unsupported-claim rate: `0.0000` / `0.0000`

### gov-bitcoin-reserve-target-unanswerable

- Type / mode: `unanswerable` / `pipeline`
- Question: What Bitcoin reserve target did SBP set for FY25?
- Reference: None
- Answer: I cannot answer this reliably from the selected document evidence.
- Status: `insufficient_evidence`
- Retrieval success: `None`
- Answer success: `true`
- Failure stage: `none`
- Evidence pages: `[]`
- Exact evidence records in JSON: `0`
- Correctness / faithfulness / citations: `1.0000` / `1.0000` / `1.0000`
- Completeness: `not applicable`
- Refusal accuracy / unsupported-claim rate: `1.0000` / `0.0000`

### partial-policy-rate-and-bitcoin

- Type / mode: `partially_answerable` / `synthetic_context`
- Question: How much was the policy rate reduced, and what Bitcoin reserve target was announced?
- Reference: The policy rate was reduced by 1,100 basis points; the evidence contains no Bitcoin reserve target.
- Answer: The policy rate was reduced by 1,100 basis points. [S1] Limitation: The selected evidence does not fully support or consistently resolve every requested part.
- Status: `partially_answered`
- Retrieval success: `None`
- Answer success: `true`
- Failure stage: `none`
- Evidence pages: `[(1, 1)]`
- Exact evidence records in JSON: `1`
- Correctness / faithfulness / citations: `1.0000` / `1.0000` / `1.0000`
- Completeness: `1.0000`
- Refusal accuracy / unsupported-claim rate: `1.0000` / `0.0000`

### conflicting-inflation-rates

- Type / mode: `conflicting_evidence` / `synthetic_context`
- Question: What inflation rate did the report state?
- Reference: One passage says 10 percent and another says 12 percent; the conflict must be reported rather than silently resolved.
- Answer: The report states that inflation was 10 percent. [S1] The report states that inflation was 12 percent. [S2] Limitation: The selected evidence does not fully support or consistently resolve every requested part.
- Status: `partially_answered`
- Retrieval success: `None`
- Answer success: `true`
- Failure stage: `none`
- Evidence pages: `[(1, 1), (2, 2)]`
- Exact evidence records in JSON: `2`
- Correctness / faithfulness / citations: `1.0000` / `1.0000` / `1.0000`
- Completeness: `1.0000`
- Refusal accuracy / unsupported-claim rate: `1.0000` / `0.0000`

## Reproducibility and interpretation

- Dataset: `evaluation\datasets\phase11_answers.json`
- Source retrieval labels: `evaluation/datasets/phase3.json`
- Configuration: `{"answer_model": "gpt-4o-mini-2024-07-18", "dense_candidate_k": 20, "embedding_model": "text-embedding-3-small", "expansion_model": "gpt-4o-mini-2024-07-18", "final_evidence_k": 5, "grader_version": "phase11-deterministic-gold-v1", "lexical_candidate_k": 20, "model_assisted_judge": null, "relevance_cutoff": 2, "rerank_k": 10, "reranker_batch_size": 5, "reranker_model": "gpt-4o-mini-2024-07-18", "rrf_k": 10}`
- The local Git-ignored JSON report preserves full generation evidence, raw answers, citations, raw deterministic grader checks, and stage traces.
- Synthetic cases bypass retrieval intentionally; they isolate partial/conflicting generation and therefore have no retrieval-success score.
