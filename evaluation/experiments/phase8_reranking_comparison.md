# Phase 8 reranking comparison

- Selected cutoff: `2`
- Named reranked run: `81754b6e9c15`
- Candidate recall before reranking: `1.0000`
- Measured reranker latency: `104794.1 ms`
- Estimated API cost: `$0.011591`

| Baseline | Hit@1 | Hit@5 | MRR | Recall@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| hybrid_phase7 | 0.5385 | 1.0000 | 0.7115 | 0.9231 | 0.7542 |
| reranked_phase8 | 1.0000 | 1.0000 | 1.0000 | 0.9615 | 0.9579 |

## Rank changes

- Promoted: 6
- Unchanged: 7
- Demoted: 0

## Cutoff calibration

| Cutoff | Dev Hit@5 | Dev Recall@5 | Dev nDCG@5 | Unanswerable empty |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0000 | 0.8750 | 0.8832 | 0.0000 |
| 1 | 1.0000 | 0.8750 | 0.8832 | 1.0000 |
| 2 | 1.0000 | 0.8750 | 0.8832 | 1.0000 |
| 3 | 1.0000 | 0.8750 | 0.8832 | 1.0000 |
