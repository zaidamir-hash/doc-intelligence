# Phase 7 hybrid RRF comparison

- Selected configuration: `d20-l20-rrf10`
- Named hybrid run: `8d6a5317d321`

| Baseline | Hit@1 | Hit@5 | MRR | Recall@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense_phase5 | 0.2308 | 1.0000 | 0.5179 | 0.9231 | 0.6246 |
| lexical_phase6 | 0.5385 | 1.0000 | 0.7244 | 0.8974 | 0.7485 |
| hybrid_phase7 | 0.5385 | 1.0000 | 0.7115 | 0.9231 | 0.7542 |

## Candidate union versus final fused ranking

- Held-out union Hit: `1.0000`
- Held-out union mean recall: `1.0000`
- Held-out final Hit@5: `1.0000`
- Held-out final Recall@5: `0.9444`

## Rank changes

- Versus dense_phase5: 7 helped, 6 unchanged, 0 harmed.
- Versus lexical_phase6: 1 helped, 9 unchanged, 3 harmed.
