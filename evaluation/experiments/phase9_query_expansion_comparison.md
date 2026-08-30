# Phase 9 query expansion comparison

## Fixed precise-query benchmark

| Mode | Candidate Hit@5 | Candidate MRR | Final Hit@5 | Final MRR | Final Recall@5 | Final nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| disabled | 1.0000 | 0.7115 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| enabled | 1.0000 | 0.6859 | 1.0000 | 1.0000 | 1.0000 | 0.9938 |

Candidate changes: 1 helped, 11 unchanged, 1 harmed.

## Development vocabulary-mismatch stress set

| Mode | Candidate Hit@5 | Candidate MRR | Final Hit@5 | Final MRR |
| --- | ---: | ---: | ---: | ---: |
| disabled | 0.5000 | 0.2604 | 1.0000 | 0.5833 |
| enabled | 0.5000 | 0.2708 | 0.7500 | 0.7917 |

Candidate changes: 1 helped, 3 unchanged, 0 harmed.

## Expansion overhead on the fixed benchmark

- Accepted expansions: `12/14`
- Safe fallbacks: `2`
- Expansion-only cost: `$0.001134`
- Disabled duration: `138563.5 ms`
- Enabled duration: `183681.7 ms`
