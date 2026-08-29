# Phase 6 lexical retrieval comparison

| Baseline | Hit@1 | Hit@5 | MRR | Recall@5 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense_phase5 | 0.2308 | 1.0000 | 0.5179 | 0.9231 | 0.6246 |
| lexical_pure_fts | 0.2308 | 0.7692 | 0.4559 | 0.7051 | 0.5058 |
| lexical_supplemented | 0.5385 | 1.0000 | 0.7244 | 0.8974 | 0.7485 |

## Dense versus selected lexical by category

| Category | Cases | Dense Hit@5 | Lexical Hit@5 | Dense MRR | Lexical MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| acronym | 5 | 1.0000 | 1.0000 | 0.5067 | 0.6500 |
| clause | 1 | 1.0000 | 1.0000 | 0.3333 | 0.3333 |
| date | 4 | 1.0000 | 1.0000 | 0.3833 | 0.7500 |
| exact-term | 3 | 1.0000 | 1.0000 | 0.3444 | 0.3056 |
| long-document | 13 | 1.0000 | 1.0000 | 0.5179 | 0.7244 |
| multiple-evidence | 8 | 1.0000 | 1.0000 | 0.4417 | 0.6146 |
| multiple-fact | 2 | 1.0000 | 1.0000 | 0.4167 | 0.7500 |
| name | 2 | 1.0000 | 1.0000 | 0.7500 | 1.0000 |
| number | 10 | 1.0000 | 1.0000 | 0.5200 | 0.7833 |
| paraphrase | 3 | 1.0000 | 1.0000 | 0.7778 | 1.0000 |
