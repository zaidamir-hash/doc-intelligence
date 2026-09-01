# Phase 11 — Answer evaluation

## Purpose

Retrieval success and answer success are different questions. A relevant chunk
can be found while generation omits a required fact, cites the wrong evidence,
refuses incorrectly, or fails operationally. Conversely, an answer cannot be
credited as a generation success when the required evidence never reached the
answer model.

Phase 11 therefore evaluates this sequence explicitly:

```text
gold evidence exists in index
        ↓
candidate retrieval
        ↓
fusion
        ↓
reranking and relevance cutoff
        ↓
generation evidence
        ↓
grounded answer and citations
        ↓
separate answer-quality dimensions
```

The implementation reuses Phase 1's versioned-dataset/report pattern and Phase
10's structured claims, supporting quotes, page-aware citations, exact
generation context, and safe fallback behavior. No Phase 12 API/frontend work
is included.

## Bounded gold dataset

`evaluation/datasets/phase11_answers.json` is a versioned test set containing
six cases:

- three answerable questions run through the complete indexed pipeline;
- one unanswerable question run through the complete indexed pipeline;
- one controlled partially answerable case;
- one controlled conflicting-evidence case.

The real cases reuse reference answers, document hashes, and stable chunk
hashes from the fixed human-labelled Phase 3 retrieval dataset. The answer
dataset adds human-readable expected facts, required/optional status, accepted
term alternatives, allowed evidence hashes, expected answer status, and case
type. These labels were fixed before the Phase 11 run.

The partial and conflict cases use small fixed passages. They intentionally
bypass retrieval so they isolate generation behavior. They do not receive a
retrieval-success score and cannot inflate the retrieval metric.

The dataset loader in `evaluation/answer_dataset.py` rejects, among other
problems:

- unsupported schema versions, splits, case types, run modes, or statuses;
- malformed document/chunk SHA-256 identities;
- duplicate case or expected-claim IDs;
- unanswerable cases that nevertheless require claims or evidence;
- synthetic cases without fixed evidence; and
- claim evidence that is not included in the case's expected evidence.

## Rubric and separate metrics

The evaluator is deterministic and versioned as
`phase11-deterministic-gold-v1`. It does not use an LLM judge.

### Correctness

Correctness is the fraction of displayed factual claims that match at least
one fixed gold fact. Matching requires every explicit term group for that fact;
alternatives such as `Section 39(1)` and `Section 39 1` are visible in the
dataset rather than hidden in a judge prompt.

This is deliberately conservative. A true extra claim outside the bounded
gold labels can be marked incorrect, so the raw answer and matches must remain
available for human inspection.

### Faithfulness

Faithfulness is the fraction of displayed claims that pass the Phase 10
grounding validator. The validator checks supplied source IDs, supporting
quotes, quote presence in the exact passage, claim-term coverage, and
clause-level coverage.

### Citation accuracy

Every claim-to-source link must:

1. map to evidence actually sent to generation;
2. map to the same stable chunk identity returned in the citation;
3. contain the claim and checked supporting quote; and
4. use one of the allowed human-labelled evidence hashes when a gold fact has
   an evidence restriction.

Citation accuracy is therefore not inferred merely from a `[S1]` marker.

### Completeness

Completeness is the fraction of required gold facts represented in the final
claims. Optional accepted facts do not raise or lower completeness.

This makes an important distinction possible: an answer containing one correct
requested value but omitting the second can have perfect correctness and only
half completeness.

### Refusal accuracy

The expected behavior depends on the case type:

- answerable: `answered`, with claims and no evidence-gap reason;
- unanswerable: `insufficient_evidence`, with no claims or citations;
- partially answerable: `partially_answered`, with a supported claim and an
  explicit limitation;
- conflicting evidence: `partially_answered`, preserving both supported sides
  and the unresolved limitation.

### Unsupported-claim rate

This is the number of displayed claims that fail grounding validation divided
by all displayed claims. Claims rejected before display are recorded separately
in `rejected_pre_display_claims`; they do not count as user-visible unsupported
claims.

An empty refusal therefore has a zero unsupported-claim rate, but it can still
fail correctness, completeness, or refusal accuracy for an answerable case.

## Retrieval and answer success remain separate

For an answerable real case, retrieval success means at least one expected gold
chunk survived into the final fused candidate list before reranking. Answer
success requires every applicable answer dimension to equal `1.0`, no displayed
unsupported claim, and no generation fallback.

The report never substitutes answer success for retrieval success. Synthetic
generation cases use `null` retrieval success because no retrieval occurred.

## Failure attribution

The evaluator records the earliest observable failed stage for every
unsuccessful case:

| Check | Failure classification |
| --- | --- |
| Matching ready document is absent | `extraction` |
| Stable gold chunk is absent from the indexed document | `chunking` |
| Gold chunk is absent from all raw dense/lexical candidates | `candidate_retrieval` |
| Gold chunk was retrieved but disappeared from the final fused list | `fusion` |
| Gold chunk was fused but did not pass the relevance cutoff | `reranking` |
| Gold chunk passed the cutoff but did not enter model context | `evidence_selection` |
| Evidence reached context but answer metrics failed | `generation` |

For an unanswerable case, generation evidence indicates a relevance/reranking
failure; otherwise an incorrect answer/refusal is a generation failure.
Controlled synthetic cases can only fail at generation because earlier stages
were deliberately bypassed.

## Exact evidence and reproducibility

The locally generated JSON report preserves, per question:

- question, reference answer, expected claims, and stable gold evidence;
- original and expanded dense/lexical candidate identities and ranks;
- final fused and reranked candidates with scores/rationales;
- the complete `passage_text` of every evidence item sent to generation;
- raw answer, structured claims, citations, supporting quotes, and fallback;
- raw deterministic grader checks and all separate scores;
- retrieval success, answer success, and failure stage; and
- expansion, reranker, and answer usage/diagnostics.

Run metadata includes dataset path and SHA-256, split, Git commit, Python and
platform versions, all model snapshots, retrieval limits, RRF constant,
reranker batch size, relevance cutoff, grader version, timestamps, and run ID.
There is no model-judge prompt or raw model-judge output because no model judge
was used. Raw deterministic grader outputs are preserved instead.

Because that raw JSON contains complete passages from the uploaded document,
`evaluation/experiments/.gitignore` deliberately excludes it from Git. It
remains available locally for exact regrading. The versioned Markdown report
contains only the human-readable summary, page ranges, metrics, and
configuration—not the full retrieved passages.

Reproduce the bounded run from the repository root with PostgreSQL and the
matching indexed document available:

```powershell
.venv\Scripts\python.exe evaluate_answers.py
```

The command writes:

- `evaluation/experiments/phase11_answer_evaluation.json` locally (Git-ignored
  because it contains exact uploaded-document passages);
- `evaluation/experiments/phase11_answer_evaluation.md`.

The deterministic grader can be reapplied to the exact saved model outputs
without another retrieval/model run:

```powershell
.venv\Scripts\python.exe evaluate_answers.py `
  --regrade-existing evaluation/experiments/phase11_answer_evaluation.json
```

This is useful when reviewing grader code because it prevents a new stochastic
model output from being mistaken for a grader change.

## Recorded result

Run `01903932da6b` produced:

| Measure | Result |
| --- | ---: |
| Retrieval success (three applicable live answerable cases) | 1.0000 |
| Answer success (all six cases) | 0.8333 (5/6) |
| Correctness | 0.8333 |
| Faithfulness | 0.8333 |
| Citation accuracy | 0.8333 |
| Completeness | 0.8000 |
| Refusal accuracy | 0.8333 |
| Unsupported-claim rate | 0.0000 |

The public-debt, projections, unanswerable, partial, and conflicting cases all
passed. The statutory-submission case had perfect upstream retrieval:

- both labelled chunks existed in the index;
- labelled evidence appeared in raw candidates and the fused list;
- the evidence received reranker score `3`;
- the labelled pages entered generation context.

The answer-model call then timed out. Phase 10 correctly returned its safe
fallback with no factual claims or citations, and Phase 11 classified the case
as `generation`. The result is intentionally retained rather than rerunning
until a perfect score appears. This demonstrates why operational generation
failure must not be reported as retrieval failure—or hidden by retrieval's
1.0000 score.

Two earlier attempts did not produce reports: the first was blocked by the
workspace network sandbox and the permitted retry was reset during the first
embedding request. They are operational attempts, not scored experiment runs.

## Tests

Focused tests cover:

- dataset versioning and all required case categories;
- explicit gold-term matching;
- correctness versus completeness separation;
- faithfulness versus gold citation-evidence separation;
- unsupported displayed-claim measurement;
- unanswerable refusal behavior; and
- every failure-attribution branch from extraction through generation.

## Limitations and follow-up

- The six-case set is bounded and cannot establish production-level answer
  quality or hallucination rates.
- Deterministic term groups are transparent but do not understand every valid
  paraphrase. Conservative false negatives require human inspection and future
  label expansion—not silent threshold tuning on this fixed test set.
- Structural faithfulness is stronger than checking citation syntax, but it is
  not a full natural-language entailment proof.
- Hosted model and network behavior can vary. The pinned model versions, raw
  outputs, timestamps, and fallback diagnostics make that variation visible.
- The test split must not be repeatedly tuned against. Future rubric or model
  experiments should use a development/validation set and keep this report as a
  regression record.
- Phase 12 may display these diagnostics and citations, but it must not alter
  the Phase 11 scoring methodology merely to improve reported results.
