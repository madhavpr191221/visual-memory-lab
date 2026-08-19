# Phase 7: Technician-Style Task Benchmark

## Purpose

This phase checks whether the current office-memory system is useful for
practical inspection questions. It is a self-guided benchmark: I author the
questions, define the expected evidence, run the system without revealing the
answer, and review the result.

It is not an external user study and it does not claim to measure technician
performance. It measures whether the system retrieves inspectable evidence and
handles uncertainty safely.

## Question set

The manifest contains 24 questions in four groups:

- place retrieval;
- visible scene content;
- earlier logical visits;
- object and evidence boundaries.

Each question has a concrete source observation, dataset, category,
answerability label, rationale, and an objective expected-evidence rule. The
labels are `supported`,
`supported_with_limits`, `requires_manual_review`, and `unsupported`.

The benchmark deliberately includes questions the current system must not answer
definitively, such as whether a chair moved or whether a missing detection proves
absence.

## Evaluation

The evaluator runs source-anchored CLIP retrieval for the 7-Scenes questions.
ETH questions inspect the localization, RGB-D, and association records tied to
their source frame. It does not call a cloud model and does not add a
depth-estimation model.

```powershell
uv run visual-memory-lab evaluate-technician-benchmark `
  --output outputs/phase7/technician-benchmark
```

The output contains `summary.json`, `per_question.jsonl`, and category-level
metrics. Reported measures include evidence hit@k, mean reciprocal rank, zone
and visit matches, ETH artifact presence, boundary-question count, and safe
abstention. A single overall accuracy is intentionally avoided: a wrong place,
wrong visit, missing evidence, and safe abstention are different failures.

The earlier smoke-test value of `0.8` is not a benchmark result. It measured
only four successful zone checks out of five and left most questions unscored.

## UI

Open `/app/tasks` to choose a question and move into the technician workflow.
The page shows the expected handling only as a benchmark reference; the actual
evidence must be inspected in `/app`.

The research view at `/research/evaluation` remains the place for retrieval and
failure metrics. This separation keeps the application language practical and
the diagnostics inspectable.

## Scope boundary

This phase uses RGB, recorded poses, semantic zones, detector masks, and existing
ETH RGB-D evidence. It does not train a new model, add learned depth, establish
persistent object identity, or prove that an object moved. Depth perception can
be studied separately before a later phase decides whether it solves a measured
failure.
