# Phase 1: Simulator observations

## Goal

Generate reproducible MiniGrid inspection trajectories and save each egocentric
RGB observation with enough simulator state to support later retrieval and
failure analysis.

## In scope

- one controlled two-room environment;
- scripted inspection routes;
- seeded object placement between episodes;
- egocentric RGB frames and one overview image per episode;
- JSON manifests and observation records;
- a small generation command;
- reproducibility and contract tests.

## Not in scope

- CLIP embeddings or retrieval;
- Torch, FAISS, or transformers;
- random or learned policies;
- object movement within an episode;
- depth, video, point clouds, or a user interface.

## Done when

- the documented command generates ten episodes on Python 3.13;
- repeated runs with the same configuration produce identical records and images;
- object placement changes across episode seeds;
- every scene object is observed and the metadata matches simulator visibility;
- all tests pass and the generated artifact is manually inspected.

## Fixed decisions

- Branch: `phase/01-simulator-observations`.
- Default run: ten episodes, base seed 42, at most 100 actions per episode.
- Memory observations: 7x7 egocentric view rendered with 8-pixel tiles.
- Debug artifact: one full-map overview per episode.
- Logical simulator time equals the zero-based observation step.
- Generated data remains outside Git.

## Discovered changes

Record additions here before implementing them. Changes that do not support the
phase exit criteria are deferred to a later branch.

## Results

Completed on 2026-08-10.

The controlled environment contains two mirrored rooms connected by a narrow
corridor. Two identical red balls and one blue box have stable simulator
identities, while their positions vary deterministically between episode seeds.
The scripted inspection route uses 37 actions and records 38 observations per
episode, including the reset observation.

Validation evidence:

- `uv run python -m pytest -q`: 9 tests passed;
- the documented command generated 10 episodes and 380 observations;
- the artifact contains 380 egocentric frames and 10 overview images;
- all PNG files together use about 132 KiB;
- 314 observations contain at least one visible scene object;
- rerunning the same configuration produces identical JSON and PNG hashes;
- every scene object is observed, and each red ball is seen from at least two
  agent positions;
- full-map and egocentric images were manually inspected;
- generated trajectories are ignored by Git.

Known limits:

- the route is scripted rather than random or learned;
- objects move between episodes, not during an episode;
- simulator state supplies object identity; identical red balls are not visually
  distinguishable;
- no embeddings, retrieval, or task-level metrics exist yet.

Phase exit decision: passed. The phase branch was validated and merged into
`main`.

For a detailed explanation of the runtime path and the files produced by each
episode, see [01_code_flow.md](01_code_flow.md).
