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

Pending implementation and validation.
