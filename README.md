# Visual Memory Lab

Visual Memory Lab is a small research project about finding useful evidence in a history of images.

The basic problem is simple: a camera may record thousands of observations over time, but storing those images is not enough. When someone asks about an earlier event, the system must retrieve the right image even if the viewpoint, lighting, surroundings, or appearance of an object has changed.

The project begins in MiniGrid, where an agent moves through rooms and corridors and records what it sees. The simulator gives us exact information about each observation, including the agent's position, direction, visible objects, episode, and timestep. This makes it possible to check whether a retrieved image is genuinely relevant rather than merely similar-looking.

## A real-world example

Imagine a maintenance technician making weekly rounds through a factory. A body camera or phone records images during each inspection. Several weeks later, a leak is found near a blue valve, and the technician wants to know:

> When was rust first visible around the blue valve beside the pressure gauge?

A normal image search may return other blue valves because they look similar. A useful visual memory system should retrieve the correct valve, in the correct part of the factory, from the inspection when the rust first appeared. It should still work if the technician approached from another direction, the lighting was different, or equipment partly blocked the view.

The same idea could support building inspections, construction progress reviews, field service, environmental surveys, or accessibility tools that help people recall where an object was last seen.

MiniGrid is used as a controlled test bench for this problem. It provides repeatable experiments and reliable ground truth without requiring a physical robot or collecting personal images.

## Research question

> Can a visual memory system retrieve the right past observation when viewpoint, time, occlusion, or scene appearance changes?

This is different from asking whether two images look alike. The most visually similar image may come from the wrong room, the wrong time, or the wrong object. The useful memory is the one that contains the evidence needed for the task.

## How the project will work

1. Generate reproducible navigation trajectories in MiniGrid.
2. Save each RGB observation with simulator metadata.
3. Create CLIP embeddings for the stored images.
4. Store the embeddings in a simple flat index.
5. Retrieve past observations using a text query or another image.
6. Compare the results with simulator ground truth.

Each retrieved result will include the image, similarity score, episode, timestep, agent pose, nearby actions, and visible objects.

## Evaluation

The project will measure more than cosine similarity:

- **Event hit@k:** whether one of the top results contains the target object or event.
- **Episode hit@k:** whether a result comes from the correct episode.
- **Temporal error:** how far the retrieved timestep is from the target event.
- **Pose error:** how far the retrieved position is from the relevant location.

These measurements will help separate visual resemblance from task-relevant memory.

## Failure atlas

The project will also document cases where retrieval fails. Initial categories include:

- viewpoint confusion;
- perceptual aliasing between similar-looking places;
- partial or complete occlusion;
- retrieval from the wrong moment;
- stale memories after an object moves;
- confusion between similar objects.

Each case will show the query, the retrieved observation, the simulator ground truth, and a short explanation of what went wrong.

## Current status

Phases 1 and 2 are implemented. The repository can generate reproducible
MiniGrid inspection trajectories and search them with frozen CLIP ViT-B/32:

- egocentric RGB frames showing what the agent sees;
- a full-map overview for each episode;
- agent position, direction, action, seed, and logical time;
- stable object identities and visible-object metadata;
- deterministic JSON manifests for later retrieval experiments;
- persistent normalized image embeddings;
- exact text-to-image and image-to-image retrieval;
- episode filtering, action context, and JSON query output.

Task-relevant evaluation, the failure atlas, and the local interface belong to
later phases and have not been implemented yet.

The full research plan is available in [docs/visual_memory_lab_plan.md](docs/visual_memory_lab_plan.md).
The Phase 2 design and real-model results are documented in
[docs/phases/02_visual_memory.md](docs/phases/02_visual_memory.md).

## Local setup

The project uses Python 3.13 and `uv`.

```powershell
uv sync
uv run visual-memory-lab generate `
  --episodes 10 `
  --seed 42 `
  --max-steps 100 `
  --output data/trajectories/phase-01-demo

uv run visual-memory-lab index `
  --input data/trajectories/phase-01-demo `
  --output outputs/phase-02-clip-index

uv run visual-memory-lab query `
  --index outputs/phase-02-clip-index `
  --text "a blue box" `
  --top-k 5
```

The default research run contains 10 episodes and 380 observations. Generated
images, manifests, embeddings, and model weights remain local and are ignored
by Git. Choose a new output directory for each generated run or index; commands
will not replace an existing non-empty directory.

Run the test suite with:

```powershell
uv run python -m pytest -q
```
