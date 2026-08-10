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

## How the first version will work

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

The repository currently contains the project plan and the initial Python scaffold. The simulator, memory index, evaluation pipeline, and interface have not been implemented yet.

The full research plan is available in [docs/visual_memory_lab_plan.md](docs/visual_memory_lab_plan.md).

## Local setup

The project uses Python 3.13 and `uv`.

```powershell
uv sync
uv run python main.py
```

At this stage, the command only runs the scaffold entry point. It will be replaced with the first simulator workflow during implementation.
