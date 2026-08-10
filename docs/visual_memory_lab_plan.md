# Visual Memory Lab

## Project idea

Build a simulator-based visual memory system for physical-AI research.

The system generates trajectories in a simulator, stores visual observations, retrieves relevant past observations using text or image queries, and analyzes when visual memory succeeds or fails.

The central question is:

> Can an embodied agent retrieve the right past observation when viewpoint, time, occlusion, or scene appearance changes?

This project does not require a physical robot. All observations are generated locally from a simulator, so no personal images or videos are needed.

## Phase 1 - Simulator and observation contract

Status: implemented and merged.

Use MiniGrid with Gymnasium to generate simple navigation trajectories.

Each observation record contains:

- `episode_id`;
- `step`;
- RGB image path;
- agent position and orientation;
- action taken;
- visible objects;
- simulator timestamp;
- environment seed.

The first environment will contain:

- rooms and corridors;
- repeated-looking locations;
- movable or differently positioned objects;
- partial occlusions;
- viewpoint changes.

Depth and point clouds remain future extensions, not part of the first version.

## Phase 2 - Visual memory

Status: implemented and validated. The implementation uses the pinned
`openai/clip-vit-base-patch32` checkpoint, a persistent NumPy artifact, and
exact in-memory cosine search. See
[`phases/02_visual_memory.md`](phases/02_visual_memory.md) for the contract and
acceptance results.

Build the simplest memory system:

```text
trajectory observations
        |
        v
CLIP image embeddings
        |
        v
Flat in-memory index
        |
        v
query image or text
        |
        v
top-k past observations
```

The query interface supports:

- text query: "Where did the red object appear?";
- image query: upload or select an observation;
- optional episode filtering.

Each result returns:

- similarity score;
- episode and timestep;
- retrieved image;
- agent pose;
- nearby actions;
- visible objects.

Use Flat search first. Add one compressed index only after the memory behavior works.

## Phase 3 - Task-relevant retrieval

Do not judge memory only by cosine similarity.

For each query, simulator state provides ground truth. Measure:

- **event hit@k:** does one of the retrieved observations contain the target object/event?
- **episode hit@k:** did the system retrieve an observation from the correct episode?
- **temporal error:** how far is the retrieved timestep from the target event?
- **pose error:** how far is the retrieved agent position from the relevant location?

The important comparison is:

```text
visual similarity retrieval
versus
task-relevant memory retrieval
```

A visually similar frame is not necessarily the frame useful for navigation or reasoning.

## Phase 4 - Failure atlas

Create a small failure taxonomy:

| Failure | Example |
|---|---|
| Viewpoint confusion | Same room looks different from another direction |
| Perceptual aliasing | Two locations look visually similar |
| Occlusion failure | The target object is hidden in the query frame |
| Temporal boundary error | The correct event is nearby but outside the retrieved window |
| Stale memory | The object moved after the stored observation |
| Identity confusion | Similar objects are mistaken for the same object |

For every failure, store:

- query observation;
- retrieved observation;
- simulator ground truth;
- failure category;
- short explanation;
- corrected or improved retrieval result, if available.

The first explanations will be mechanical using simulator state. A VLM may be added later only for visual case descriptions.

## Phase 5 - Minimal interface

Build a simple local interface with:

- environment/episode selector;
- text or image query;
- top-k control;
- retrieved observations displayed horizontally;
- timestep and pose information;
- failure label when ground truth is available.

No agents, chat orchestration, generation, or complicated backend are needed initially.

## Public project structure

Use a small `uv` layout:

```text
visual-memory-lab/
|-- src/visual_memory_lab/
|   |-- environment.py
|   |-- encoder.py
|   |-- observations.py
|   |-- memory.py
|   |-- evaluation.py
|   `-- cli.py
|-- data/
|   `-- .gitkeep
|-- outputs/
|   `-- .gitkeep
|-- docs/
|   `-- visual_memory_lab_plan.md
|-- tests/
|-- pyproject.toml
`-- README.md
```

The repository will contain code and small examples only. Generated trajectories and local images remain ignored.

## Initial dependencies

Keep the first implementation small:

- `minigrid`;
- `gymnasium`;
- `torch`;
- `transformers`;
- `numpy`;
- `pillow`;
- `opencv-python-headless` only if video export is needed.

The Phase 2 corpus is small enough for exact NumPy search. FAISS or another
compressed index should be added only after exact retrieval is understood and
there is a scale-related reason to introduce it.

Do not add LangChain, FastAPI, Qdrant, diffusion models, or video models in the first phase.

## Definition of done

The first useful milestone is complete when:

- a simulator generates reproducible trajectories;
- observations and state metadata are saved;
- text and image queries retrieve past observations;
- the UI displays top-k memory results with timestep and pose;
- event hit@k and temporal error are computed from simulator ground truth;
- at least four failure cases are documented;
- the README explains why visual similarity and task relevance can disagree;
- the project runs with one documented `uv` command.

## Why this is the next project

This moves from:

```text
retrieval benchmark
```

to:

```text
perception -> memory -> evidence retrieval -> embodied reasoning
```

That is closer to physical AI while preserving the strongest idea from the retrieval failure atlas: explaining why visual retrieval fails.
