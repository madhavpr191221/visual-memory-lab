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

## Phase 3 - Real-image place memory

Status: implemented and validated on all 10,000 RGB frames from the 7-Scenes
Office scene. See
[`phases/03_real_image_place_memory.md`](phases/03_real_image_place_memory.md)
for the exact split, VLM-assisted zone protocol, pose-grounded metrics, actual
results, and technician interpretation.

For the complete Phase 3 to Phase 4 handoff, including exactly how the frozen
zones become UI cards and search-time agreement counts, see
[`phases/03_04_real_office_visual_memory_system.md`](phases/03_04_real_office_visual_memory_system.md).

Do not judge memory only by cosine similarity.

The implemented real-image study uses held-out camera sequences and measured
camera poses. It reports:

- strict and relaxed pose hit@1/5/10;
- query coverage and pose-oracle coverage ceiling;
- top-1 translation and orientation error;
- fixed-seed random baselines and per-sequence results;
- stride-10 sensitivity;
- semantic zone hit@k and precision@k for frozen VLM-assisted labels.

The important comparison is:

```text
visual similarity retrieval
versus
task-relevant memory retrieval
```

A visually similar frame is not necessarily the frame from the correct physical
work area. Simulator event and temporal metrics remain useful future controlled
experiments; they are not mixed into the completed real-image pose benchmark.

## Phase 4 - Office memory explorer (implemented)

Turn the real-image Office experiment into a local evidence-first application.
The interface supports text and uploaded-image retrieval, interpretable zone
agreement, and inspection of the frozen evaluation results. It also makes the
failure taxonomy browsable:

| Failure | Example |
|---|---|
| Viewpoint confusion | Same room looks different from another direction |
| Perceptual aliasing | Two locations look visually similar |
| Occlusion failure | The target object is hidden in the query frame |
| Temporal boundary error | The correct event is nearby but outside the retrieved window |
| Stale memory | The object moved after the stored observation |
| Identity confusion | Similar objects are mistaken for the same object |

For every held-out query, expose:

- query observation;
- retrieved observation;
- camera-pose ground truth;
- mechanical outcome tags;
- all ten retrieved memories with similarity and pose errors.

The main retrieval path stays local and displays images before any generated
answer. Optional VLM judgment is a separate confirmed action over one to five
user-selected public frames. It uses a strict response schema, validates
evidence citations, caches public text judgments, and never caches an uploaded
query image. The detailed design is in
[phases/04_office_memory_explorer.md](phases/04_office_memory_explorer.md).
The combined Phase 3 and 4 system guide is the preferred end-to-end reference.

## Phase 5 - Cross-traversal revisit memory (implemented)

Before comparing scene state, retrieve a comparable observation from a
designated reference traversal. The implemented 7-Scenes protocol evaluates
all 24 held-out-query to training-memory traversal pairs and reports:

- strict and relaxed pose coverage inside each reference traversal;
- traversal-conditioned CLIP hit@1/5/10;
- random-within-traversal baselines;
- covered-query and all-query-target rates;
- micro, macro-pair, per-pair, and pose-error results.

The system treats sequence IDs as traversal identifiers, not timestamps. See
[`phases/05_cross_traversal_memory.md`](phases/05_cross_traversal_memory.md) for
the complete protocol and measured results.

## Phase 6A - Controlled 3D state-change baseline (implemented)

Use the public ETH ASL Change Detection Office dataset to compare four aligned
real RGB-D/3D observations. The implemented phase adds:

- a 96-frame browsable RGB audit;
- deterministic 2 cm voxel representations;
- bidirectional point-to-point and point-to-plane residuals;
- 2/5/10 cm threshold sensitivity;
- connected geometric candidates across all six observation pairs;
- a strict VLM-supported pseudo-reference over the largest candidates;
- a failure atlas that exposes fragmentation and reconstruction ambiguity.

The acceptance run produced 917 raw candidate clusters. The VLM reviewed the
72 largest, marked 53 supported, 15 uncertain, and 4 unsupported, and admitted
47 medium/high-confidence candidates to the pseudo-reference. These are not
human ground-truth accuracy measurements. See
[`phases/06a_controlled_3d_change_baseline.md`](phases/06a_controlled_3d_change_baseline.md).

## Phase 6B - Object-aware change memory (in progress)

The high-level subphase roadmap is documented in
[`phases/06_phase6_overview.md`](phases/06_phase6_overview.md).

### Phase 6B1 - Frozen RGB object localization (implemented)

Before associating or training object identities, establish a visible baseline
that automatically locates movable office objects. Phase 6B1 samples 96
pose-diverse keyframes per ETH observation, uses frozen Grounding DINO to detect
chairs, waste bins, and boxes, and uses frozen SAM 2.1 to create masks. The
Objects UI exposes predictions, confidence filters, masks, model provenance,
and an optional 48-frame VLM pseudo-audit. It makes no cross-visit identity or
movement claim. See
[`phases/06b1_object_localization.md`](phases/06b1_object_localization.md).

Later Phase 6B subphases will establish an honest labelled evaluation slice,
project verified masks into the shared 3D coordinate frame, associate likely
object identities across visits, and compare object state. A trained component
should be introduced only against a measured failure from those baselines,
using a separately labelled or synthetic training source where necessary.

Do not infer temporal-change capabilities from 7-Scenes Office: its manifests do
not contain real calendar time, verified traversal chronology, or recorded
maintenance events.

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
