# Visual Memory Lab

Visual Memory Lab is a small research project about finding useful evidence in a history of images. It now works with both controlled MiniGrid trajectories and the publicly available 7-Scenes Office research dataset.

Phase 6A also evaluates controlled real-scene change using four RGB-D/3D
observations from ETH Zurich's public Change Detection Office dataset. The
dataset itself is not redistributed.

Phase 6B1 adds automatic RGB object localization over 384 dense office
keyframes. Frozen Grounding DINO predictions and SAM 2.1 masks replace the
hand-drawn boxes used in the earlier change showcase.

> **Research-use notice:** 7-Scenes is provided by Microsoft Research for
> non-commercial use. This repository is a personal research and portfolio
> demonstration for hiring-manager review, not a commercial deployment. The
> original RGB-D dataset, embeddings, and model weights are not redistributed.
> See [Third-Party Notices](THIRD_PARTY_NOTICES.md).

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

Phases 1 through 5, Phase 6A, and Phase 6B1 are implemented. The repository can generate reproducible
MiniGrid inspection trajectories, search them with frozen CLIP ViT-B/32, and
evaluate real-image place memory on 10,000 7-Scenes Office frames:

- egocentric RGB frames showing what the agent sees;
- a full-map overview for each episode;
- agent position, direction, action, seed, and logical time;
- stable object identities and visible-object metadata;
- deterministic JSON manifests for later retrieval experiments;
- persistent normalized image embeddings;
- exact text-to-image and image-to-image retrieval;
- episode filtering, action context, and JSON query output.
- official 6,000-memory / 4,000-query Office split;
- camera-pose hit@1/5/10, coverage, pose error, random baseline, and stride-10 sensitivity;
- seven cached VLM-assisted semantic place zones and 21 frozen text queries;
- a local React/TypeScript office-memory explorer backed by FastAPI;
- text and uploaded-image retrieval with visible evidence and zone agreement;
- evaluation, failure, query-detail, and place-zone browsers;
- optional, explicitly confirmed VLM analysis over selected licensed-dataset evidence.
- cross-traversal retrieval over 24 designated source-target traversal pairs;
- separate measurement of reference-route coverage and retrieval quality.
- automatic chair, waste-bin, and box localization over four ETH Office visits;
- inspectable detector boxes, segmentation masks, confidence filters, and a
  sampled VLM pseudo-audit in the Objects UI.

On the strict 0.25 m / 30 degree criterion, 65.4% of held-out queries have a
qualifying stored memory. Among those covered queries, exact CLIP retrieval
reaches 32.2% hit@1, 48.3% hit@5, and 56.3% hit@10. The semantic-zone benchmark
reaches 71.4% macro hit@10.

Phase 5 evaluates 24,000 query-target combinations. Strict pose coverage is
17.1%; among covered combinations, traversal-conditioned CLIP reaches 36.8%
Hit@1, 52.1% Hit@5, and 60.6% Hit@10, compared with 3.6%, 15.1%, and 27.1% for
random selection within the same traversal.

The interface keeps ordinary retrieval local. Cloud analysis is a separate,
confirmed action and remains disabled when no OpenAI API key is configured.

The full research plan is available in [docs/visual_memory_lab_plan.md](docs/visual_memory_lab_plan.md).
The longer-term product and research thesis is documented in
[Visual Memory Lab: Research and Application Direction](docs/visual_memory_lab_research_and_application_direction.md).
The Phase 2 design and real-model results are documented in
[docs/phases/02_visual_memory.md](docs/phases/02_visual_memory.md).
The best single guide to the real-image system is
[Phases 3 and 4: The Real-Office Visual Memory System](docs/phases/03_04_real_office_visual_memory_system.md).
It follows the complete path from the public dataset and VLM-assisted zone
creation to local retrieval, zone voting, evidence analysis, and the React UI.
The original [Phase 3 methodology](docs/phases/03_real_image_place_memory.md)
and [Phase 4 interface notes](docs/phases/04_office_memory_explorer.md) remain as
phase-specific references.
The retrieval-and-alignment bridge is documented in
[Phase 5: Cross-Traversal Revisit Memory](docs/phases/05_cross_traversal_memory.md).

For a user-facing catalogue of supported, planned, and unsupported questions,
see [What Can a User Ask the Visual Memory Lab?](docs/user_question_catalog.md).

[Phase 6A: Controlled 3D State-Change Baseline](docs/phases/06a_controlled_3d_change_baseline.md).

Phase 6A extracts 96 browsable RGB frames, compares all six aligned mesh pairs,
and produces inspectable geometric candidates. Its VLM output is explicitly a
pseudo-reference rather than human ground truth. The acceptance run produced
917 raw clusters; 72 large candidates were reviewed, and 47 medium/high-confidence
candidates entered the pseudo-reference. The compact frozen counts are in
[`artifacts/phase6a/summary.json`](artifacts/phase6a/summary.json).

The automatic object-localization baseline is documented in
[Phase 6B1: Automatic Object Localization](docs/phases/06b1_object_localization.md).
The broader object-aware memory design, including the RGB-D and 3D mathematics,
is documented in [Phase 6B: Object-Aware Change Memory](docs/phases/06b_object_aware_change_memory.md).
The high-level roadmap for all Phase 6 subphases is in
[Phase 6: Object-Aware Physical Change Memory](docs/phases/06_phase6_overview.md).
Its CUDA acceptance run processed 384 keyframes and retained 1,417 predictions:
515 chairs, 477 waste bins, and 425 boxes. These are predictions, not correct
object counts. The completed VLM audit reviewed all 48 requested frames and
found substantial false positives: 79 supported, 15 uncertain, and 71
unsupported predictions. Frozen counts and the pseudo-audit boundary are in
[`artifacts/phase6b1/summary.json`](artifacts/phase6b1/summary.json).

Prepare and view the ETH Office observations:

```powershell
uv run visual-memory-lab prepare-eth-office `
  --input data/eth-change-detection/office/office `
  --output outputs/phase6a/office-audit `
  --rgb-samples 24 `
  --vlm-samples 8
```

Then open `outputs/phase6a/office-audit/index.html` in a browser. The complete
geometry and VLM commands are documented in the Phase 6A guide.

To use the React showcase:

```powershell
cd web
npm run build
cd ..
uv run --extra cuda visual-memory-lab serve-ui
```

Open `http://127.0.0.1:8000/lab/objects` to browse model-generated object
evidence. The page filters 384 dense office keyframes by visit, object class,
detector score, and optional VLM audit status; it can show raw images, boxes,
masks, or both. These boxes are Grounding DINO predictions and the masks are
SAM 2.1 predictions. They are not hand-drawn annotations and do not establish
object identity across visits. The older `/lab/changes` route redirects here;
the Phase 6A research artifacts remain in `outputs/phase6a/`.

Generate the Phase 6B1 artifact on an NVIDIA GPU with:

```powershell
uv sync --extra cuda
uv run --extra cuda visual-memory-lab localize-eth-objects `
  --input data/eth-change-detection/office/office `
  --output outputs/phase6b1/object-localization `
  --keyframes-per-observation 96 `
  --device cuda
```

The optional 48-frame VLM pseudo-audit and full method are documented in the
Phase 6B1 guide.

## Dataset and model citations

This project reports results on the 7-Scenes dataset and uses the frozen CLIP
ViT-B/32 representation. Please use the following original sources when
referencing the project:

- Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio
  Criminisi, and Andrew Fitzgibbon. “Scene Coordinate Regression Forests for
  Camera Relocalization in RGB-D Images.” *CVPR*, 2013.
  [Microsoft Research publication](https://www.microsoft.com/en-us/research/publication/scene-coordinate-regression-forests-for-camera-relocalization-in-rgb-d-images-2/)
- Alec Radford et al. “Learning Transferable Visual Models From Natural
  Language Supervision.” *ICML*, 2021.
  [Paper](https://proceedings.mlr.press/v139/radford21a.html) ·
  [official CLIP repository](https://github.com/openai/CLIP)
- Marius Fehr et al. “TSDF-based Change Detection for Consistent Long-Term
  Dense Reconstruction and Dynamic Object Discovery.” *ICRA*, 2017.
  [Paper](https://cesarcadena.ethz.ch/files/ICRA2017_mfehr.pdf) ·
  [ETH dataset page](https://projects.asl.ethz.ch/datasets/change-detection/)
- Shilong Liu et al. “Grounding DINO: Marrying DINO with Grounded Pre-Training
  for Open-Set Object Detection.” *ECCV*, 2024.
  [Paper](https://arxiv.org/abs/2303.05499) ·
  [official repository](https://github.com/IDEA-Research/GroundingDINO)
- Nikhila Ravi et al. “SAM 2: Segment Anything in Images and Videos.” 2024.
  [Paper](https://arxiv.org/abs/2408.00714) ·
  [official repository](https://github.com/facebookresearch/sam2)

The [7-Scenes dataset page and license](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
restrict the dataset to non-commercial use. CLIP code is published under the
[MIT License](https://github.com/openai/CLIP/blob/main/LICENSE); its
[model card](https://github.com/openai/CLIP/blob/main/model-card.md) describes
the intended research use and deployment limitations.

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

The Phase 3 data, indexing, labeling, and evaluation commands are documented in
the Phase 3 document. `.env` is ignored; copy `.env.example` and set
`OPENAI_API_KEY` only when generating a new frozen zone artifact. Evaluation is
offline and never calls the OpenAI API.

Run the Phase 5 traversal-memory evaluation with:

```powershell
uv run visual-memory-lab evaluate-traversal-memory `
  --memory-index outputs/phase3/train-index `
  --query-index outputs/phase3/test-index `
  --output outputs/phase5/traversal-evaluation `
  --seed 42
```

This uses sequence IDs as traversal identifiers, not timestamps. It makes no
claim that the 7-Scenes sequences are chronologically ordered.

Build and run the Phase 4 interface with:

```powershell
Set-Location web
npm install
npm run build
Set-Location ..
uv run --extra cuda visual-memory-lab serve-ui
```

Then open `http://127.0.0.1:8000`. Search remains local. If `.env` contains an
OpenAI API key, the interface offers a separate confirmation step before it
sends a question and up to five selected public evidence frames for analysis.

The default research run contains 10 episodes and 380 observations. Generated
images, manifests, embeddings, and model weights remain local and are ignored
by Git. Choose a new output directory for each generated run or index; commands
will not replace an existing non-empty directory.

Run the test suite with:

```powershell
uv run python -m pytest -q
```
