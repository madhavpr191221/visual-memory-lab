# Visual Memory Lab

Visual Memory Lab is a small research project about finding useful evidence in a history of real office images. It uses the publicly available 7-Scenes Office and ETH Office research datasets.

The current Phase 6.1 work adds automatic RGB object localization over 384 dense office
keyframes. Frozen Grounding DINO predictions and SAM 2.1 masks replace the
hand-drawn boxes used in the earlier change showcase. The dataset itself is not
redistributed.

> **Research-use notice:** 7-Scenes is provided by Microsoft Research for
> non-commercial use. This repository is a personal research and portfolio
> demonstration for hiring-manager review, not a commercial deployment. The
> original RGB-D dataset, embeddings, and model weights are not redistributed.
> See [Third-Party Notices](THIRD_PARTY_NOTICES.md).

The basic problem is simple: a camera may record thousands of observations over time, but storing those images is not enough. When someone asks about an earlier event, the system must retrieve the right image even if the viewpoint, lighting, surroundings, or appearance of an object has changed.

## A real-world example

Imagine a maintenance technician making weekly rounds through a factory. A body camera or phone records images during each inspection. Several weeks later, a leak is found near a blue valve, and the technician wants to know:

> When was rust first visible around the blue valve beside the pressure gauge?

A normal image search may return other blue valves because they look similar. A useful visual memory system should retrieve the correct valve, in the correct part of the factory, from the inspection when the rust first appeared. It should still work if the technician approached from another direction, the lighting was different, or equipment partly blocked the view.

The same idea could support building inspections, construction progress reviews, field service, environmental surveys, or accessibility tools that help people recall where an object was last seen.

## Research question

> Can a visual memory system retrieve the right past observation when viewpoint, time, occlusion, or scene appearance changes?

This is different from asking whether two images look alike. The most visually similar image may come from the wrong room, the wrong time, or the wrong object. The useful memory is the one that contains the evidence needed for the task.

## How the project will work

1. Prepare public office recordings and deterministic manifests.
2. Create CLIP embeddings for stored RGB images.
3. Store embeddings and metadata in a simple exact index.
4. Retrieve past observations using text or another image.
5. Add pose, zone, object, and RGB-D evidence where available.
6. Inspect the evidence and its limitations in the local UI.

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

The repository searches real office imagery with frozen CLIP ViT-B/32, evaluates
place memory on 10,000 7-Scenes Office frames, and adds ETH Office object and
RGB-D evidence:

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
The current implementation and target evolution are described in
[System Design and Architecture](docs/system_design_and_architecture.md).
The longer-term product and research thesis is documented in
[Visual Memory Lab: Research and Application Direction](docs/visual_memory_lab_research_and_application_direction.md).
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

The current object-localization baseline is documented in
[Phase 6.1.1: Automatic Object Localization](docs/phases/06_1_object_localization.md).
The RGB-D evidence and visit-comparison step is documented in
[Phase 6.1.2: RGB-D Object Evidence](docs/phases/06_1_2_rgbd_object_evidence.md).
The candidate identity-association step is documented in
[Phase 6.1.3: Cross-Visit Object Association](docs/phases/06_1_3_cross_visit_object_association.md).
The broader object-aware memory design is documented in
[Phase 6.1: Object-Aware Change Memory](docs/phases/06_1_object_aware_change_memory.md).
The high-level roadmap is in
[Phase 6.1 overview](docs/phases/06_1_overview.md).
Its CUDA acceptance run processed 384 keyframes and retained 1,417 predictions:
515 chairs, 477 waste bins, and 425 boxes. These are predictions, not correct
object counts. The completed VLM audit reviewed all 48 requested frames and
found substantial false positives: 79 supported, 15 uncertain, and 71
unsupported predictions. Frozen counts and the pseudo-audit boundary are in
[`artifacts/phase6b1/summary.json`](artifacts/phase6b1/summary.json).

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
object identity across visits. Cross-visit identity is intentionally not claimed.

Open `http://127.0.0.1:8000/lab/object-evidence` to compare visible RGB-D
evidence for an object class across two logical visits.
Open `http://127.0.0.1:8000/lab/object-association` to inspect ranked candidate
matches across visits.

Generate the Phase 6.1.1 artifact on an NVIDIA GPU with:

```powershell
uv sync --extra cuda
uv run --extra cuda visual-memory-lab localize-eth-objects `
  --input data/eth-change-detection/office/office `
  --output outputs/phase6b1/object-localization `
  --keyframes-per-observation 96 `
  --device cuda
```

The optional 48-frame VLM pseudo-audit and full method are documented in the
Phase 6.1.1 guide.

Build the Phase 6.1.2 RGB-D evidence artifact with:

```powershell
uv run visual-memory-lab build-eth-rgbd-evidence `
  --input data/eth-change-detection/office/office `
  --localization outputs/phase6b1/object-localization `
  --output outputs/phase612/rgbd-evidence
```

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
uv run visual-memory-lab prepare-7-scenes `
  --input data/7-scenes/office `
  --output outputs/phase3/office

uv run visual-memory-lab index `
  --input outputs/phase3/office/train `
  --output outputs/phase3/train-index

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

Generated images, manifests, embeddings, and model weights remain local and are
ignored by Git. Choose a new output directory for each generated run or index; commands
will not replace an existing non-empty directory.

Run the test suite with:

```powershell
uv run python -m pytest -q
```
