# Visual Memory Lab

Visual Memory Lab is a research prototype for remembering and inspecting real office spaces.

It is built around a practical technician question:

> “Where was this object or workstation seen during an earlier inspection, and what evidence supports the answer?”

The system retrieves real RGB observations first, then adds camera pose, semantic place zones, object masks, recorded RGB-D geometry, and cautious cross-visit comparisons. It does not pretend that a detector prediction is a verified identity or that a missing prediction proves an object disappeared.

## Technician example

Imagine a facilities technician who photographs an office during several visits. Later, they want to know where a chair was seen, which desk was beside a window, or whether two views may show the same physical object.

The system can:

- retrieve earlier views using text or an image;
- show the source frame, visit, and camera metadata;
- compare semantic office zones;
- locate candidate chairs, bins, and boxes with Grounding DINO and SAM 2.1;
- connect visible object masks to recorded ETH RGB-D point clouds;
- rank possible cross-visit matches using appearance and approximate geometry;
- show uncertainty and evidence boundaries next to every result.

The system cannot currently prove persistent object identity, prove that an object moved, or infer calendar time from public sequence IDs.

## Current architecture

```text
7-Scenes / ETH Office recordings
            ↓
validated manifests and deterministic keyframes
            ↓
offline pipelines
  CLIP retrieval | place zones | object detection | masks | RGB-D | association
            ↓
versioned local artifacts
            ↓
FastAPI read-only API
            ↓
React/TypeScript evidence explorer
```

Expensive model work runs offline. The browser reads prepared artifacts instead of rerunning perception on every page load. This makes results reproducible, inspectable, and fast to browse.

The detailed design is in [System Design and Architecture](docs/system_design_and_architecture.md).

## The landing page

The root page (`/`) is a deliberate starting point, not another research
dashboard. It asks what kind of work the visitor wants to do:

- **Use Visual Memory** opens `/app`, the technician-facing workflow for asking
  about an earlier office view, finding candidate objects, comparing visits,
  and opening the supporting evidence.
- **System Insights** opens `/research`, the engineering-facing workflow for
  checking retrieval quality, zones, detector and mask outputs, 3D evidence,
  associations, and known failure cases.

Both views use the same prepared office artifacts and API. The difference is
the question being answered: `/app` helps someone inspect an office, while
`/research` helps someone understand how reliable the system is.

## Data and models

### 7-Scenes Office

Used for real-image place memory. The dataset supplies RGB frames and camera poses. The project uses an official 6,000-memory / 4,000-query split, pose-grounded coverage, hit@k, and pose-error evaluation.

### ETH Office

Used for object-aware evidence. The recordings contain RGB images, coloured point clouds, and recorded transforms. The project uses four logical office visits and does not redistribute the dataset.

### Models

- CLIP ViT-B/32: frozen image/text representation for exact retrieval;
- Grounding DINO: text-guided candidate object boxes;
- SAM 2.1: pixel masks for candidate boxes;
- optional VLM review: bounded pseudo-audit for selected evidence, never human ground truth.

## Active research phases

1. **Office place memory:** retrieve relevant real office views and evaluate them against camera pose.
2. **Office explorer:** inspect images, zones, evaluations, and failure cases in a local UI.
3. **Cross-traversal memory:** measure retrieval across designated office traversals.
4. **Object localization:** produce inspectable boxes and masks for chairs, bins, and boxes.
5. **RGB-D object evidence:** summarize visible object geometry in the recorded room frame.
6. **Cross-visit association:** rank cautious candidate matches across visits.
7. **Technician task benchmark:** evaluate manually authored office questions with evidence and safe-abstention labels.

The detailed phase documents are listed in the [docs/phases](docs/phases) directory.

## Public API

The API uses domain-oriented routes rather than phase-numbered routes:

```text
GET  /api/health
GET  /api/capabilities
GET  /api/memory
GET  /api/memory/evaluation
POST /api/search/text
POST /api/search/image
GET  /api/zones
GET  /api/zones/{slug}
GET  /api/objects
GET  /api/objects/images/{image_id}
GET  /api/evidence
GET  /api/associations
```

The ordinary retrieval path is local. Cloud analysis is a separate, explicit action and is unavailable unless `OPENAI_API_KEY` is configured.

## Run the office UI

The project uses Python 3.13 and `uv`.

```powershell
uv sync
Set-Location web
npm install
npm run build
Set-Location ..
uv run --extra cuda visual-memory-lab serve-ui
```

Open `http://127.0.0.1:8000`.

Useful pages:

```text
/                 Landing page: choose a workflow
/app              Technician view: ask memory
/app/objects      Find candidate objects
/app/compare      Compare two visits
/app/evidence     Open supporting evidence
/app/tasks        Technician-style task benchmark
/research         Research overview
/research/evaluation       Retrieval evaluation
/research/failures         Failure browser
/research/zones             Office place zones
/research/objects           Detector and mask outputs
/research/evidence          RGB-D evidence
/research/associations      Cross-visit candidates
```

## Build the real-image artifacts

Prepare the 7-Scenes Office split and build a CLIP index:

```powershell
uv run visual-memory-lab prepare-7-scenes `
  --input data/7-scenes/office `
  --output outputs/phase3/office

uv run visual-memory-lab index `
  --input outputs/phase3/office/train `
  --output outputs/phase3/train-index
```

The complete place-memory, zone, and evaluation commands are documented in [Phase 3/4](docs/phases/03_04_real_office_visual_memory_system.md).

Run object localization on an NVIDIA GPU:

```powershell
uv sync --extra cuda
uv run --extra cuda visual-memory-lab localize-eth-objects `
  --input data/eth-change-detection/office/office `
  --output outputs/phase6b1/object-localization `
  --keyframes-per-observation 96 `
  --device cuda
```

Build RGB-D evidence:

```powershell
uv run visual-memory-lab build-eth-rgbd-evidence `
  --input data/eth-change-detection/office/office `
  --localization outputs/phase6b1/object-localization `
  --output outputs/phase612/rgbd-evidence
```

## Evaluation and limitations

The project reports:

- pose coverage and pose-grounded hit@1/5/10;
- translation and rotation error;
- zone retrieval agreement;
- cross-traversal coverage and retrieval quality;
- detector, mask, geometry, and association evidence.

Important boundaries:

- a high CLIP score means visual similarity, not physical identity;
- a detection is a category prediction, not a persistent object ID;
- no detection does not prove absence;
- recorded point clouds describe visible geometry, not a complete object model;
- a cross-visit candidate is a possible match, not a verified move;
- sequence order is a logical visit order, not a calendar timestamp;
- VLM judgments are cached pseudo-audits, not human labels.

## Tests

```powershell
uv run python -m pytest -q
```

If the local `uv` cache is unavailable, the repository’s Python 3.13 environment can be used directly:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q tests --basetemp .tmp-pytest\run
```

## Citations and licensing

The project uses public research datasets and published models. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the citations at the end of this README for dataset and model sources. The 7-Scenes dataset is restricted to non-commercial use; the dataset, embeddings, and model weights are not redistributed.

Key sources:

- Shotton et al., “Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images,” CVPR 2013. [Microsoft Research](https://www.microsoft.com/en-us/research/publication/scene-coordinate-regression-forests-for-camera-relocalization-in-rgb-d-images-2/)
- Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” ICML 2021. [Paper](https://proceedings.mlr.press/v139/radford21a.html)
- Fehr et al., “TSDF-based Change Detection for Consistent Long-Term Dense Reconstruction and Dynamic Object Discovery,” ICRA 2017. [Paper](https://cesarcadena.ethz.ch/files/ICRA2017_mfehr.pdf)
- Liu et al., “Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection,” ECCV 2024. [Paper](https://arxiv.org/abs/2303.05499)
- Ravi et al., “SAM 2: Segment Anything in Images and Videos,” 2024. [Paper](https://arxiv.org/abs/2408.00714)

## Project direction

The long-term question is:

> “What was this office area like during the previous visit, and what evidence supports any difference?”

The next research steps should improve coverage, persistent object identity, temporal metadata, calibrated abstention, and evaluation on controlled repeated real-world visits.
