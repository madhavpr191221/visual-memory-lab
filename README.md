# Visual Memory Lab

Visual Memory Lab is an evidence-first office inspection assistant. It helps a technician or facilities worker answer practical questions about a photographed office:

> Where was this workstation or object seen before, and what can I safely conclude from the available evidence?

The system retrieves real office images, compares a current photo with earlier views, and produces a cautious report that separates observations from conclusions. It is designed for inspection support, not autonomous maintenance decisions.

## The technician workflow

Imagine a technician checking a workstation after a service call. They upload a photo and may ask about a desk, chair, cables, papers, or visible damage. The assistant:

1. summarizes what is visible in the current photo;
2. retrieves visually relevant earlier office views;
3. lets the technician choose an earlier view to compare;
4. shows the two images side by side;
5. explains what is visible and what still needs a manual check;
6. saves the inspection and its evidence locally for later review.

The result is deliberately careful. A similar-looking chair is not automatically the same chair, and a missing detection is not proof that an object disappeared.

## What is implemented

- CLIP ViT-B/32 image and text retrieval over real office frames;
- pose-grounded retrieval evaluation and recurring office place zones;
- Grounding DINO candidate boxes and SAM 2.1 visible-pixel masks;
- recorded ETH RGB-D point-cloud evidence in a shared room frame;
- cautious cross-visit association candidates using appearance and approximate geometry;
- an Office assistant UI for asking questions, uploading photos, comparing views, and saving inspections;
- a Charades Video memory page for retrieving timestamped action windows;
- a Research workspace for reviewing retrieval, perception, geometry, association, and failure measurements.

## Two views of one system

The landing page (`/`) offers two entry points over the same prepared artifacts and API:

- **Office assistant** (`/app`): the user-facing workflow. Its primary pages are Ask memory, Inspect, and History.
- **Research workspace** (`/research`): a secondary validation view for engineers and reviewers. It exposes evaluation, failures, zones, object masks, RGB-D evidence, and association candidates.

The Research workspace is not a second product or separate pipeline. It makes the intermediate evidence and limitations visible so the application can be evaluated honestly.

## Hiring-manager demo path

For a quick walkthrough, open `/app/demo` or choose **Watch the guided case** on the landing page. The case takes about 90 seconds:

```text
office question → real retrieved views → side-by-side evidence → safe conclusion → manual check
```

It is a deterministic presentation of the existing office memory artifacts. The case is intentionally explicit about uncertainty: it demonstrates evidence selection and inspection support, not guaranteed object identity or autonomous maintenance.

The walkthrough and interview-facing technical notes are in [Guided Office Inspection Demo](docs/guided_demo.md).

## Current architecture

```text
Office RGB/RGB-D recordings + uploaded photo
                    ↓
Offline preparation and perception artifacts
                    ↓
Retrieval and evidence services
                    ↓
FastAPI domain API
                    ↓
Landing page (/)
          ↙                         ↘
Office assistant (/app)       Research workspace (/research)
          ↓
Inspection history and reports
```

Expensive model work runs offline. The browser reads prepared artifacts rather than rerunning indexing, detection, segmentation, or RGB-D processing on every page load. An uploaded photo can be summarized and compared through explicit API actions when `OPENAI_API_KEY` is configured.

The detailed design is in [System Design and Architecture](docs/system_design_and_architecture.md).

## Data and models

### 7-Scenes Office

Used for real-image place memory. It supplies RGB frames and camera poses. The project uses the official 6,000-memory / 4,000-query split for pose-grounded retrieval evaluation. Sequence order is not treated as calendar time.

### ETH Office

Used for object-aware evidence. The recordings contain RGB images, coloured point clouds, and recorded transforms for four logical office visits. The dataset is referenced locally and is not redistributed.

### Models

- CLIP ViT-B/32: frozen image/text representation for exact retrieval;
- Grounding DINO: text-guided candidate object boxes;
- SAM 2.1: pixel masks for candidate boxes;
- optional VLM analysis: bounded summaries and reports for selected evidence, never human ground truth.

## Public API

Core retrieval and evidence routes:

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
GET  /api/evidence
GET  /api/associations
```

Technician inspection routes:

```text
GET  /api/inspections
GET  /api/inspections/{inspection_id}
POST /api/inspections
POST /api/inspections/with-image
GET  /api/inspections/{inspection_id}/current-image
POST /api/inspections/{inspection_id}/compare
POST /api/inspection-summary/image
POST /api/inspections/{inspection_id}/summary
POST /api/inspections/{inspection_id}/report
```

Ordinary retrieval is local. Cloud/VLM analysis is a separate explicit action and requires `OPENAI_API_KEY`. Saved inspection records and reports use local SQLite; uploaded images are kept in the configured local output directory.

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
/                 Landing page
/app              Office assistant: ask memory
/app/inspect      Upload a current photo and compare it with earlier views
/app/inspections Saved inspection history
/app/video         Charades video memory: find an action or object moment
/research         Research overview
/research/evaluation
/research/failures
/research/zones
/research/objects
/research/evidence
/research/associations
```

The application navigation intentionally stays small. Detailed research routes remain available for focused review but are not primary technician tabs.

## Charades video memory

The downloaded Charades copy is kept locally under `data/Charades_v1_480`. It
contains the videos plus the official annotation and license files. Prepare a
small reproducible subset and timestamped windows with:

```powershell
uv run visual-memory-lab prepare-charades `
  --input data/Charades_v1_480 `
  --output outputs/charades/subset

uv run visual-memory-lab build-charades-windows `
  --manifest outputs/charades/subset/manifest.jsonl `
  --output outputs/charades/windows
```

Then start the UI as usual and open `/app/video`. The first retrieval method is
an explicit annotation-text baseline. The next model stage will replace it
with frozen CLIP frame embeddings, a trainable temporal head, and eventually
ordinary gradient-based CLIP fine-tuning. See [Phase 9 — Charades video memory](docs/phases/09_charades_video_memory.md).

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

The complete preparation and evaluation commands are documented in [Phase 3/4](docs/phases/03_04_real_office_visual_memory_system.md).

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

## Evidence boundaries

- a high CLIP score means visual similarity, not physical identity;
- a detection is a category prediction, not a persistent object ID;
- no detection does not prove absence;
- recorded point clouds describe visible geometry, not a complete object model;
- a cross-visit candidate is a possible match, not a verified move;
- sequence order is a logical visit order, not a calendar timestamp;
- an uploaded photo has no dataset zone or visit metadata unless supplied separately;
- VLM summaries and reports are supporting analysis, not human labels.

## Evaluation and tests

The research workspace reports pose coverage, hit@k, translation and rotation error, zone agreement, detector and mask evidence, RGB-D point coverage, association uncertainty, and technician-question evidence recall.

```powershell
uv run python -m pytest -q
```

If the local `uv` cache is unavailable:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q tests --basetemp .tmp-pytest\run
```

## Citations and licensing

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The 7-Scenes dataset is restricted to non-commercial use; the dataset, embeddings, and model weights are not redistributed.

Key sources:

- Shotton et al., “Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images,” [Microsoft Research](https://www.microsoft.com/en-us/research/publication/scene-coordinate-regression-forests-for-camera-relocalization-in-rgb-d-images-2/)
- Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” [ICML 2021](https://proceedings.mlr.press/v139/radford21a.html)
- Fehr et al., “TSDF-based Change Detection for Consistent Long-Term Dense Reconstruction and Dynamic Object Discovery,” [ICRA 2017](https://cesarcadena.ethz.ch/files/ICRA2017_mfehr.pdf)
- Liu et al., “Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection,” [ECCV 2024](https://arxiv.org/abs/2303.05499)
- Ravi et al., “SAM 2: Segment Anything in Images and Videos,” [2024](https://arxiv.org/abs/2408.00714)

## Current limitations and next steps

The current system is an offline, local prototype. It does not yet provide persistent object identity, reliable true change detection under viewpoint and lighting changes, live video ingestion, or authenticated multi-user deployment. Future work should be driven by measured failures and controlled repeated-visit data rather than adding complexity for its own sake.
