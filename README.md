# Visual Memory Lab

Visual Memory Lab is an evidence-first video memory assistant. It helps a
technician or reviewer search a recording for a practical event:

> When did the person open the cabinet, pick up the drill, or complete the inspection step?

The system retrieves timestamped video evidence, groups overlapping windows into
distinct events, and optionally asks a VLM to explain selected RGB evidence. It
is designed for inspection support, not autonomous maintenance decisions. The
older office-image workflows remain available as an archive and research
surface.

## The technician workflow

Imagine a technician reviewing a camera recording after a service call. They
choose the recording and ask a question. The assistant:

1. converts the question into a CLIP text representation;
2. searches the learned temporal index;
3. groups overlapping windows into distinct candidate events;
4. shows playable evidence with event time and surrounding context;
5. optionally samples six RGB frames for a VLM-grounded explanation;
6. lets the technician confirm, reject, or save the finding.

The result is deliberately careful. A similar-looking chair is not automatically the same chair, and a missing detection is not proof that an object disappeared.

## What is implemented

- CLIP ViT-B/32 image and text retrieval over video windows;
- application-facing video memory with “Find a moment” and timestamped video summaries;
- pose-grounded retrieval evaluation and recurring office place zones;
- three-head temporal video retrieval: retrieval, action, and boundary heads;
- event grouping, timestamped playback, and evidence-scoped follow-ups;
- optional VLM-grounded answer synthesis from selected RGB frames;
- train/test evaluation with retrieval, temporal overlap, boundary, duplicate,
  and miss metrics;
- Grounding DINO candidate boxes and SAM 2.1 visible-pixel masks;
- recorded ETH RGB-D point-cloud evidence in a shared room frame;
- cautious cross-visit association candidates using appearance and approximate geometry;
- an archived office/image UI for the earlier inspection experiments;
- a video-first UI for recording-first event search, playable evidence, follow-ups, and saved findings;
- a Research workspace for reviewing retrieval, perception, geometry, association, and failure measurements.

## Two views of one system

The landing page (`/`) puts video memory first and offers two entry points over the same prepared artifacts and API:

- **Video memory** (`/app`): choose a recording, ask a question, review a playable event, ask a follow-up, and save a finding.
- **Research workspace** (`/research`): a secondary validation view for engineers and reviewers. It exposes video metrics plus the archived office/image evaluation pages.
- **Office archive** (`/archive/office`): the earlier image-based memory and inspection workflows, kept available but removed from the primary user path.

The Research workspace is not a second product or separate pipeline. It makes the intermediate evidence and limitations visible so the application can be evaluated honestly.

## Hiring-manager demo path

For a quick walkthrough of the primary product, open `/app/video` (or choose
Video memory from the landing page). The older office walkthrough remains
available in the archive:

```text
office question → real retrieved views → side-by-side evidence → safe conclusion → manual check
```

The video workflow is intentionally explicit about uncertainty: it demonstrates
evidence selection and review support, not guaranteed event identity or
autonomous maintenance.

The walkthrough and interview-facing technical notes are in [Guided Office Inspection Demo](docs/guided_demo.md).

## Current architecture

```text
Charades RGB videos + official action annotations
                    ↓
Offline preparation, CLIP features, and three-head temporal training
                    ↓
Retrieval, boundary refinement, RGB evidence sampling, and VLM synthesis
                    ↓
FastAPI domain API
                    ↓
Landing page (/)
          ↙                         ↘
Video memory (/app/video)     Research workspace (/research)
          ↓
Saved video findings
          ↑
Office archive (/archive/office)
```

Expensive model work runs offline. The browser reads prepared video artifacts
rather than rerunning indexing or temporal training on every page load. VLM
analysis is an explicit post-retrieval action and requires `OPENAI_API_KEY`.
Office photo and RGB-D processing is available only through the archive/research
surface.

The detailed design is in [System Design and Architecture](docs/system_design_and_architecture.md).

## Data and models

### 7-Scenes Office

Used for real-image place memory. It supplies RGB frames and camera poses. The project uses the official 6,000-memory / 4,000-query split for pose-grounded retrieval evaluation. Sequence order is not treated as calendar time.

### ETH Office

Used for object-aware evidence. The recordings contain RGB images, coloured point clouds, and recorded transforms for four logical office visits. The dataset is referenced locally and is not redistributed.

### Models

- CLIP ViT-B/32: frozen image/text representation for exact retrieval;
- PyTorch temporal encoder with retrieval, action, and boundary heads;
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

Video-memory routes:

```text
GET  /api/video-memory/catalog
GET  /api/video-memory?q=when did the person open the cabinet&video_id=...
GET  /api/video-memory/videos/{video_id}
POST /api/video-memory/follow-up
POST /api/video-memory/synthesize
POST /api/video-memory/findings
GET  /api/video-memory/findings
```

`/api/video-memory/synthesize` is deliberately bounded: it receives the
selected event and evidence IDs, samples RGB frames from that interval, and
returns a cited VLM answer or an annotation-grounded fallback.

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
/app              Video-memory landing/application
/app/video        Find an event in a recording or review its timeline
/app/inspect      Upload a current photo and compare it with earlier views
/app/inspections Saved inspection history
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
an explicit annotation-text baseline. The learned pipeline below adds frozen
CLIP frame/text embeddings and a trainable temporal head. See [Phase 9 —
Charades video memory](docs/phases/09_charades_video_memory.md) and the living
[video memory data and retrieval notes](docs/charades_video_memory.md).

### Learned video artifacts (Phase 10)

Phase 10 adds deterministic sixteen-frame records, cached CLIP frame/text
features, retrieval/action/boundary heads, an exact learned window index, event
grouping, and API
fallback behaviour. Prepare the first learned artifacts with:

```powershell
uv run visual-memory-lab prepare-charades --input data/Charades_v1_480 --output outputs/charades/learned --train-limit 1000 --test-limit 300
uv run visual-memory-lab build-charades-windows --manifest outputs/charades/learned/manifest.jsonl --output outputs/charades/learned/windows
uv run visual-memory-lab build-charades-frames --manifest outputs/charades/learned/windows/windows.jsonl --output outputs/charades/learned/frames
# First validate the pipeline on 100 videos. PyAV decodes on the CPU; CLIP uses CUDA when available.
uv run --extra cuda visual-memory-lab build-charades-video-cache --manifest outputs/charades/learned/frames/frames.jsonl --output outputs/charades/learned/pilot/cache --device auto --max-videos 100 --workers 4 --batch-size 16
uv run --extra cuda visual-memory-lab build-charades-video-cache --manifest outputs/charades/learned/frames/frames.jsonl --output outputs/charades/learned/pilot/cache --device auto --max-videos 100 --workers 4 --batch-size 16 --resume
uv run --extra cuda visual-memory-lab train-charades-video --cache outputs/charades/learned/pilot/cache --output outputs/charades/learned/pilot/training --device auto
uv run --extra cuda visual-memory-lab index-charades-video --cache outputs/charades/learned/pilot/cache --checkpoint outputs/charades/learned/pilot/training/temporal_head.pt --output outputs/charades/learned/pilot/index --device auto
uv run --extra cuda visual-memory-lab evaluate-charades-video --index outputs/charades/learned/pilot/index --test-manifest outputs/charades/learned/frames/frames.jsonl --output outputs/charades/learned/pilot/evaluation --device auto

# The full learned run contains 18,994 windows from 1,300 videos. Keep it
# separate from the pilot so the two experiments remain comparable.
uv run --extra cuda visual-memory-lab build-charades-video-cache --manifest outputs/charades/learned/frames/frames.jsonl --output outputs/charades/learned/full/cache --device auto --workers 4 --batch-size 16
uv run --extra cuda visual-memory-lab train-charades-video --cache outputs/charades/learned/full/cache --output outputs/charades/learned/full/training --device auto --split train
uv run --extra cuda visual-memory-lab index-charades-video --cache outputs/charades/learned/full/cache --checkpoint outputs/charades/learned/full/training/temporal_multitask.pt --output outputs/charades/learned/full/index --device auto --split train
uv run --extra cuda visual-memory-lab evaluate-charades-video --index outputs/charades/learned/full/index --test-manifest outputs/charades/learned/frames/frames.jsonl --output outputs/charades/learned/full/evaluation --device auto
```

The earlier annotation baseline contains 5,883 windows from the smaller
300-train/100-test preparation. The learned manifest contains 18,994 windows
from 1,000 train and 300 test videos. The Video memory page uses the learned
index when it exists and labels the annotation search as the fallback. This phase retrieves temporal windows; it
does not yet claim precise frame boundaries, audio understanding, depth, or 3D
reasoning. See [Phase 10 — learned video memory architecture](docs/phases/10_learned_video_memory_architecture.md).

The current three-head training-only index contains 14,824 windows. On 4,170
held-out test queries, it reached:

| Metric | Result |
| --- | ---: |
| Recall@1 | 0.6763 |
| Recall@5 | 0.9113 |
| Recall@10 | 0.9321 |
| Mean temporal IoU | 0.2598 |
| Median temporal IoU | 0.1979 |
| Mean boundary error | 7.23 s |
| Median boundary error | 6.50 s |
| Mean duplicate rate | 0.1650 |
| Misses | 283 / 4,170 |

The earlier one-head checkpoint reached Recall@1 0.6360, Recall@5 0.8746,
Recall@10 0.9173, and 345 misses. Retrieval improved, while temporal IoU and
boundary error changed only slightly. The boundary head is therefore a useful
localization experiment, not yet a production-grade timestamp guarantee.

### What the user sees

The video UI is intentionally simple:

1. choose a recording;
2. ask “When did the person open the cabinet?”;
3. review a small set of distinct candidate events;
4. play the event with nearby context;
5. optionally request a VLM explanation grounded in six sampled RGB frames;
6. save the finding with its timestamp, evidence IDs, status, and note.

Before displaying a candidate, the selected recording's action vocabulary is
checked. CLIP creates a semantic shortlist, then a text-only LLM maps the
question to exact supplied Charades action names. Only matching windows are
retrieved. If the recording has no supported action, the UI shows a safe
no-result and skips VLM synthesis; it will not display an unrelated event.

The VLM does not search the archive or create the labels. Official Charades
annotations supervise training and evaluation. If VLM analysis is unavailable,
the UI returns an annotation-grounded fallback and labels its provenance.

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

The learned video path now groups overlapping retrieval windows into distinct
events and can optionally synthesize a cited answer from six sampled RGB frames.
Official Charades annotations remain the training and evaluation reference; VLM
prose is an interpretation layer, not temporal ground truth.

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
