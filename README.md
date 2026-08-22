# Visual Memory Lab

Visual Memory Lab is an evidence-first video memory assistant. It helps a
technician or reviewer answer questions such as:

> When did the person open the cabinet, pick up the drill, or complete the inspection step?

The system retrieves timestamped video evidence, groups overlapping windows
into distinct events, and can optionally ask a VLM to explain selected RGB
frames. It supports inspection review; it does not make autonomous maintenance
decisions.

## What the user does

1. Choose a recording.
2. Ask a natural-language question.
3. Review distinct candidate events with playable timestamps.
4. Inspect surrounding context and sampled frames.
5. Optionally request a VLM explanation.
6. Confirm, reject, or save the finding.

The detailed inference and UI contract is documented in [Charades Video
Memory](docs/charades_video_memory.md). In brief, the current application uses
prepared recordings: it loads the video catalog and learned temporal index,
maps the question to compatible recorded actions, searches CLIP-based temporal
vectors, refines the timestamp, groups overlapping windows, and shows playable
RGB evidence. An optional VLM explains only the selected evidence.

The action list shown by **Review the timeline** comes from official Charades
annotations. **Find an event** hides those labels before retrieval so the user
can test whether the system can find the right moment from the summary and
question alone. Arbitrary uploaded-video inference is available locally
through **Try your own video locally**. A local MP4 is copied into an
ephemeral session, split into overlapping RGB windows, embedded with CLIP, and
searched without requiring Charades annotations. It is not uploaded to a
hosted service. Because a private recording has no ground truth, the UI labels
its timestamps as visual candidates rather than verified action intervals.

The application is deliberately evidence-first: a similarity score is not
proof, and an unsupported question returns a safe no-result instead of an
unrelated event.

## Current system

```text
Charades RGB video + official annotations
              ↓
Offline preparation and 16-frame window sampling
              ↓
Frozen CLIP frame/text embeddings
              ↓
Temporal encoder: retrieval + action + boundary heads
              ↓
Temporal index and overlapping-event grouping
              ↓
Timestamped playback, frame evidence, and optional VLM explanation
              ↓
FastAPI API → video-memory UI → saved findings
```

Expensive model work runs offline. The browser reads prepared artifacts; it
does not retrain or rebuild an index during normal use. The current detailed
architecture is in [System Design and Architecture](docs/system_design_and_architecture.md).

After an event is selected, the UI can request an object-evidence pass over
that event's RGB frames. Grounding DINO predicts object boxes and the SAM
adapter attempts masks when available. The result reports frame coverage,
confidence, and limitations beside the evidence. This expensive pass is on
demand; it does not run over every recording during page load.

## Public documentation

The project has a deliberately small public reading path:

- [System Design and Architecture](docs/system_design_and_architecture.md)
- [Charades Video Memory](docs/charades_video_memory.md)
- [Guided Demo](docs/guided_demo.md)

Earlier image-based office experiments and phase notes are preserved in the
[documentation archive](docs/archive/README.md). They are research history,
not competing current products. The interview preparation handbook is kept
outside the repository.

## Current reference result

The authoritative reference uses 16 RGB samples per four-second window, a
three-head temporal model, and a held-out test set:

| Metric | Result |
| --- | ---: |
| Held-out queries | 4,170 |
| Recall@1 | 0.6604 |
| Recall@5 | 0.9070 |
| Recall@10 | 0.9326 |
| Mean temporal IoU | 0.2606 |
| Median temporal IoU | 0.2010 |
| Mean boundary error | 7.305 s |
| Mean duplicate rate | 0.1753 |
| Misses | 281 / 4,170 |

These results show strong top-k retrieval but coarse temporal boundaries. They
are an honest research baseline, not a production timestamp guarantee.

The Phase 12 frame-refinement implementation is currently on the
`phase/12-trustworthy-temporal-evidence` branch. Its code and methodology are
documented, and its first training, refined-index, and held-out evaluation
artifacts are now available under `outputs/phase12/frames16/`. The metrics above
remain the Phase 11 baseline; the first Phase 12 comparison is documented in
[Charades Video Memory](docs/charades_video_memory.md). It slightly improves
mean boundary error but does not yet improve retrieval recall or temporal IoU.

## Run the application

The project uses Python 3.13 and `uv`.

```powershell
uv sync
Set-Location web
npm install
npm run build
Set-Location ..
uv run visual-memory-lab serve-ui --device auto
```

Open <http://127.0.0.1:8000> and choose **Video memory**. The research
workspace remains available at `/research`; the earlier office UI is available
under `/archive/office` for historical comparison.

## Prepare Charades artifacts

The local dataset is expected at `data/Charades_v1_480` and is not redistributed.
The full preparation, caching, training, indexing, and evaluation commands are
documented in [Charades Video Memory](docs/charades_video_memory.md).

The current reference artifact family is under `outputs/phase11/frames16/`:

- `frames.jsonl` — sampled frame windows;
- `cache-v2` — frozen CLIP features;
- `training` — temporal checkpoint and epoch diagnostics;
- `index` — searchable learned windows;
- `evaluation` — held-out metrics.

## Models and boundaries

## Let others try it safely

The safest public workflow is local: each person clones the repository, runs
the UI, and imports their own MP4. Private videos remain on that machine. A
small, separately licensed demo clip can be supplied separately from the
repository; private footage should never be committed to GitHub.

In the UI, choose **Try your own video locally**, select an MP4 (up to 500 MB),
wait for RGB-window preparation, and then ask a question. The result is a
CLIP-ranked visual candidate with playable evidence. The upload progress shows
the current stage (uploading, checking, building visual memory, or finishing),
the number of four-second windows processed, and whether the GPU or CPU is in
use. Overlapping windows are grouped into candidate moments so the UI does not
pretend that several adjacent retrieval windows are separate events. Object
boxes are run only for the selected moment. Cloud/VLM analysis is a separate
explicit action.

For example, raw windows at 0--4 s, 2--6 s, and 4--8 s are displayed as one
0--8 s candidate moment. The grouping improves review ergonomics; it does not
create an action label or prove the exact event boundaries. Object inspection
uses at most 32 uniformly spaced timestamps from the selected moment, so a
long grouped interval remains within the evidence API limit while retaining
coverage from its beginning to its end.

- CLIP ViT-B/32 supplies frozen image and text representations.
- A PyTorch temporal encoder learns window-level retrieval, action, and
  boundary signals.
- An optional VLM explains selected RGB evidence after retrieval.
- Official Charades annotations supervise training and evaluation; VLM prose is
  not ground truth.

The system does not currently claim audio understanding, depth reasoning, 3D
reconstruction, persistent object identity, or autonomous action selection.

## Evaluation and tests

```powershell
uv run python -m pytest -q
```

The research workspace exposes retrieval, temporal, evidence, and historical
office diagnostics without adding those surfaces to the primary technician
workflow.

## Citations and licensing

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Public datasets remain
subject to their own licenses; datasets, embeddings, and model weights are not
redistributed by this repository.

Key sources include CLIP (Radford et al., ICML 2021), Grounding DINO (Liu et
al., ECCV 2024), and SAM 2 (Ravi et al., 2024). Links and complete historical
references are retained in the architecture document and archive.
