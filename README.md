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
