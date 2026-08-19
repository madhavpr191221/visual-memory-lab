# Phase 11 — Multimodal temporal-memory transfer

> **Current reference.** The detailed data contracts and implementation flow
> are maintained in [Charades video memory](../../charades_video_memory.md). The
> completed Charades reference uses 16 RGB samples per four-second window and
> is stored under `outputs/phase11/frames16/`. The multimodal record schema in
> this document remains a transfer design; audio, depth, and pose are not yet
> part of the Charades model input.

Phase 10 established the Charades video-memory baseline: CLIP frame features,
a temporal encoder, retrieval/action/boundary heads, overlap grouping, and
playable evidence. Its next limitation is temporal precision: retrieving the
right kind of event is easier than locating its exact start and end.

Phase 11 prepares the system for transfer to richer real-world recordings with
RGB, audio, depth, and pose. The project does not assume that a dataset has all
four modalities. Each recording declares which sensors are present, and a
missing modality is represented by an explicit availability mask rather than a
fake zero-valued sensor.

Before adding those modalities, the first implementation milestone is temporal
precision on Charades. The boundary head now has an explicit training weight,
defaulting to 2.0, so timestamp quality is not treated as an incidental side
effect of retrieval training:

```powershell
uv run visual-memory-lab train-charades-video `
  --cache outputs/phase11/frames16/cache-v2 `
  --output outputs/phase11/frames16/training `
  --boundary-weight 2.0 `
  --action-weight 1.0
```

Evaluation reports both seconds and a duration-normalized boundary error. A
0.5-second error is more serious in a 2-second clip than in a 30-second clip,
so the normalized value makes comparisons across recordings fairer.

### Current reference training run

The first CUDA run used three epochs, `action_weight=1.0`, and
`boundary_weight=2.0`:

| Epoch | Total loss | Retrieval loss | Action loss | Boundary loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.2478 | 2.4475 | 0.6209 | 0.0897 |
| 2 | 2.3197 | 1.7432 | 0.4637 | 0.0564 |
| 3 | 1.7963 | 1.3761 | 0.3477 | 0.0363 |

The total is computed as:

$$
\mathcal{L} =
\mathcal{L}_{retrieval}
+ 1.0\,\mathcal{L}_{action}
+ 2.0\,\mathcal{L}_{boundary}.
$$

For example, the first epoch is approximately
`2.5152 + 0.6155 + 2(0.0885) = 3.3077`.

The downward trend means the model is fitting the training examples. It is not
evidence by itself that held-out temporal IoU improved. The completed held-out
evaluation is in `outputs/phase11/frames16/evaluation/metrics.json`. The
nested-tensor warning is a PyTorch performance
warning caused by the transformer's pre-normalization configuration; training
still completed successfully on CUDA.

### Current held-out result

The 16-frame index was evaluated on 4,170 held-out queries:

| Metric | Result |
| --- | ---: |
| Recall@1 | 0.6604 |
| Recall@5 | 0.9070 |
| Recall@10 | 0.9326 |
| Mean temporal IoU | 0.2606 |
| Mean boundary error | 7.305 s |
| Mean normalized boundary error | 0.5710 |
| Duplicate rate | 0.1753 |
| Misses | 281 / 4,170 |

The event family is often retrieved in the top few candidates, but the exact
start and end are still coarse. Phase 11 improves the training contract and
measurement; it does not claim production-grade temporal localization.

## The record used by the pipeline

Each recording is represented as a JSONL record:

```json
{
  "video_id": "inspection-01",
  "duration_s": 12.0,
  "paths": {
    "rgb": "rgb.mp4",
    "audio": "audio.wav",
    "depth": "depth/",
    "pose": "pose.json"
  },
  "split": "test",
  "summary": "A technician opens a cabinet and removes a tool.",
  "annotations": [
    {
      "start_s": 1.0,
      "end_s": 4.0,
      "label": "open cabinet",
      "objects": ["cabinet"],
      "source": "dataset"
    }
  ]
}
```

The annotation interval is the target for temporal evaluation. A wider player
context may be shown in the UI, but it is not confused with the event itself.

## Multimodal representation

For a temporal window, the RGB encoder produces a sequence of frame vectors
`**x**_rgb,1, ..., **x**_rgb,N`. Audio and depth adapters produce aligned
sequences when those sensors exist. The fusion layer computes:

$$
\mathbf{h}_t = \mathrm{LayerNorm}\left(
\sum_m a_{t,m} P_m(\mathbf{x}_{t,m}) + Q(\mathbf{a}_t)
\right),
$$

where:

- `**x**_{t,m}` is the feature for modality `m` at time `t`;
- `P_m` projects that modality into the shared hidden size;
- `a_{t,m}` is 1 when the modality is available and 0 when it is missing;
- `**a**_t` is the complete availability vector;
- `Q` tells the model which sensors were actually present.

This prevents the model from treating “no depth was recorded” as “the measured
depth was zero.”

The temporal encoder then maps the fused sequence to one window vector:

$$
\mathbf{v}_i = g(\mathbf{h}_{i,1}, \ldots, \mathbf{h}_{i,N}).
$$

The current three heads remain:

1. retrieval representation;
2. multi-label action prediction;
3. start/end boundary prediction.

Joint training uses:

$$
\mathcal{L} =
\lambda_r \mathcal{L}_{retrieval} +
\lambda_a \mathcal{L}_{action} +
\lambda_b \mathcal{L}_{boundary}.
$$

## Technician example

Suppose a camera records a technician opening a cabinet, removing a drill, and
closing the door.

The user asks:

> When did the technician remove the drill?

The system searches the learned temporal index, predicts an event interval,
and returns:

```text
Matched event: removing a drill
Event interval: 14.2–17.6 s
Context shown: 12.2–19.6 s
Evidence: RGB frames, audio if available, depth/pose coverage if available
```

Depth is not needed to understand the words “remove the drill.” It becomes
useful when the system must check whether the drill was visible, whether the
camera had coverage of the cabinet, or whether a change occurred in a shared
physical coordinate frame.

## Dataset feasibility gate

Before an expensive training run, use the manifest audit command:

```powershell
uv run visual-memory-lab audit-multimodal-manifest `
  --input data/multimodal/records.jsonl `
  --output outputs/phase11/modality-audit.json
```

The audit reports recording count, split counts, duration, annotation count,
label count, and how many recordings contain RGB, audio, depth, and pose. The
pipeline must not claim an all-modality experiment unless the audit confirms
those files and their timestamp alignment.

## Evaluation

Charades remains the controlled source benchmark. A transfer dataset is kept
separate and evaluated by recording, participant, or scene. We report:

- Recall@1, Recall@5, and Recall@10;
- temporal IoU;
- start and end boundary error;
- unsupported-query rejection;
- duplicate-window rate;
- action-mapping accuracy;
- visual-support and coverage status;
- latency and GPU/CPU cost.

The primary research comparison is:

```text
frozen RGB CLIP
→ fine-tuned RGB + temporal heads
→ RGB + audio
→ RGB + depth/pose
→ all available modalities with explicit missing-modality masks
```

No transfer result is allowed to claim object identity or physical movement
unless the evidence and dataset annotations support that claim.
