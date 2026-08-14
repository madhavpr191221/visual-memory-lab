# Phase 9 — Charades video memory

## Purpose

Phase 9 extends Visual Memory Lab from single office images to short, searchable
video moments. The first user question is:

> When did this action happen?

Charades supplies 9,848 indoor videos with action intervals, object labels, and
descriptions. The local copy is not redistributed; consult the dataset's
README and license before sharing artifacts.

## What is implemented

The preparation commands create a deterministic 300-video training and
100-video test subset without copying the original videos:

```powershell
uv run visual-memory-lab prepare-charades `
  --input data/Charades_v1_480 `
  --output outputs/charades/subset

uv run visual-memory-lab build-charades-windows `
  --manifest outputs/charades/subset/manifest.jsonl `
  --output outputs/charades/windows
```

Each temporal window records its video, start/end time, overlapping action
intervals, objects, description, and source path. The first retrieval mode is
an explicit annotation-text baseline. It is useful for validating the product
flow before training a temporal model; it is not presented as learned vision.

## Connection to the office system

The office system stores image memories. Phase 9 stores time-window memories:

```text
image memory:  image + embedding + place metadata
video memory:  time window + frames + embedding + action metadata
```

Both use the same pattern: retrieve evidence first, then explain what the
evidence supports. ETH Office remains the spatial/place-memory dataset;
Charades is the temporal/action-memory dataset.

## Training path

The planned learned progression is:

1. frozen CLIP frame embeddings;
2. a trained temporal head over frame embeddings;
3. conventional gradient-based fine-tuning of the final CLIP vision blocks;
4. full fine-tuning only if validation supports it.

The trainable `TemporalWindowEncoder` and symmetric contrastive loss are now
available in `visual_memory_lab.temporal`. They consume precomputed frame
embeddings, so decoding and model training can be tested independently.

For frames `x_1, ..., x_T` in a window:

```math
z_t = f_{CLIP}(x_t),
\qquad
z_{window} = g(z_1, ..., z_T)
```

The temporal model `g` should learn that a sequence such as approaching a
chair, sitting, and standing again is different from a static chair image.

## Application UI

The Office assistant now has a Video memory page at `/app/video`. A user can:

- ask when an action happened;
- retrieve object-related windows;
- open the video at the returned timestamp;
- read the action and object evidence beside the player.

The first page intentionally labels the current method as an annotation
baseline. It does not claim that lexical matching is the final learned model.

## Evidence boundaries

- Charades provides RGB video, not metric depth or camera poses.
- A retrieved interval is evidence for a candidate moment, not proof of every
  event in that interval.
- Object labels do not establish persistent object identity.
- A no-result answer does not prove that an event never happened.
- Depth and 3D remain a later extension for physical location and change.
