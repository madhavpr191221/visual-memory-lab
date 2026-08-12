# Phase 6.1.3: Cross-Visit Object Association

## Objective

Phase 6.1.2 showed independently selected detections from two visits. It did
not know whether they referred to the same physical object. This phase ranks
candidate matches without pretending that every pair is a confirmed identity.

For a technician, the question becomes:

> “Could these two detections be the same chair, bin, or box?”

## Association score

Only detections with the same predicted class are compared. A frozen CLIP
ViT-B/32 crop embedding supplies the appearance signal:

$$
s_{appearance}=\cos(e_i,e_j).
$$

The baseline combines appearance, mask shape, evidence quality, and approximate
position:

$$
S_{ij}=0.55s_{appearance}+0.15s_{shape}+0.15s_{evidence}+0.15s_{position}.
$$

The position term is intentionally weak. A real object may move, so a large
position difference should not automatically reject a visually compatible
candidate.

Operating labels are:

```text
score >= 0.70       likely same object
0.50 to 0.70        possible match
below 0.50          uncertain / weak candidate
```

These are baseline thresholds, not measured accuracy claims.

For two valid RGB-D centroids:

$$
d_{3D}=\lVert\bar p_j-\bar p_i\rVert_2.
$$

The UI may say **possible movement** when a likely match has adequate evidence
and (d_{3D}>0.5\) metres. It never calls that definitive movement.

## What changed after the screenshot failure

The old page selected the highest-point-count detection independently in each
visit. That could pair a false-positive bin with a real box. Phase 6.1.3
instead shows the candidate pair, detector boxes, masks, score breakdown, and
uncertainty. If no association has been established, the UI says so directly.

## VLM pseudo-audit

The optional audit reviews the 200 highest-ranked pairs using paired RGB crops.
It returns `same`, `different`, or `uncertain`. It does not see the geometry
score and does not decide movement. These judgments are a pseudo-reference for
failure analysis, not ground truth.

## Running it

```powershell
uv run visual-memory-lab associate-eth-objects `
  --localization outputs/phase6b1/object-localization `
  --rgbd-evidence outputs/phase612/rgbd-evidence `
  --output outputs/phase613/associations `
  --device cuda
```

Optional VLM review:

```powershell
uv run visual-memory-lab audit-eth-object-associations `
  --associations outputs/phase613/associations `
  --localization outputs/phase6b1/object-localization `
  --output outputs/phase613/vlm-audit `
  --cache-dir outputs/phase613/vlm-cache `
  --limit 200
```

Open the UI at:

```text
http://127.0.0.1:8000/lab/object-association
```

The acceptance run ranked 3,600 candidates from 1,417 detections. The local
run used the pinned CLIP checkpoint on CUDA; if model weights are unavailable,
the command records and uses a deterministic RGB-histogram fallback instead.

## Boundary

This phase does not assign permanent object IDs, prove that a person moved an
object, or classify additions and removals. Those require validated identity,
visibility, and temporal reasoning in a later phase.
