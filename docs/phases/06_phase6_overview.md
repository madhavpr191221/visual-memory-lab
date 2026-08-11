# Phase 6: Object-Aware Physical Change Memory

## Purpose

Phase 6 is the project's move from **“which image looks relevant?”** to
**“what physical object was present, where was it, and what may have changed?”**

The motivating user is a technician, inspector, or facilities worker who visits
the same place repeatedly and later asks:

> “What changed since my previous inspection?”

The answer should be based on visible RGB evidence, geometry, camera coverage,
and explicit uncertainty.

Phase 6 is intentionally divided into subphases. Each subphase should produce a
working artifact and UI evidence before the next one is started.

## High-level pipeline

```text
real office visits
      ↓
RGB and 3D observations
      ↓
find candidate objects
      ↓
connect object evidence to physical space
      ↓
match observations across visits
      ↓
compare object state
      ↓
show an evidence-grounded technician answer
```

Example:

```text
Visit 1: chair beside workstation
Visit 2: chair near window
          ↓
possible chair relocation, with RGB and 3D evidence
```

This document is a high-level roadmap. Detailed mathematics, implementation
notes, commands, and measured results belong in the individual subphase
documents. Completed sections will be filled in as the project progresses.

## Phase 6A — Coarse 3D change baseline

### Question

Where do two real office reconstructions disagree geometrically?

### What it does

Phase 6A compares aligned ETH Office observations and identifies regions where
surfaces appear in one observation but not the other. It uses RGB-D-derived
reconstructions, aligned coordinates, distance thresholds, clustering, and a
VLM-assisted pseudo-review.

### Technician example

The system may show that a floor-level region near a desk changed between two
visits. It can show the RGB views and a 3D difference region, but it cannot yet
reliably say “the chair moved.”

### Status

Implemented. The detailed record is
[Phase 6A: Controlled 3D State-Change Baseline](06a_controlled_3d_change_baseline.md).

### Results to record

```text
Raw geometric candidates:      [fill after future reruns]
Reviewed candidates:            [fill after future reruns]
Main failure modes:             reconstruction fragments, viewpoint coverage,
                                and unnamed changed objects
```

## Phase 6B1 — RGB object localization

### Question

Where are likely chairs, waste bins, and boxes in the office images?

### What it does

Phase 6B1 processes dense RGB keyframes with frozen Grounding DINO detection
and SAM 2.1 segmentation. The Objects UI shows raw images, predicted boxes,
masks, confidence scores, and optional VLM pseudo-audit judgments.

### Technician example

The system can answer:

> “Show me the frames in which a chair was detected near the workstation.”

It cannot yet answer whether two chair detections are the same physical chair.

### Status

Implemented. See
[Phase 6B1: Automatic Object Localization](06b1_object_localization.md).

### Results

The current acceptance artifact contains 384 keyframes and 1,417 retained
predictions. The complete measured counts are kept in
[`artifacts/phase6b1/summary.json`](../../artifacts/phase6b1/summary.json).

## Phase 6B2 — RGB-D object evidence

### Question

Where is the visible part of each detected object in the shared room frame?

### Planned capability

Use each RGB mask together with valid depth and the recorded camera pose. The
system will turn masked pixels into partial 3D object evidence and expose the
result beside the RGB image.

### Technician example

Instead of only saying “chair pixels are here,” the system can say:

> “The visible chair surface was approximately 2.1 metres from the desk in the
> room coordinate frame.”

### Important boundary

This will be a partial visible-surface representation, not a complete object
model. It will not automatically establish object identity or movement.

### Status

Planned. The detailed design will be written when implementation begins.

### Results to record

```text
Depth alignment checks:          [fill]
Valid masked-depth rate:         [fill]
3D object evidence examples:    [fill]
Failure cases:                   [fill]
```

## Phase 6B3 — Cross-visit object association

### Question

Which observations from different visits may refer to the same physical object?

### Planned capability

Compare multiple signals together:

- RGB appearance;
- object class;
- visible shape and mask;
- approximate 3D position;
- office zone;
- camera coverage and occlusion.

The output should be a cautious match such as `likely same object`, `possible
match`, or `uncertain`, rather than an unqualified permanent identity.

### Technician example

Two visits both contain a black chair. If their appearance and shape are
compatible but their room positions differ by 1.2 metres, the system may propose
a relocation. If two identical chairs are present, the match should remain
uncertain without stronger evidence.

### Status

Planned.

### Results to record

```text
Labelled object-pair protocol:  [fill]
Same-object accuracy:           [fill]
Identity-confusion cases:       [fill]
Coverage-related failures:      [fill]
```

## Phase 6B4 — Object state and change reasoning

### Question

Was an object moved, added, removed, or simply not visible?

### Planned capability

Compare earlier and later object records while checking whether the relevant
area was actually observed. The system should report one interpretable outcome:

```text
possible move
possible addition
possible removal
uncertain because of coverage, occlusion, or weak evidence
```

### Technician example

If a chair is visible beside a desk in Visit 1, the old area is clearly visible
in Visit 2, and a compatible chair appears near the window, the system may show
“possible move.” If the old area is hidden behind a cabinet, it must show
“uncertain,” not “removed.”

### Status

Planned.

### Results to record

```text
Move precision/recall:          [fill]
Addition/removal performance:   [fill]
Uncertainty calibration:        [fill]
Main false-change categories:   [fill]
```

## Phase 6B5 — Learned improvement

### Question

Which measured failure is important enough to justify training a model?

### Planned capability

Only after the earlier baselines produce a labelled failure set should we train
one component. Possible targets include:

- object-pair association;
- RGB-D change classification;
- mask-quality correction;
- object-level correspondence;
- visibility-aware change classification.

The training dataset may need to be separately labelled, synthetic, or
controlled. The ETH Office dataset does not automatically provide object-level
ground truth for every desired claim.

### Technician example

If the baseline often confuses two identical office chairs, a learned
association model could be trained specifically for that failure. Training a
large model without measuring this failure first would make the experiment much
harder to interpret.

### Status

Planned.

### Results to record

```text
Training data:                   [fill]
Baseline:                        [fill]
Learned model:                   [fill]
Held-out improvement:            [fill]
Remaining failures:              [fill]
```

## Phase 6 UI progression

Each subphase should have a corresponding inspection view:

| Subphase | UI evidence |
| --- | --- |
| 6A | RGB visit comparison and coarse 3D difference regions |
| 6B1 | RGB images, detector boxes, segmentation masks, confidence and audit status |
| 6B2 | RGB masks beside partial 3D object evidence |
| 6B3 | Earlier/later object pair with match explanation |
| 6B4 | One change outcome with before/after RGB and 3D evidence |
| 6B5 | Baseline-versus-trained-model failure comparison |

The UI is part of the research workflow, not only a final presentation layer.
It lets us notice false positives, bad masks, missing coverage, and misleading
claims before building another stage on top of them.

## Overall success criteria

Phase 6 will be successful when a technician can select two office visits and
receive an answer that:

1. identifies the relevant object evidence;
2. places comparable observations in physical space;
3. distinguishes likely change from missing coverage;
4. shows the earlier and later RGB evidence;
5. shows the supporting 3D evidence when available;
6. states uncertainty and limitations plainly;
7. has a measured evaluation protocol behind its claims.

The project should not claim a persistent world-state memory until these pieces
have been implemented and evaluated separately.
