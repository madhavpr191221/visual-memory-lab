# Phase 6A: Controlled 3D State-Change Baseline

## What this phase asks

Phase 6A asks a deliberately narrow question:

> When the same real office is reconstructed more than once, which 3D surfaces
> appear only in one observation, and which of those differences look like
> plausible physical changes rather than reconstruction noise?

This is the first phase that compares different states of a real scene. It is
not yet a trained change detector and it does not claim human-labelled ground
truth. Its purpose is to establish an interpretable geometric baseline and
identify the failures a learned method must improve.

The experiment uses the controlled `office` scene from ETH Zurich's
[Change Detection Datasets](https://projects.asl.ethz.ch/datasets/change-detection/),
released with the paper [*TSDF-based Change Detection for Consistent Long-Term
Dense Reconstruction and Dynamic Object Discovery*](https://cesarcadena.ethz.ch/files/ICRA2017_mfehr.pdf).
The repository does not redistribute the dataset.

## Real application

Imagine a facilities technician scanning an equipment room during repeated
inspection rounds. The technician needs evidence for questions such as:

- Is there an object here that was not present during the earlier inspection?
- Is equipment missing from its previous position?
- Is this a physical change, or did the reconstruction simply miss a surface?

The ETH Office data is easier than that deployment: its four observations were
recorded from a tripod near the centre of one controlled office, with close to
complete overlap. That controlled setup is useful because it lets the project
study scene differencing before adding handheld-camera motion and incomplete
coverage.

## Inputs

The local dataset contains four logical observations. Each observation has:

- one ROS1 bag containing 1,208 colour images, camera transforms, camera-frame
  point clouds, global point clouds, and a transform tree;
- one aligned binary PLY reconstruction containing positions, RGB colours,
  normals, and triangle faces.

Together, the four bags and meshes occupy approximately 16.74 GiB. Observation
indices define logical comparison order only. They are not presented as
calendar timestamps.

Open3D was considered but rejected because its current wheel does not support
the repository's Python 3.13 runtime. The implementation instead uses:

- `rosbags` to stream ROS1 bags without installing ROS;
- `plyfile` to read and write binary PLY data;
- NumPy and SciPy for voxel aggregation and nearest-neighbour search;
- Matplotlib for deterministic, headless evidence plots.

## Stage 1: Visual and structural audit

Run:

```powershell
uv run visual-memory-lab prepare-eth-office `
  --input data/eth-change-detection/office/office `
  --output outputs/phase6a/office-audit `
  --rgb-samples 24 `
  --vlm-samples 8
```

The command validates all four mesh/bag pairs, records topic and geometry
metadata, hashes each source file, and streams the bags without unpacking them.

It extracts 24 evenly spaced RGB frames from each observation, producing 96
human-viewable office images. Eight of those 24 frames are selected again for
the compact VLM contact sheet. The eight-frame selection is not a training
sample or a geometry input; it exists only to keep each view large enough for
visual review.

Open the local gallery at:

```text
outputs/phase6a/office-audit/index.html
```

The gallery shows the desk, monitors, purple cabinet, bins, windows, shelves,
floor, walls, and ceiling across the pan/tilt scan.

## Stage 2: Voxel representation

The PLY vertices are aggregated into a deterministic grid with voxel size
$r=0.02$ m, matching the spatial scale used by the original ETH experiment.
For points assigned to voxel key

$$
\mathbf k=\left\lfloor\frac{\mathbf x}{r}\right\rfloor,
$$

the implementation stores the mean position, colour, and normal, together
with the number of original mesh vertices represented by that voxel.

ETH supplies aligned reconstructions, so the implementation does not apply
ICP. An extra registration step could absorb a real object displacement and
make the change benchmark less transparent.

## Stage 3: Bidirectional geometric residuals

For earlier voxel cloud $P$ and current voxel cloud $Q$, the current-to-earlier
point residual is

$$
d_{Q\rightarrow P}(\mathbf q)
=
\min_{\mathbf p\in P}\|\mathbf q-\mathbf p\|_2.
$$

The reverse residual is

$$
d_{P\rightarrow Q}(\mathbf p)
=
\min_{\mathbf q\in Q}\|\mathbf p-\mathbf q\|_2.
$$

A large $d_{Q\rightarrow P}$ produces a `current-only` candidate. A large
$d_{P\rightarrow Q}$ produces an `earlier-only` candidate. Those names describe
geometric evidence; they do not prove that an object was added or removed.

The primary threshold is $\tau=0.05$ m. Sensitivity is also measured at 0.02 m
and 0.10 m:

$$
C_\tau^{Q\rightarrow P}
=
\{\mathbf q\in Q:d_{Q\rightarrow P}(\mathbf q)>\tau\}.
$$

The point-to-plane ablation uses the matched earlier normal:

$$
r_{\mathrm{plane}}(\mathbf q)
=
| (\mathbf q-\mathbf p^*)^\top\mathbf n_{\mathbf p^*} |.
$$

It measures disagreement normal to the surface rather than treating tangential
displacement identically.

## Stage 4: Candidate clusters

Changed voxels are grouped with 26-neighbour connectivity. Components smaller
than 20 voxels are removed. Surviving components receive stable IDs such as:

```text
eth-office:0-to-1:current-only:cluster-003
```

Run:

```powershell
uv run visual-memory-lab evaluate-eth-change `
  --manifest outputs/phase6a/office-audit/manifest.json `
  --output outputs/phase6a/change-baseline `
  --voxel-size 0.02 `
  --distance-thresholds 0.02 0.05 0.10 `
  --primary-threshold 0.05 `
  --min-cluster-voxels 20
```

All six lower-index-to-higher-index pairs are evaluated. The consecutive
logical comparisons are 0-to-1, 1-to-2, and 2-to-3.

The output includes JSON/JSONL records, coloured candidate PLY files, and
top/front/side projections. All clusters remain in the machine-readable
artifacts. The plots label the six largest clusters in each direction and
render smaller fragments faintly so that the review images remain readable
without hiding fragmentation.

## Stage 5: Required VLM review

The download does not contain spatial human annotations that this project can
use as ground truth, and the user cannot manually label all pairwise changes.
The project therefore uses a constrained VLM review:

```powershell
uv run visual-memory-lab review-eth-change `
  --baseline outputs/phase6a/change-baseline `
  --audit outputs/phase6a/office-audit `
  --output outputs/phase6a/vlm-review `
  --cache-dir outputs/phase6a/vlm-cache `
  --model gpt-5.6-terra
```

For each pair, the reviewer receives:

1. the earlier eight-frame RGB contact sheet;
2. the current eight-frame RGB contact sheet;
3. the labelled `current-only` projection;
4. the labelled `earlier-only` projection.

It must review exactly the six largest candidates in each direction. Structured
output validation rejects missing, duplicate, or invented candidate IDs and
unknown evidence citations. Requests use `store=False`; results are cached by
model, prompt version, schema, prompt, and image hashes.

The VLM assigns `supported`, `unsupported`, or `uncertain`, with a confidence
and explicit limitations. Only medium/high-confidence supported candidates
enter `pseudo_reference.json`.

This is a **VLM-supported pseudo-reference**, not ground truth. It cannot support
precision, recall, F1, or accuracy claims, and it cannot reveal changes missing
from the geometric candidate generator.

## Measured acceptance run

The real-data run produced:

| Measurement | Result |
|---|---:|
| Office observations | 4 |
| Human-viewable RGB samples | 96 |
| Pairwise mesh comparisons | 6 |
| Raw geometric candidate clusters | 917 |
| Largest candidates reviewed by the VLM | 72 |
| VLM-supported | 53 |
| VLM-uncertain | 15 |
| VLM-unsupported | 4 |
| Medium/high-confidence candidates accepted into the pseudo-reference | 47 |

Across pairs, median current-to-earlier nearest-neighbour residuals were about
0.007-0.013 m, while the 95th percentiles were about 0.096-0.173 m. Most geometry
is closely aligned, but a meaningful tail exceeds the 5 cm threshold.

The headline result is not "47 correct changes." It is:

> A simple 5 cm geometric baseline recovers visually plausible large scene
> differences, but fragments the six comparisons into 917 clusters. Candidate
> grouping, coverage, and uncertainty are now measurable bottlenecks.

The final local reports are:

```text
outputs/phase6a/office-audit/index.html
outputs/phase6a/change-baseline/index.html
outputs/phase6a/vlm-review/index.html
```

## Failure atlas

| Failure | Meaning |
|---|---|
| Fragmentation | One physical difference becomes several disconnected clusters |
| Reconstruction boundary | Small mesh disagreement appears along otherwise static surfaces |
| Missing coverage | A surface is reconstructed in only one observation |
| Thin or reflective surface | Sensor/reconstruction noise affects a narrow object |
| Threshold instability | Candidate support changes sharply between 2, 5, and 10 cm |
| Direction ambiguity | Earlier-only and current-only clusters may represent one moved object |
| VLM uncertainty | Available RGB/projection evidence cannot support a confident judgment |
| Candidate-generator blind spot | VLM review cannot recover changes that geometry never proposed |

## What Phase 6A does and does not establish

Phase 6A establishes a reproducible real RGB-D/3D change workflow, transparent
geometry, sensitivity measurements, inspectable evidence, and a failure atlas.

It does not establish supervised change-detection accuracy, visibility-aware
occlusion reasoning, object identity, real calendar time, or a learned model.
Those boundaries keep the next training experiment honest.

## Phase 6B handoff

Phase 6B will select one measured bottleneck and train against this frozen
baseline. The leading candidates are:

- learn a representation or affinity that merges fragmented clusters belonging
  to the same physical object;
- learn viewpoint/coverage-aware correspondence before differencing;
- train a small RGB-D candidate classifier using a separately labelled or
  synthetic training source, then evaluate transfer to ETH Office.

The choice will be made from the Phase 6A failure distribution rather than from
the desire to train a network for its own sake.
