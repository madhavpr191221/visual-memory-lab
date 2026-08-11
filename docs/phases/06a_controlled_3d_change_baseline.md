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

## The complete flow, in physical terms

Suppose the office is scanned twice:

```text
Earlier office scan
        +
Current office scan
        |
        v
Put both reconstructions in the same 3D coordinate system
        |
        v
Compare corresponding physical locations
        |
        v
Highlight geometry found in only one observation
        |
        v
Group those differences into candidate changes
        |
        v
Show the RGB and 3D evidence
```

Each line hides a specific perception operation. The following sections unpack
those operations and distinguish what ETH already supplies from what Visual
Memory Lab computes.

### 1. What is an office scan?

One **scan**, called an observation in this dataset, is not one photograph. It
is a sequence of synchronized sensor measurements collected while a tripod
camera pans and tilts around the room.

For image index $k$ in logical observation $t$, the sensor provides quantities
conceptually written as

$$
O_t^k = \left(I_t^k, D_t^k, T_{G\leftarrow C,t}^k\right),
$$

where:

- $I_t^k$ is the RGB image;
- $D_t^k$ is depth information, represented in the bags as point clouds;
- $T_{G\leftarrow C,t}^k$ transforms a camera-frame point into the common
  global frame $G$;
- $t\in\{0,1,2,3\}$ identifies the logical office observation;
- $k$ identifies one camera measurement within that observation.

The camera sees only part of the office at any instant. The complete scan is
the collection of those partial views.

### 2. From depth pixels to physical 3D points

For a conventional depth image, pixel $(u,v)$ with depth $z$ is back-projected
through the camera intrinsic matrix $K$:

$$
\mathbf x_C
=
zK^{-1}
\begin{bmatrix}
u\\v\\1
\end{bmatrix}.
$$

$\mathbf x_C$ is a metric point in the camera coordinate system. Its global
position is

$$
\tilde{\mathbf x}_G
=
T_{G\leftarrow C,t}^k\tilde{\mathbf x}_C,
$$

where the tilde denotes homogeneous coordinates. This transformation removes
the camera's changing pan/tilt pose: points on the same physical desk should
land near the same global coordinates even when observed from different camera
directions.

The ETH bags already contain `point_cloud_G`, so they expose points after this
camera-to-global transformation as well as the raw camera-frame cloud.

### 3. What is a reconstruction?

A single RGB-D frame contains only a partial surface measurement. A
**reconstruction** fuses measurements from the complete scan into a coherent
3D model:

$$
R_t
=
\operatorname{Fuse}
\left(
O_t^1,O_t^2,\ldots,O_t^{K_t}
\right).
$$

The original ETH system used a truncated signed distance field, or TSDF. A
TSDF stores, at each voxel, an estimate of signed distance to the closest
observed surface. Repeated depth measurements are integrated so that noisy
individual frames contribute to one denser surface model. A mesh is then
extracted from the zero crossing of that field.

Important implementation boundary:

> Visual Memory Lab does not rebuild the TSDF in Phase 6A. ETH supplies one
> complete, already aligned PLY reconstruction for each observation. Phase 6A
> begins from those four meshes.

Rebuilding and evaluating the fusion process would be a separate 3D systems
experiment.

### 4. What does “the same coordinate system” mean?

Let $G$ be the global office coordinate frame. If two reconstructions are
aligned, a physical location such as the left corner of a desk has comparable
coordinates in both:

$$
\mathbf x_G^{(t_1)} \approx \mathbf x_G^{(t_2)}.
$$

Without this property, the entire room would appear to move when the camera
pose changed. Normally an application might estimate a rigid transform

$$
T^* = \arg\min_T \sum_i
\left\|T\mathbf p_i-\mathbf q_{\pi(i)}\right\|_2^2
$$

using feature matching or ICP. Phase 6A deliberately does not do this: ETH
states that its supplied reconstructions are aligned. Applying new ICP could
reduce a genuine object displacement by treating it as registration error.

### 5. Compare corresponding physical locations

Let $P$ contain earlier surface points and $Q$ contain current surface points.
Exact point indices do not correspond across independently reconstructed
meshes, so the implementation uses the nearest surface location:

$$
d_{Q\rightarrow P}(\mathbf q)
=
\min_{\mathbf p\in P}
\|\mathbf q-\mathbf p\|_2.
$$

If this distance is small, the current surface has a nearby earlier
counterpart. If it exceeds threshold $\tau$, the surface becomes a difference
candidate:

$$
\mathbf q\in C_{\mathrm{current}}
\iff
d_{Q\rightarrow P}(\mathbf q)>\tau.
$$

The reverse comparison is equally important:

$$
\mathbf p\in C_{\mathrm{earlier}}
\iff
d_{P\rightarrow Q}(\mathbf p)>\tau.
$$

This bidirectional computation distinguishes geometry found only in the
current reconstruction from geometry found only in the earlier reconstruction.

### 6. Why the comparison is performed on voxels

Meshes contain hundreds of thousands of vertices and the two reconstructions
do not sample surfaces identically. Phase 6A first maps them to a regular 2 cm
grid:

$$
\mathbf k(\mathbf x)
=
\left\lfloor\frac{\mathbf x}{0.02}\right\rfloor.
$$

All vertices with the same integer key are represented by their mean position,
colour, and normal. This reduces redundant surface samples and gives the
clustering stage a stable spatial neighbourhood.

### 7. Highlight geometry found in only one observation

The primary threshold is $\tau=0.05$ m. Therefore, a current voxel is
highlighted when no earlier voxel lies within five centimetres. The experiment
also repeats the measurement at 2 cm and 10 cm.

The labels mean only:

- `current-only`: reconstructed here in $Q$, with no nearby surface in $P$;
- `earlier-only`: reconstructed here in $P$, with no nearby surface in $Q$.

They do **not** automatically mean “object added” and “object removed.” Missing
coverage, noise, and reconstruction holes can produce the same geometric
pattern.

### 8. Group changed voxels into candidates

One object should be more useful than thousands of isolated changed points.
Changed voxels therefore form an undirected graph

$$
\mathcal G=(V,E),
$$

where each changed voxel is a vertex and

$$
(i,j)\in E
\iff
\|\mathbf k_i-\mathbf k_j\|_\infty\le 1.
$$

This is 26-neighbour connectivity in a 3D grid. Each connected component is a
candidate cluster. Components containing fewer than 20 changed voxels are
discarded.

A moved object can produce two candidates:

```text
earlier-only cluster at the old location
             +
current-only cluster at the new location
             =
possible moved object
```

Phase 6A does not yet learn or guarantee that association.

### 9. Show RGB and 3D evidence

The geometric projection identifies *where* the reconstruction differs, but
RGB helps a reviewer decide *what is visibly there*. For each pair the system
shows:

1. eight representative RGB views from the earlier scan;
2. eight representative RGB views from the current scan;
3. top/front/side projections of current-only clusters;
4. top/front/side projections of earlier-only clusters.

The VLM reviews only named, numbered candidates from that evidence. It cannot
invent new cluster IDs, and its supported candidates remain a pseudo-reference
rather than human ground truth.

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

## How to test and inspect Phase 6A

### Fast automated checks

From the repository root:

```powershell
uv run python -m pytest
cd web
npm test
npm run build
```

The Python tests cover PLY loading, voxel aggregation, bidirectional residuals,
connected components, VLM schema validation, caching, the secure UI catalog,
and API image allowlisting. The web tests cover the evidence page and existing
retrieval interface.

### Inspect the already generated evidence

Open these files directly:

```text
outputs/phase6a/office-audit/index.html
outputs/phase6a/change-baseline/index.html
outputs/phase6a/vlm-review/index.html
```

The first proves that real RGB frames were decoded. The second shows all six
3D comparisons. The third shows the structured VLM judgments and limitations.

### Run the React showcase

Build and serve the application:

```powershell
cd web
npm run build
cd ..
uv run visual-memory-lab serve-ui
```

Then visit:

```text
http://127.0.0.1:8000/lab/changes
```

The **Changes** page is intentionally narrower than the raw experiment. It
starts with the three consecutive visit pairs and presents one curated chair
relocation story at a time:

1. an earlier and later RGB frame are shown side by side;
2. the relevant chair is highlighted in both frames;
3. the page states the cautious outcome and its identity limitation;
4. a compact crop shows the corresponding coarse 3D difference regions;
5. the raw counts, full-room projections, threshold, and claim boundary remain
   available under **How did the comparison reach this result?**

The highlighted RGB cases and orange bounding boxes are manually curated
presentation examples. No YOLO model or other object detector produced these
boxes. They are not automatic object tracks or human-labelled benchmark ground
truth. They make a
specific limitation visible: the geometry can recover chair-shaped displaced
regions, but the system does not yet assign persistent object identities.

The diagnostics section deliberately uses operational language such as
"surfaces seen only later" and "separate regions" instead of presenting voxel
residuals as verified object changes. The complete 96-frame audit, all six pair
comparisons, and every structured VLM judgment remain available in the static
research reports listed above.

### Re-run the local geometry without overwriting accepted artifacts

Use a new output directory:

```powershell
uv run visual-memory-lab evaluate-eth-change `
  --manifest outputs/phase6a/office-audit/manifest.json `
  --output outputs/phase6a/test-change-baseline
```

The expected acceptance count is 917 geometric candidates with the documented
defaults. Small floating-point differences across SciPy/platform versions
should be investigated rather than silently replacing the frozen artifact.

The VLM review is a separate paid cloud action. It should not be rerun merely
to test local geometry; use the existing cache and frozen summary unless a
prompt, model, schema, or evidence image intentionally changes.
