# Phase 6.1.2: RGB-D Object Evidence

## Objective

Phase 6.1.1 can say where a model sees a chair, bin, or box in an RGB image.
This phase connects that prediction to the office scan's recorded 3D evidence.

For a technician, the useful question is:

> “How does the visible evidence for this object differ between two visits?”

The answer is deliberately limited. The system shows RGB masks, coloured
point-cloud evidence, and approximate room-frame coordinates. It does not claim
that two detections are the same physical object or that an object moved.

## What the ETH data actually provides

The ETH Office bags do not provide a simple depth image plus camera intrinsics.
They provide synchronized `color_image`, `point_cloud_D`, `point_cloud_G`, and
pose topics. `point_cloud_G` already contains points in the shared world frame
and stores an RGB colour with each point.

The implementation therefore links a segmentation mask to the RGB-coloured
points and reports the matching world-frame points. This is approximate visible
geometry, not a complete 3D object reconstruction.

## Pipeline

```text
frozen RGB detection and mask
          +
ETH RGB-coloured world point cloud
          ↓
colour-linked visible point subset
          ↓
robust centroid and 3D extent
          ↓
side-by-side visit comparison in the UI
```

For selected points (P = \{p_i\}_{i=1}^{n}), the reported centroid is the
coordinate-wise median:

$$
\bar p =
\begin{bmatrix}
\operatorname{median}_i(p_{i,x})\\
\operatorname{median}_i(p_{i,y})\\
\operatorname{median}_i(p_{i,z})
\end{bmatrix}.
$$

The visible extent uses the 5th and 95th percentiles rather than the most
extreme points, so a few noisy points do not dominate the summary.

## Technician example

Suppose Visit 0 contains a chair beside a desk and Visit 1 contains a chair
near a window. The UI shows the two RGB frames, each mask, the number of linked
3D points, and each visible centroid in the room coordinate frame.

The responsible conclusion is:

> “The visible chair evidence appears in different room positions. This phase
> does not establish that the detections are the same chair or prove movement.”

That boundary is intentional. Same-object matching and movement claims are the
next research problem, not something to smuggle into a depth visualization.

## Running it

```powershell
uv run visual-memory-lab build-eth-rgbd-evidence `
  --input data/eth-change-detection/office/office `
  --localization outputs/phase6b1/object-localization `
  --output outputs/phase612/rgbd-evidence
```

Open `http://127.0.0.1:8000/research/evidence` after starting the UI.

## Results and limitations

The acceptance run creates 1,417 evidence records from the 384 Phase 6.1.1
keyframes. 1,124 records contain non-empty linked point-cloud evidence.

The colour-linking step can include background points with similar colours. A
low point count does not prove that an object is absent; it may indicate weak
mask quality, missing point-cloud coverage, or ambiguous colours.

The phase does not train a new model, infer calendar time, track identities, or
classify objects as added, removed, or moved.
