# Phase 6.1.1: Automatic Object Localization in Real Office Images

## Objective

Earlier experiments showed that a geometric difference is not an object. That
useful, but a geometric cluster is not an object: it cannot tell us whether a
changed region is a chair, a waste bin, a box, incomplete scan coverage, or a
reconstruction artifact.

Phase 6.1.1 adds the missing RGB object-localization layer. It asks a narrower,
testable question:

> In each office image, where does a frozen vision model predict a chair, a
> waste bin, or a box, and which pixels does it assign to that object?

The answer is saved as inspectable evidence and shown in the React interface.
This phase does **not** yet decide whether detections in two visits are the same
physical object, whether an object moved, or whether an undetected object is
absent.

## Real-world application

Imagine a facilities technician walking through an office with a camera. The
system records many overlapping views. Before it can answer, “Where was this
chair last seen?” or “Was this waste bin moved?”, it first needs to find those
objects in each frame.

Phase 6.1.1 performs that first perception step:

```text
RGB inspection frames
        ↓
locate chairs, waste bins, and boxes
        ↓
separate each predicted object from its background
        ↓
save boxes, masks, confidence scores, camera pose, and model provenance
        ↓
inspect predictions and failures in the Objects UI
```

For a commercial inspection product, this layer could support asset inventory,
workspace safety checks, housekeeping audits, and the object-level evidence
needed by later change detection. The current artifact is a research baseline,
not a deployment-quality detector.

## Data used

The input is the public ETH ASL Change Detection Office dataset. It contains
four logically ordered office observations. Each RGB message has an exactly
timestamp-aligned camera pose `T_G_C`.

The pipeline selects 96 keyframes from each observation, for 384 frames total.
The dense set is used for localization. A smaller, independent sample of
12 frames per observation—48 frames total—is used only for the optional VLM
pseudo-audit.

The four observations provide ordered revisits, not calendar dates. The code
must not turn their message timestamps into claims such as “the chair moved on
Tuesday.”

### Boundary: depth comes later

Phase 6.1.1 is an RGB-only object-localization baseline. It does not use depth to
build 3D object clouds or to decide whether an object moved. The recorded camera
pose is retained as metadata and for viewpoint-aware keyframe selection.

The next subphase will connect the RGB masks to the available depth and aligned
3D observations. That work will be documented separately when implemented.

## Keyframe selection

Running two vision models on every highly redundant video frame would waste
compute. At the same time, selecting only a few evenly spaced frames could miss
useful viewpoints. The implemented sampler therefore combines temporal spread
with camera-pose diversity.

For a requested count $K$, observation $v$ is divided into $K$ temporal
windows. The first window contributes its middle frame. In each later window,
the sampler chooses the pose most different from the last selected pose.

For candidate pose $i$ and the previous selected pose $j$, the score is

$$
d(i,j) = \max\left(
\frac{\lVert \mathbf t_i - \mathbf t_j \rVert_2}{0.10\ \mathrm{m}},
\frac{\theta(\mathbf q_i,\mathbf q_j)}{10^\circ}
\right),
$$

where $\mathbf t$ is camera translation and $\mathbf q$ is its orientation
quaternion. Quaternion angular distance is

$$
\theta(\mathbf q_i,\mathbf q_j)
= 2\cos^{-1}\left(\left|\mathbf q_i^\top\mathbf q_j\right|\right).
$$

The normalizers do not assert that 10 cm or 10 degrees is a semantic change.
They simply put translation and rotation on comparable scales for sampling.

## Detection with Grounding DINO

The detector is the frozen
[`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
checkpoint. It is an open-vocabulary detector: instead of a fixed numeric class
list, it receives a text prompt:

```text
office chair. desk chair. waste bin. trash bin. wastebasket.
cardboard box. storage box.
```

For frame $I$, the detector produces candidate tuples

$$
D(I)=\{(b_i,p_i,s_i)\}_{i=1}^{N},
$$

where $b_i=(x_1,y_1,x_2,y_2)$ is a pixel-space box, $p_i$ is the returned text
phrase, and $s_i$ is the detector confidence score. Phrases are mapped to three
canonical classes: `chair`, `waste_bin`, and `box`.

The default box threshold is 0.25 and the text threshold is 0.20. These are
baseline operating points, not calibrated probabilities.

### Duplicate suppression

Open-vocabulary prompts can produce overlapping predictions such as “office
chair” and “desk chair” on the same chair. For two boxes $A$ and $B$, intersection
over union is

$$
\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

Within each canonical class, lower-scored boxes are suppressed when
$\operatorname{IoU}>0.50$. Boxes are clamped to the image boundary, invalid
boxes are rejected, and at most 20 detections are retained per frame.

## Segmentation with SAM 2.1

A detector box is intentionally coarse. The frozen
[`facebook/sam2.1-hiera-small`](https://huggingface.co/facebook/sam2.1-hiera-small)
checkpoint receives the RGB frame and each retained Grounding DINO box as a
prompt. It predicts a binary mask

$$
M_i(x,y)\in\{0,1\}
$$

for the pixels belonging to that proposed object. The artifact stores the mask,
SAM's mask-quality score, and the image-area fraction

$$
r_i=\frac{1}{HW}\sum_{x=1}^{W}\sum_{y=1}^{H}M_i(x,y).
$$

Very small masks ($r_i<0.001$) and unusually large masks ($r_i>0.60$) receive
automatic warnings. These are review signals, not correctness labels.

## GPU and reproducibility

The models can run on CPU, but the 384-frame acceptance run is intended for the
local NVIDIA GPU. The project has mutually exclusive `cpu` and `cuda` uv extras:

```powershell
uv sync --extra cuda
```

CUDA inference uses PyTorch inference mode and BF16 autocasting. Detection and
segmentation run in separate stages so the detector can be unloaded before SAM
2.1 is loaded. This reduces peak GPU memory use.

The pipeline is resumable while work is in progress. Per-frame records live in
a temporary sibling directory. The final output appears atomically only after
all frames, detections, masks, and overlays are complete. Model IDs, downloaded
model revisions, device, dtype, thresholds, Python version, prompt, timing, and
rejected counts are recorded in `run.json`.

## Artifact layout

```text
outputs/phase6b1/object-localization/
|-- run.json
|-- frames.jsonl
|-- detections.jsonl
|-- frames/
|   `-- observation-0/ ... observation-3/
|-- masks/
`-- overlays/
```

`frames.jsonl` connects each selected RGB image to its observation, source
message index, timestamp, camera pose, rendered overlay, and detection IDs.
`detections.jsonl` stores the canonical class, original detector phrase,
scores, pixel and normalized boxes, mask path, mask area, and warnings.

Generated images and model weights remain local and are not committed.

## Acceptance run

The CUDA acceptance run processed all 384 requested frames and retained 1,417
predictions after class mapping, box validation, and within-class duplicate
suppression:

| Result | Count |
| --- | ---: |
| Frames with at least one prediction | 383 |
| Frames with no retained prediction | 1 |
| Chair predictions | 515 |
| Waste-bin predictions | 477 |
| Box predictions | 425 |
| Rejected or duplicate predictions | 414 |

These large counts are not a success metric. Direct inspection shows genuine
objects alongside false positives, especially at the permissive 0.25 baseline
threshold. That is exactly why the UI exposes threshold filtering and why this
phase records a failure audit instead of presenting every box as correct.

The optional VLM audit reviewed all 48 requested frames, covering 165
predictions. It judged 79 predictions supported, 15 uncertain, and 71
unsupported. For predictions with detector score at least 0.35, the
pseudo-support rate was 63.8%. Because the judge is not human ground truth,
this is a diagnostic result—not detector accuracy. The frozen counts and audit
boundary are recorded in
[`artifacts/phase6b1/summary.json`](../../artifacts/phase6b1/summary.json).

## VLM pseudo-audit

The ETH dataset does not provide object boxes or masks for these target classes,
so Phase 6.1.1 cannot report genuine detector precision or recall. Instead, an
optional VLM reviews a fixed sample of 12 frames from every observation. It sees
both the raw image and model-rendered overlay and must:

- judge every supplied prediction as `supported`, `uncertain`, or `unsupported`;
- assess the mask as `good`, `partial`, `excessive`, or `uncertain`;
- note any clearly visible target class missed in the raw frame;
- state limitations without inferring identity, movement, or absence.

Responses use a strict structured schema and a content-addressed cache. The
audit reports a descriptive high-confidence pseudo-support rate

$$
R_{\mathrm{pseudo}}=
\frac{N_{\mathrm{supported},\ s\ge 0.35}}
{N_{\mathrm{supported},\ s\ge 0.35}+N_{\mathrm{unsupported},\ s\ge 0.35}}.
$$

Uncertain cases are excluded from this fraction. This number is **not** model
accuracy, because the judge is another model rather than a human-labelled
ground-truth set. Its purpose is rapid failure discovery and sample curation.

## Objects UI

The active `/research/objects` page replaces the earlier hand-curated Changes
presentation as the main Phase 6 view. It provides:

- real predictions over all 384 keyframes;
- filters for visit, target class, VLM audit status, and detector score;
- raw, box-only, mask-only, and combined display modes;
- detector phrase, detector score, SAM score, mask coverage, and warnings;
- a gallery for inspecting successes, empty frames, and disagreements;
- VLM pseudo-audit counts when a complete audit artifact exists, otherwise a
  clear unavailable status and an explicit ground-truth boundary;
- expandable model and threshold details.

The UI draws boxes from model output. They are not manually curated. The
application view links to the practical object-finding workflow; this page keeps
the underlying predictions and their limits visible for review.

## How to run

Install the CUDA environment:

```powershell
uv sync --extra cuda
```

Generate the dense localization artifact:

```powershell
uv run --extra cuda visual-memory-lab localize-eth-objects `
  --input data/eth-change-detection/office/office `
  --output outputs/phase6b1/object-localization `
  --keyframes-per-observation 96 `
  --device cuda
```

Optionally generate the 48-frame VLM pseudo-audit. This is the only step below
that sends public dataset images to the configured OpenAI API:

```powershell
uv run --extra cuda visual-memory-lab audit-eth-object-localization `
  --localization outputs/phase6b1/object-localization `
  --output outputs/phase6b1/vlm-audit `
  --cache-dir outputs/phase6b1/vlm-cache `
  --frames-per-observation 12
```

Build and run the UI:

```powershell
Set-Location web
npm run build
Set-Location ..
uv run --extra cuda visual-memory-lab serve-ui
```

Open `http://127.0.0.1:8000/research/objects`.

The default runtime request is `auto`: it selects CUDA when
`torch.cuda.is_available()` is true and otherwise uses CPU. For the normal
showcase, install and run the CUDA extra:

```powershell
uv sync --extra cuda
uv run --extra cuda visual-memory-lab serve-ui
```

If no compatible GPU is available, install the CPU extra instead:

```powershell
uv sync --extra cpu
uv run --extra cpu visual-memory-lab serve-ui
```

Run the test suite with either environment; tests use fakes and do not load
the real vision checkpoints:

```powershell
uv run --extra cpu python -m pytest -q
```

## What this phase proves—and does not prove

Phase 6.1.1 proves that the project can turn dense, pose-linked real-office RGB
observations into automatically localized, inspectable object evidence using
frozen modern vision models.

It does not prove:

- that every target object was found;
- that every box or mask is correct;
- that two similar chairs are the same chair;
- that an object moved, appeared, or disappeared;
- that a missing detection means physical absence;
- that the VLM pseudo-audit is human ground truth.

Those boundaries define the next research subphases: build an honest labelled
evaluation slice, connect 2D masks to the aligned 3D observations, associate
object hypotheses across visits, and only then classify added, removed, moved,
or uncertain state changes.

## References

- Shilong Liu et al. “Grounding DINO: Marrying DINO with Grounded Pre-Training
  for Open-Set Object Detection.” *ECCV*, 2024.
  [Paper](https://arxiv.org/abs/2303.05499) ·
  [official repository](https://github.com/IDEA-Research/GroundingDINO)
- Nikhila Ravi et al. “SAM 2: Segment Anything in Images and Videos.” 2024.
  [Paper](https://arxiv.org/abs/2408.00714) ·
  [official repository](https://github.com/facebookresearch/sam2)
- Marius Fehr et al. “TSDF-based Change Detection for Consistent Long-Term
  Dense Reconstruction and Dynamic Object Discovery.” *ICRA*, 2017.
  [Paper](https://cesarcadena.ethz.ch/files/ICRA2017_mfehr.pdf) ·
  [ETH dataset page](https://projects.asl.ethz.ch/datasets/change-detection/)
