# Phase 3: Real-Image Place Memory

## Objective

Phase 3 establishes the real-office place-memory pipeline using a public
real-world dataset. It asks one concrete question:

> Given a camera image from a new pass through an office, can CLIP retrieve a
> stored image from the same physical place during an earlier pass?

The experiment uses the Office scene from the publicly available 7-Scenes
dataset, which Microsoft Research provides for non-commercial use. It does not
train or fine-tune a model. The purpose is to measure how useful the frozen
The representation must remain useful when viewpoint, camera path, clutter, and visual
appearance vary naturally.

This phase also adds a separate text-query experiment. A cloud VLM looks only
at a pose-spaced subset of training images and creates a small place vocabulary,
such as `window-side-dual-monitor-workstation`. Those labels are useful but are
not treated as measured ground truth. The report therefore keeps the two
experiments separate:

- image retrieval uses camera pose as geometric ground truth;
- text retrieval uses VLM-assisted semantic silver labels.

## Technician interpretation

Imagine that a maintenance technician records an office while performing an
inspection. The camera history from earlier inspection rounds becomes the
memory bank.

During a later round, the technician approaches a workstation from another
direction. The current image is used as a query. The memory system should
retrieve images captured near the same workstation during an earlier round,
even if the monitors occupy different parts of the image or nearby chairs and
papers have moved.

The technician can also issue a semantic query such as:

> Where is the window-side workstation with two monitors?

That query does not request the most generally office-like image. It requests
evidence associated with one recognizable area of the office.

In a deployed system, this could support comparisons across maintenance rounds,
finding the last view of an asset, reviewing changes around a workstation, or
returning to the location where a problem was observed. This phase evaluates
the memory component only; it does not claim to implement a complete technician
assistant.

## Dataset and split

The local dataset contains ten independently recorded Office sequences, each
with 1,000 RGB frames and 1,000 camera poses. RGB frames are 640 by 480 pixels.

The official split is preserved:

| Use | Sequences | Frames |
|---|---|---:|
| Stored memory | 01, 03, 04, 05, 08, 10 | 6,000 |
| Held-out image queries | 02, 06, 07, 09 | 4,000 |

The query sequences never appear in the memory index. This prevents a query
from retrieving the identical frame and makes the task a revisit experiment.

The preparation command validates all expected RGB files and finite 4 by 4
camera-to-world matrices. It writes small manifests that reference the existing
dataset paths. It does not copy 10,000 images.

```powershell
uv run visual-memory-lab prepare-7-scenes `
  --input data/7-scenes/office `
  --output outputs/phase3/office
```

Depth files are reported when present but are not opened or used.

## Observation contract

Each real-image observation contains:

- a stable ID such as `office:seq-02:000417`;
- sequence and frame number;
- official train or test split;
- RGB image path;
- camera-to-world 4 by 4 matrix;
- camera translation in metres.

The memory artifact supports a generic RGB contract and a separate image root.
Real-image queries return sequence, frame number, translation, score, and
resolved image path.

## Memory construction

Both splits use the same pinned `openai/clip-vit-base-patch32` checkpoint and
the pinned CLIP preprocessing contract. Every vector is a normalized 512-dimensional
`float32` embedding. Retrieval remains exact cosine search; no approximate index
is introduced.

```powershell
uv run visual-memory-lab index `
  --input outputs/phase3/office/train `
  --output outputs/phase3/train-index `
  --batch-size 32

uv run visual-memory-lab index `
  --input outputs/phase3/office/test `
  --output outputs/phase3/test-index `
  --batch-size 32
```

The train artifact is the stored visual memory. The test artifact supplies
held-out query embeddings. The evaluator can therefore score all 4,000 queries
without decoding or re-encoding their images.

## Pose-grounded relevance

Visual similarity alone cannot say whether a retrieved frame came from the same
physical place. The camera poses provide that check.

For query pose $T_q$ and retrieved pose $T_r$, translation error is the Euclidean
distance between their translation vectors:

$$
d_t = \|t_q - t_r\|_2.
$$

Orientation error is the geodesic distance on $SO(3)$:

$$
d_R = \cos^{-1}\left(\frac{\operatorname{tr}(R_q^T R_r)-1}{2}\right).
$$

The cosine argument is clipped to $[-1, 1]$ to avoid numerical errors near the
valid boundary.

Two relevance definitions are reported:

| Criterion | Translation | Orientation | Meaning |
|---|---:|---:|---|
| Strict | at most 0.25 m | at most 30 degrees | Same local inspection viewpoint |
| Relaxed | at most 0.50 m | at most 30 degrees | Same broader work area |

A query is *eligible* when at least one stored training frame satisfies the
criterion. Coverage is the fraction of queries that are eligible. This matters:
CLIP cannot retrieve a relevant stored image when the earlier inspection routes
never recorded one.

For eligible queries, hit@k asks whether at least one of the top $k$ retrieved
frames satisfies both pose thresholds. The report also gives the corresponding
rate over all 4,000 queries, so coverage is never hidden.

The evaluator additionally reports:

- median and 90th-percentile top-1 translation and orientation error;
- fixed-seed random retrieval as a sanity baseline;
- maximum eligible-query coverage as the pose-oracle ceiling;
- per-test-sequence results;
- a 400-query stride-10 sensitivity run.

The stride-10 result tests whether conclusions are being dominated by many
nearly adjacent video frames.

## VLM-assisted place zones

Manually naming and assigning thousands of office frames would be slow and hard
to reproduce. The labeling pipeline therefore combines exact poses with a
separate VLM.

### 1. Select representative frames

Frames are processed in sequence order. The first frame in each training
sequence is retained. A later frame becomes a keyframe when it is at least
0.5 metres or 30 degrees from the last retained pose in that sequence.

This selected 132 representatives from the 6,000 training frames.

### 2. Discover landmarks

The representatives are sent to `gpt-5.6-terra` in batches of at most six
images. The model identifies stable landmarks and proposes physical-area names.
Temporary details such as movable papers are explicitly excluded.

### 3. Freeze the ontology

A text-only consolidation pass converts the suggestions into a small fixed
vocabulary. The completed run produced seven zones:

- window-side dual-monitor workstation;
- bookshelf between workstations;
- central aisle by bookshelf;
- interior-window paired desks;
- poster-side dual-monitor workstation;
- interior-window single-monitor desk;
- curved-monitor bookshelf workstation.

### 4. Verify rather than freely rename

The representative images are shown again. This time the VLM may select only a
frozen zone or `unassigned`. Low-confidence answers become `unassigned`
mechanically.

### 5. Propagate by pose

Every training frame is compared with verified representatives. A label is
inherited only from a representative within 0.5 metres and 45 degrees. The
nearest qualifying representative is chosen using normalized translation and
orientation distance.

The completed run assigned all 6,000 training frames. The frozen artifact
contains assignment counts, representative evidence, prompt/model provenance,
and a hash of the source manifests. It contains no API key or user-specific
absolute path.

These are *silver labels*: consistent machine-assisted annotations, not human
ground truth. They are appropriate for a bounded semantic retrieval study, but
pose-grounded and text-grounded scores must not be merged into one headline
metric.

All structured responses are schema-validated and cached by model, prompt
version, schema, and image hashes. The live run made 45 bounded requests rather
than 6,000 frame-level requests. Normal evaluation performs no OpenAI API calls.

## Text-query protocol

Each zone has three frozen CLIP prompts:

1. a short place name;
2. a landmark description;
3. a natural technician question beginning with “Where”.

This gives 21 text queries over the 6,000-frame memory. A retrieved frame is
relevant when its frozen zone assignment matches the queried zone.

The report gives hit@1/5/10 and precision@1/5/10, macro-averaged across zones
and prompt styles. Assignment coverage is reported separately.

## Completed results

### Image-to-image retrieval

| Metric | Strict | Relaxed |
|---|---:|---:|
| Query coverage | 65.4% | 94.8% |
| Hit@1 among covered queries | 32.2% | 61.7% |
| Hit@5 among covered queries | 48.3% | 77.6% |
| Hit@10 among covered queries | 56.3% | 83.0% |
| Hit@1 over all queries | 21.1% | 58.5% |
| Hit@5 over all queries | 31.6% | 73.6% |
| Hit@10 over all queries | 36.8% | 78.7% |

Top-1 pose errors across all queries were:

| Error | Median | 90th percentile |
|---|---:|---:|
| Translation | 0.432 m | 0.917 m |
| Orientation | 14.48 degrees | 32.54 degrees |

The stride-10 strict run retained 65.8% coverage and produced 31.6%, 47.1%, and
57.0% covered-query hit@1/5/10. These values are close to the full-query result,
so the main conclusion is not an artifact of scoring every adjacent frame.

Strict all-query hit@1 varied across held-out sequences:

| Sequence | Coverage | Hit@1 | Hit@5 |
|---|---:|---:|---:|
| 02 | 70.4% | 28.0% | 40.3% |
| 06 | 61.4% | 18.4% | 28.6% |
| 07 | 73.4% | 17.5% | 29.4% |
| 09 | 56.3% | 20.3% | 28.1% |

### Text-to-image retrieval

| Metric | @1 | @5 | @10 |
|---|---:|---:|---:|
| Macro hit | 33.3% | 57.1% | 71.4% |
| Macro precision | 33.3% | 41.0% | 41.0% |

Zone-assignment coverage was 100%. This does not mean the labels are perfect;
it means every frame found a verified representative within the registered pose
thresholds.

## Interpretation

The relaxed result shows that frozen CLIP is often useful for recovering the
correct broad office area. Strict localization is substantially harder. A top
result may contain the same class of desks and monitors while coming from a
different workstation, which is exactly the perceptual-aliasing failure this
project is meant to expose.

Increasing $k$ helps both protocols. This suggests the useful location is often
present in the candidate set even when CLIP does not rank it first. That creates
a clear future research direction: reranking with pose history, temporal
continuity, geometry, or a stronger place representation.

Coverage also sets a real ceiling. Under the strict definition, roughly one
third of test queries have no valid training memory at all. Those are route
coverage failures, not representation failures.

## Why depth is deferred

Depth could later answer whether geometry separates visually similar
workstations, whether retrieval survives changed clutter, and whether structural
change can be distinguished from appearance change. However, 7-Scenes RGB and
depth are provided as raw streams and pixel-level fusion requires an explicit
calibration/alignment decision.

Adding depth here would mix two questions:

1. whether CLIP provides useful real-image place memory;
2. whether RGB-depth calibration and geometric representation improve it.

Phase 3 keeps the first question identifiable. A later depth ablation can use
this exact RGB result as its baseline.

## Reproduction

After preparing and indexing the two splits, generate zones once:

```powershell
uv run visual-memory-lab label-zones `
  --input outputs/phase3/office/train `
  --output artifacts/phase3/office-zones.json `
  --cache-dir outputs/phase3/vlm-cache `
  --model gpt-5.6-terra
```

Then run the offline evaluator:

```powershell
uv run visual-memory-lab evaluate-real-memory `
  --memory-index outputs/phase3/train-index `
  --query-index outputs/phase3/test-index `
  --zones artifacts/phase3/office-zones.json `
  --output outputs/phase3/evaluation `
  --seed 42
```

Generated images, indexes, API caches, and per-query results stay ignored.
The compact zone annotation is tracked because it is required to audit and
reproduce the semantic benchmark.

## Attribution and use

7-Scenes data is not redistributed by this repository. The experiment is a
non-commercial research and portfolio demonstration governed by the original
[Microsoft Research dataset terms](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).
Results based on the dataset cite:

> Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio
> Criminisi, and Andrew Fitzgibbon. “Scene Coordinate Regression Forests for
> Camera Relocalization in RGB-D Images.” CVPR, 2013.

The frozen representation is from Alec Radford et al., “Learning Transferable
Visual Models From Natural Language Supervision,” ICML 2021. See the
[paper](https://proceedings.mlr.press/v139/radford21a.html), the
[official CLIP repository](https://github.com/openai/CLIP), and the repository's
[Third-Party Notices](../../THIRD_PARTY_NOTICES.md).
