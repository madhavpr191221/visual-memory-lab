# Phases 3 and 4: The Real-Office Visual Memory System

## Why these phases belong together

Phase 3 and Phase 4 form one complete real-image system:

```text
Phase 3: build, label, and measure the memory
                         |
                         v
Phase 4: let a person search and inspect that memory
```

Phase 3 prepares the publicly available, non-commercial 7-Scenes Office
research data, creates the CLIP memory,
measures retrieval with camera poses, and produces seven VLM-assisted place
zones. Phase 4 loads those frozen artifacts into a local application. It
retrieves evidence, counts zone agreement, exposes failures, and optionally
asks a VLM to interpret a small image set selected by the user.

The phase boundary is important:

| Capability | Phase 3 | Phase 4 |
|---|---|---|
| Prepare Office observations | Creates them | Loads them |
| Build CLIP embeddings | Creates them | Queries them |
| Define the seven zones | Creates and verifies them | Displays them |
| Assign 6,000 frames to zones | Performs and freezes assignments | Looks them up |
| Evaluate using camera pose | Computes results | Presents results |
| Count zone labels in the top ten | No | Yes |
| React interface | No | Yes |
| VLM analysis of selected results | No | Yes, after confirmation |

Therefore, the **Office Zones page is a Phase 4 screen**, but the names,
descriptions, landmarks, and frame assignments on that screen were created in
Phase 3.

## The problem

A collection of images is not automatically a useful memory. A person needs to
recover the right observation even when they remember only a landmark, approach
the place from another direction, or cannot name the exact workstation.

The central question is:

> Can a frozen visual representation retrieve useful evidence from an earlier
> pass through a real office, and can the system show enough evidence for a
> person to judge the result?

This is deliberately different from building a chatbot. Retrieval comes first.
The user can inspect the images without requesting a generated answer.

## Technician example

Imagine that a maintenance technician records an office during routine rounds.
Earlier rounds become the visual memory. During a later visit, they ask:

> Where is the dual-monitor workstation directly beneath the exterior window?

The system turns the text into a CLIP vector and compares it with 6,000 stored
Office vectors. It retrieves the ten exact nearest memories. If six already
carry the frozen label `window-side-dual-monitor-workstation`, the interface
reports **moderate agreement** and shows the requested three, five, or ten
images.

That result is useful without generation: the technician can recognize the
desk, window, chair, and computer tower. If they need a visual judgment, they
may select up to five frames and explicitly ask the VLM to analyze them. The
answer must cite supplied observations and state what the images cannot prove.

The same workflow supports location, context, revisit planning, visible-state,
maintenance-evidence, comparison, and object-recall questions. The current
dataset does not support claims about real dates, who moved an object, or unseen
events.

## Part I: Phase 3 -- Building and measuring the memory

### 1. Dataset and split

The experiment uses the Office scene from the publicly available 7-Scenes
research dataset. Microsoft Research provides it for non-commercial use. It has
ten recorded sequences, each containing 1,000 RGB frames and matching camera
poses. Every RGB frame is 640 by 480 pixels.

| Role | Sequences | Frames |
|---|---|---:|
| Stored memory | 01, 03, 04, 05, 08, 10 | 6,000 |
| Held-out image queries | 02, 06, 07, 09 | 4,000 |

The query sequences never enter the stored memory. This prevents an image from
retrieving itself and turns evaluation into a revisit problem.

Preparation validates the expected RGB files and finite 4 by 4 camera-to-world
matrices. It creates lightweight manifests rather than copying 10,000 images.
Every observation records a stable ID, sequence, frame, split, RGB path, camera
matrix, and camera translation. Depth may be present but is not used.

### 2. Exact CLIP memory

Both splits are encoded with the pinned
`openai/clip-vit-base-patch32` checkpoint. Every observation becomes a
normalized 512-dimensional `float32` vector.

The 6,000 training vectors form the memory. The 4,000 test vectors supply
held-out image queries. Retrieval is exact cosine search over every stored
vector. There is no approximate index, compression, or vector database.

For normalized vectors, the dot product is the cosine score:

$$
s(q, m_i) = q^T m_i.
$$

Exact NumPy search is simple and appropriate at this scale. A vector database
can later implement the same application storage contract if the corpus grows.

### 3. Pose-grounded evaluation

Visual similarity cannot prove that two views came from the same physical
place. Phase 3 therefore uses the supplied camera poses as geometric ground
truth for image-to-image retrieval.

For query pose $T_q$ and retrieved pose $T_r$, translation error is:

$$
d_t = \|t_q - t_r\|_2.
$$

Orientation error is the geodesic rotation distance:

$$
d_R = \cos^{-1}\left(\frac{\operatorname{tr}(R_q^T R_r)-1}{2}\right).
$$

| Criterion | Translation | Orientation | Meaning |
|---|---:|---:|---|
| Strict | at most 0.25 m | at most 30 degrees | Same local viewpoint |
| Relaxed | at most 0.50 m | at most 30 degrees | Same broader work area |

A query is **covered** when at least one stored frame meets the chosen pose
criterion. Coverage matters: no representation can retrieve a valid memory
that an earlier camera route never recorded.

For a covered query, hit@$k$ asks whether one of the first $k$ results satisfies
both thresholds. The report also includes rates over all queries, pose-error
percentiles, a fixed-seed random baseline, per-sequence results, and a stride-10
sensitivity run.

### 4. Why semantic zones were added

Camera pose supports rigorous image retrieval evaluation but cannot provide a
human-readable label such as “desk beneath the exterior window.” Phase 3 adds a
small semantic vocabulary for text retrieval and later UI interpretation.

Manually labeling 6,000 frames would be slow. Independently asking a VLM about
all 6,000 frames would be costly and inconsistent. The implemented method
combines a VLM with exact camera geometry.

The results are **silver labels**: useful and auditable machine-assisted
annotations, not human ground truth.

### 5. How the seven zones were created

#### 5.1 Select pose-spaced representatives

The first frame of every training sequence is retained. Another becomes a
representative when it is at least 0.5 metres or 30 degrees from the last
retained pose in that sequence. This reduces 6,000 frames to 132 representative
views.

#### 5.2 Discover durable landmarks

`gpt-5.6-terra` examines the representatives in batches of at most six. For
each exact observation ID, it identifies durable landmarks and suggests a
concise physical-area name. The prompt prioritizes windows, desk arrangements,
bookshelves, monitors, computer towers, partitions, and fixed structure while
excluding people and movable papers.

#### 5.3 Consolidate a frozen ontology

A structured text-only pass consolidates those suggestions into seven zones:

| Frozen zone | Assigned frames |
|---|---:|
| window-side dual-monitor workstation | 1,209 |
| bookshelf between workstations | 987 |
| central aisle by bookshelf | 388 |
| interior-window paired desks | 782 |
| poster-side dual-monitor workstation | 1,326 |
| interior-window single-monitor desk | 1,155 |
| curved-monitor bookshelf workstation | 153 |
| **Total** | **6,000** |

For every zone, the VLM also creates a description, stable landmarks, and three
CLIP queries: a short name, a landmark description, and a technician question.

#### 5.4 Verify representatives

The 132 images are shown again, but the VLM can now select only one frozen zone
or `unassigned`. Low-confidence answers become `unassigned` mechanically.
Schemas ensure the response contains exactly the requested IDs and no invented
zone slug.

#### 5.5 Propagate using pose

The VLM does not independently label all 6,000 frames. Every frame can inherit
a verified representative's label only when it is within 0.5 metres and 45
degrees. If several qualify, the code minimizes:

$$
\left(\frac{d_t}{0.5}\right)^2 +
\left(\frac{d_R}{45}\right)^2.
$$

All 6,000 frames found a qualifying representative. This is 100% assignment
coverage, not proof that every semantic boundary is perfect.

#### 5.6 Freeze the artifact

`office-zones.json` stores the zone definitions, all assignments, counts,
representative evidence, model and prompt provenance, thresholds, and a source
manifest hash. It contains no API key or user-specific absolute path.

Structured VLM results are cached by model, prompt version, schema, and image
hashes. The completed run used 45 bounded requests rather than 6,000 individual
requests. Normal evaluation makes no OpenAI calls.

### 6. Text retrieval evaluation

Three prompts for each of seven zones produce 21 text queries. CLIP searches the
same 6,000-image memory. A result is relevant when its frozen zone matches the
query zone. Because these are silver labels, this benchmark stays separate from
the pose-grounded image benchmark.

### 7. Phase 3 results

#### Image-to-image retrieval

| Metric | Strict | Relaxed |
|---|---:|---:|
| Query coverage | 65.4% | 94.8% |
| Hit@1 among covered | 32.2% | 61.7% |
| Hit@5 among covered | 48.3% | 77.6% |
| Hit@10 among covered | 56.3% | 83.0% |
| Hit@1 over all queries | 21.1% | 58.5% |
| Hit@5 over all queries | 31.6% | 73.6% |
| Hit@10 over all queries | 36.8% | 78.7% |

| Top-1 error | Median | 90th percentile |
|---|---:|---:|
| Translation | 0.432 m | 0.917 m |
| Orientation | 14.48 degrees | 32.54 degrees |

#### Text-to-image retrieval

| Metric | @1 | @5 | @10 |
|---|---:|---:|---:|
| Macro hit | 33.3% | 57.1% | 71.4% |
| Macro precision | 33.3% | 41.0% | 41.0% |

CLIP often recovers the correct broad area but struggles with strict
localization among similar desks. Increasing $k$ helps because the useful place
is often in the candidate set even when it is not ranked first.

## Part II: Phase 4 -- Making the memory usable

### 8. Load, do not rebuild

At startup, Phase 4 loads the Phase 3 training index, query index, zone artifact,
aggregate evaluation, per-query results, and one matching CLIP encoder.

The application does not ask a VLM to recreate zones or reassign frames:

```text
Phase 3 creates:
zone names + descriptions + landmarks + frame assignments

Phase 4 displays:
zone cards + counts + retrieved-zone agreement
```

### 9. Local text and image retrieval

The user can search with text or upload a PNG or JPEG under 10 MB. CLIP embeds
the query and exact search compares it with all 6,000 memories. Uploaded images
are decoded in memory and are not written to disk.

The backend always retrieves ten results. The UI may display three, five, or
ten, but zone agreement always uses the same top-ten set.

Every result exposes its real image, rank, CLIP score, observation ID, sequence,
frame, and frozen zone. The API never returns absolute source paths. Images are
served only after their IDs resolve through a loaded index and their paths are
verified to remain inside the image root.

### 10. What “6 of the top 10” means

```text
question
   -> CLIP exact search
   -> top 10 observation IDs
   -> look up each frozen Phase 3 zone
   -> count the leading label
```

If six results have `window-side-dual-monitor-workstation`, the interface says
that six of ten memories point to this area. This is a deterministic local
count, not a fresh VLM opinion.

- **Strong:** at least seven of ten support the leading zone.
- **Moderate:** four to six support one unique leading zone.
- **Mixed:** weaker support or a tie.

Agreement is not a calibrated probability. It describes consistency among the
frozen labels returned by CLIP.

More precise queries often help. “Workstation beside a window” can match
exterior windows, interior openings, and nearby desks. “Dual-monitor workstation
directly beneath the exterior window” supplies landmarks that better match the
frozen vocabulary.

### 11. Optional evidence-grounded analysis

Normal retrieval never calls OpenAI. If an API key exists, the user can select
one to five results and press **Analyze selected evidence**. A confirmation
panel explains what will be sent.

The structured answer contains the question family, support decision, answer,
evidence citations, evidence strength, and limitations. The backend rejects
citations to images not supplied. A supported answer must cite at least one
selected observation. The prompt forbids calendar-time, identity, object-mover,
and unseen-event claims.

Public text-query judgments may be cached. Judgments involving an uploaded
query image are never cached because that image may be personal.

| Zone agreement | Evidence analysis |
|---|---|
| Local deterministic count | Optional cloud call |
| Uses all ten frozen labels | Uses one to five selected images |
| No generated prose | Structured answer with citations |
| No confirmation | Explicit confirmation required |

### 12. Evidence Lab

The **Evaluation** page presents Phase 3 coverage, hit@$k$, pose errors, and
per-sequence measurements. The **Failures** browser exposes all 4,000 held-out
queries with mechanical tags such as strict success, rescued at five, miss at
ten, uncovered, and large pose error. Each detail page shows the query beside
its ten exact results.

The **Office Zones** page is the visible Phase 4 presentation of the frozen
Phase 3 ontology. Each card shows the VLM-created name, description, landmarks,
and the number of pose-propagated assignments. These are interpretation aids,
not official architectural room divisions.

### 13. Architecture

| Component | Responsibility |
|---|---|
| `seven_scenes.py` | Validate and prepare real-image manifests |
| `encoder.py` | Encode text and images with frozen CLIP |
| `memory.py` | Persist vectors and run exact search |
| `real_evaluation.py` | Compute pose and zone evaluation |
| `zone_labeling.py` | Create, verify, propagate, and freeze zones |
| `memory_store.py` | Define the replaceable application storage boundary |
| `ui_service.py` | Retrieval summaries, zones, and failure browsing |
| `api_models.py` | Stable JSON contracts |
| `api.py` | HTTP validation, safe images, and built UI serving |
| `vlm_analysis.py` | Optional cloud analysis and cache policy |
| `web/` | React and TypeScript interface |

### 14. Reproduce Phase 3

```powershell
uv sync

uv run visual-memory-lab prepare-7-scenes `
  --input data/7-scenes/office `
  --output outputs/phase3/office

uv run visual-memory-lab index `
  --input outputs/phase3/office/train `
  --output outputs/phase3/train-index `
  --batch-size 32

uv run visual-memory-lab index `
  --input outputs/phase3/office/test `
  --output outputs/phase3/test-index `
  --batch-size 32

uv run visual-memory-lab label-zones `
  --input outputs/phase3/office/train `
  --output artifacts/phase3/office-zones.json `
  --cache-dir outputs/phase3/vlm-cache `
  --model gpt-5.6-terra

uv run visual-memory-lab evaluate-real-memory `
  --memory-index outputs/phase3/train-index `
  --query-index outputs/phase3/test-index `
  --zones artifacts/phase3/office-zones.json `
  --output outputs/phase3/evaluation `
  --seed 42
```

Zone generation needs an API key. Indexing and evaluation do not.

### 15. Run Phase 4

```powershell
Set-Location visual-memory-lab
uv sync

Set-Location web
npm install
npm run build
Set-Location ..

uv run visual-memory-lab serve-ui
```

Open `http://127.0.0.1:8000`. Local search and the Evidence Lab work without an
API key. Only optional evidence analysis needs one. Stop with `Ctrl+C`.

### 16. Verification

```powershell
uv run python -m pytest

Set-Location web
npm run test
npm run lint
npm run build
```

### 17. Conclusions and limits

The combined system shows that frozen CLIP can support useful broad-area
retrieval, while camera pose reveals strict localization failures. A small
VLM-assisted ontology makes the memory easier to interpret, and pose propagation
avoids thousands of independent model calls. The application keeps retrieval
local and makes cloud interpretation explicit, bounded, optional, and cited.

It does not establish that the seven zones are human ground truth, that zone
agreement is a probability, or that this dataset supports real temporal-change
claims. Similar workstations still cause perceptual aliasing, and missing route
coverage remains a hard retrieval ceiling.

Phase 5 first isolates the alignment step: retrieve a comparable observation
from one designated reference traversal and measure whether that traversal
covered the query pose. After that bridge, the next meaningful extension is a
dataset with ordered inspection rounds and real observable changes, not another
interface layer.

## References and third-party use

The repository does not redistribute 7-Scenes RGB-D files. This work is a
non-commercial research and hiring-portfolio demonstration. Dataset-derived
results are based on:

> Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio
> Criminisi, and Andrew Fitzgibbon. “Scene Coordinate Regression Forests for
> Camera Relocalization in RGB-D Images.” CVPR, 2013.

See the [official dataset page and license](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
and [Microsoft Research publication](https://www.microsoft.com/en-us/research/publication/scene-coordinate-regression-forests-for-camera-relocalization-in-rgb-d-images-2/).

CLIP attribution:

> Alec Radford et al. “Learning Transferable Visual Models From Natural
> Language Supervision.” ICML, 2021.

See the [CLIP paper](https://proceedings.mlr.press/v139/radford21a.html),
[official repository](https://github.com/openai/CLIP), and this project's
[Third-Party Notices](../../THIRD_PARTY_NOTICES.md).
