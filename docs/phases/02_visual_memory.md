# Phase 2: CLIP visual memory

## Goal

Turn the Phase 1 inspection frames into a searchable visual memory. A user can
describe a scene in text, provide another image, or select an existing
observation and retrieve the most similar past observations with their
simulator context.

This phase establishes a transparent baseline. It asks whether a frozen CLIP
representation can separate useful memories in the small symbolic MiniGrid
domain before we add task-specific metrics or learned adaptation.

## In scope

- frozen CLIP ViT-B/32 image and text encoding;
- a persistent index for the generated trajectory corpus;
- exact cosine search over every stored observation;
- text, external-image, and stored-observation queries;
- optional episode filtering;
- human-readable and JSON results;
- structural and qualitative representation checks.

## Not in scope

- CLIP fine-tuning;
- FAISS, approximate search, or vector compression;
- task-level hit rates and temporal or pose error;
- a labeled query set;
- automatic failure classification;
- a graphical interface.

Those questions remain separate so that later phases can compare their methods
against an unchanged baseline.

## Fixed decisions

- Branch: `phase/02-visual-memory`.
- Model: `openai/clip-vit-base-patch32`.
- Model revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- Stored representation: L2-normalized, 512-dimensional `float32` vectors.
- Search: exact NumPy matrix multiplication, equivalent to cosine similarity
  because query and memory vectors are normalized.
- Default device: CUDA when available, otherwise CPU.
- Default image batch size: 64.
- Default result count: five.
- Stored-observation queries exclude the query itself unless explicitly asked
  to include it.
- Nearby action context means the previous, current, and next observation
  actions, clipped at episode boundaries.
- Generated indexes and model weights remain outside Git.

## Runtime flow

### Building the memory

```text
run.json + observations.jsonl + 380 PNG frames
                         |
                         v
             validate source contract
                         |
                         v
        CLIP processor: 56x56 RGB -> 224x224 tensor
                         |
                         v
            frozen CLIP image encoder
                         |
                         v
           normalized 380x512 matrix
                         |
                         v
        embeddings.npy + records.jsonl + index.json
```

The source observations stay in their existing order. Row `i` in
`embeddings.npy` therefore belongs to line `i` in `records.jsonl`.

Before encoding, the builder checks that the run declares the same number of
observations as the JSONL file, observation IDs are unique, and every referenced
frame is a readable 56x56 RGB image. It also computes a SHA-256 fingerprint over
the manifests, records, and ordered image bytes.

The model runs without gradients. Each batch is moved to the selected device,
projected into CLIP's shared image-text space, normalized, converted to CPU
`float32`, and appended in source order. The completed artifact is published
atomically and an existing non-empty output is never replaced.

### Querying the memory

A text query is encoded by CLIP's text tower. An external image uses the same
image path as indexed frames. A stored observation reuses its existing vector,
so it does not load CLIP again.

The query vector is normalized and multiplied by the complete embedding matrix.
Episode filtering happens before ranking. Equal scores retain source order, and
`top-k` is reduced to the available candidate count when necessary.

Each result contains its rank, score, absolute image path, complete observation
record, and nearby action context. The observation record supplies the episode,
timestep, agent pose, visible objects, seed, and action.

## Translation to a real maintenance inspection

MiniGrid is not meant to look like a factory. It gives us a controlled version
of the same memory problem a maintenance technician would face after recording
many inspections over time.

The project concepts translate as follows:

| Visual Memory Lab | Real maintenance system |
|---|---|
| Episode | One inspection visit, shift, or date |
| Agent | Technician carrying a camera |
| Agent position | Technician's estimated location |
| Agent direction | Direction the camera was facing |
| Action | Technician's recent movement |
| 56x56 observation | One camera snapshot |
| Red ball or blue box | Valve, pump, panel, extinguisher, or another asset |
| Visible-object metadata | Detected assets, maintenance tags, or annotated ground truth |
| CLIP index | Searchable history of inspection images |
| Episode filter | Search within one particular inspection |
| Nearby actions | What happened immediately before and after the image |

### Recording inspection rounds

Imagine a technician inspecting the same factory once every week while wearing
a body camera:

```text
episode-001 -> inspection on August 3
episode-002 -> inspection on August 10
episode-003 -> inspection on August 17
```

Each camera frame becomes one observation. A real record might look like this:

```text
observation: inspection-003:0142
image:       blue valve beside a pipe
location:    pump room, east wall
direction:   north-east
time:        10:43:18
movement:    walking forward
```

The camera image is encoded and stored alongside this context. The result is a
history that can be searched by its visual content instead of requiring the
technician to remember an exact filename, timestamp, or inspection date.

### Asking with text

Several weeks later, the technician asks:

> Where did I see the blue valve?

CLIP places the text query in the same representation space as the stored
images. Exact search then compares the query against every recorded frame. A
real result list could look like this:

```text
1. inspection-003:0142
   blue valve visible
   pump room, east wall
   August 17

2. inspection-002:0137
   blue valve visible
   pump room, east wall
   August 10

3. inspection-001:0151
   blue valve visible
   pump room, east wall
   August 3
```

This is the real-world equivalent of querying the current corpus with `a blue
box`. In the Phase 2 acceptance run, all five top results for that query
contained the blue box. The box stands in for a visually distinctive asset such
as a blue valve housing.

### Asking with a current photograph

The technician can also photograph a component today and use that image as the
query:

```text
current photograph
        |
        v
search historical inspection memory
        |
        v
earlier views of similar equipment
```

This can be more reliable than trying to write the perfect description. The
system could respond:

> This resembles an image captured during the August 10 inspection while you
> were approaching the east side of the pump room.

Our MiniGrid image query demonstrated the same behavior. A red-ball frame
retrieved equivalent and nearby red-ball viewpoints from other episodes.

### Restricting the search to one visit

Sometimes the technician does not want to search the full history. An episode
filter can restrict the candidates to one inspection before ranking them.

This supports questions such as:

- Where did I see this valve during last Monday's inspection?
- Show me another angle from the same visit.
- What did the equipment look like before I entered the next room?

Without the filter, the question is closer to "When have I ever seen something
like this?" With it, the question becomes "Where else did I see it during this
particular inspection?"

### Why the nearby actions are useful

An isolated image may not explain how the technician reached that location.
The previous, current, and next actions provide a small temporal window around
the retrieved frame.

```text
10:43:16 - walking east through the pump room
10:43:18 - camera records the blue valve
10:43:20 - technician turns toward the pressure gauge
```

This changes the result from an unordered photograph into a small piece of the
inspection sequence. A later real system could extend this context with a few
seconds of video, spoken notes, tool use, sensor readings, or work-order events.

### Why finding the first appearance of rust is harder

The longer-term example asks:

> When was rust first visible around the blue valve beside the pressure gauge?

Phase 2 cannot answer that complete question yet. It can retrieve frames that
look related to `blue valve`, `rusty valve`, or `pressure gauge`, but finding
the first appearance requires a task-level procedure:

1. retrieve frames containing the correct valve;
2. distinguish it from similar valves elsewhere in the factory;
3. determine whether rust is genuinely visible;
4. order the relevant observations by inspection time;
5. identify the earliest positive observation;
6. verify that earlier observations do not already show rust.

The phase boundary is therefore:

```text
Phase 2
Retrieve visually related historical frames.

Phase 3
Measure whether the frames contain the correct event, asset, episode,
time, and location.

Phase 4
Explain failures such as the wrong valve or confusion between rust,
lighting, dirt, and a red warning label.
```

### What the red-ball failure means in a factory

The Phase 2 query `a red ball` mostly retrieved observations without a ball.
Manual inspection showed that the top-ranked empty views still contained the
agent's small red triangular marker. CLIP recognized red content but did not
reliably resolve the requested shape.

A real system could make the same kind of mistake when asked for a red
emergency valve. It might retrieve a red helmet, warning sticker, tool case,
fire extinguisher, or status light. Visual similarity alone cannot prove that
the correct object was retrieved.

MiniGrid gives us simulator ground truth, so we know exactly whether a ball was
visible. A factory benchmark would need comparable evidence from asset tags,
equipment IDs, inspection forms, tracking, or manually reviewed queries.

### What changes outside the simulator

The main memory flow remains the same:

```text
camera observations
        |
        v
visual encoder
        |
        v
persistent historical index
        |
        v
text or image query
        |
        v
ranked evidence with time and location
```

A real deployment would replace or extend several inputs:

- camera frames would be higher resolution;
- episodes would use inspection IDs and real timestamps;
- position could come from GPS, SLAM, indoor localization, or checkpoints;
- asset identity could come from OCR, QR codes, detectors, tags, or tracking;
- actions could come from video motion, inertial sensors, or workflow events;
- results would likely include short video windows rather than single frames;
- sensitive camera footage would require access controls and retention rules.

The controlled experiment is therefore not pretending to be the complete
factory system. It isolates its central research problem: after a technician
has observed many similar places and objects, can the system retrieve the past
observation that provides the evidence needed now?

## Commands

Generate the Phase 1 corpus if it does not already exist:

```powershell
uv run visual-memory-lab generate `
  --episodes 10 `
  --seed 42 `
  --max-steps 100 `
  --output data/trajectories/phase-01-demo
```

Build the visual memory:

```powershell
uv run visual-memory-lab index `
  --input data/trajectories/phase-01-demo `
  --output outputs/phase-02-clip-index
```

Run a text query:

```powershell
uv run visual-memory-lab query `
  --index outputs/phase-02-clip-index `
  --text "a blue box" `
  --top-k 5
```

Use a frame as an external image query and restrict candidates to one episode:

```powershell
uv run visual-memory-lab query `
  --index outputs/phase-02-clip-index `
  --image data/trajectories/phase-01-demo/episodes/episode-000/frames/0016.png `
  --episode-id episode-005 `
  --top-k 3
```

Select an indexed observation and request JSON output:

```powershell
uv run visual-memory-lab query `
  --index outputs/phase-02-clip-index `
  --observation-id episode-000:0016 `
  --top-k 5 `
  --json
```

Add `--include-self` when the query observation itself should remain a
candidate. Both index construction and model-backed queries accept `--device`;
index construction also accepts `--batch-size`.

## Acceptance results

The real acceptance run used the ten-episode Phase 1 corpus on an NVIDIA RTX
5060 Laptop GPU.

Structural checks:

- `uv run python -m pytest -q`: 20 tests passed;
- 380 observations produced a `380 x 512` matrix;
- vector norms ranged from `0.99999988` to `1.00000012` before score clipping;
- the matrix rank was 141, so the representation did not collapse;
- an indexed image retrieved itself at rank one with cosine score 1.0;
- source fingerprints, record alignment, JSON output, action context, and
  episode filtering all validated successfully.

Qualitative text results:

| Query | Top-five observation |
|---|---|
| `a blue box` | All five results contained the blue box |
| `a red ball` | One of five results contained a red ball |
| `an empty room` | Four of five results contained no visible object |
| `a corridor` | Results were dominated by the same repeated north-facing view |

The red-ball result is an important baseline failure. The first four results
were visually identical empty views from different episodes with score
`0.267292`; the first actual red-ball observation appeared fifth with score
`0.266008`. The representation is healthy, but CLIP's natural-image semantics
do not align equally well with every MiniGrid symbol. Manual image inspection
also revealed a likely source of confusion: the empty top-ranked view still
contains the agent's small red triangular marker, while the actual ball is only
a few red pixels at this resolution. The text encoder may be responding to red
content without resolving the requested shape.

Image retrieval was much stronger. Using `episode-000:0016`, which sees
`red-ball-b`, and excluding the source observation returned red-ball views in
all five positions. The first match was the equivalent viewpoint in
`episode-005:0016` with score 1.0. With an `episode-005` filter, the next result
was the adjacent red-ball view at step 17 with score `0.9957`.

## Known limits and next question

- Some frames are byte-identical or visually identical across episodes, so
  exact score ties are expected and resolved deterministically.
- Text alignment is prompt- and object-dependent. The blue box works well while
  the red ball exposes a clear domain-gap failure.
- High image similarity primarily captures repeated viewpoint and layout. It
  does not yet prove that a result is the right memory for a task.
- The index stores an absolute source path and is a local derived artifact. It
  should be regenerated after moving the corpus to another machine.

Phase 3 should convert these observations into a labeled retrieval protocol.
The immediate research question is whether visually nearest memories are also
the observations that contain the requested event, episode, time, or pose.

Phase exit decision: passed. The implementation, real-model checks, and
documentation were validated before the phase branch was merged into `main`.
