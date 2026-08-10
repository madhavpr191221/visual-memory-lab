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
