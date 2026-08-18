# Phase 10 — Learned video memory: application, architecture, and mathematics

This document describes the next version of Visual Memory Lab: a system that can
search a collection of videos for a **moment**, not just for a similar still
image. It is written from the application outward. The research components are
there to make the application reliable and measurable.

The running example is an inspection assistant. A technician may have hours of
camera footage and ask:

> “When did the person open the cabinet?”

or:

> “Show the last time the operator picked up the drill.”

The system should return a short, playable time window and explain why that
window was selected. It should not claim that an event never happened merely
because retrieval found no result.

## 1. What the system is trying to do

The input is a video collection and, when available, weak supervision such as
action annotations or captions. The output is an ordered list of evidence
windows:

```text
question → likely time window → playable video evidence → explanation and limits
```

For an office or factory this can support:

- finding when a maintenance action occurred;
- reviewing whether a safety step was visible in a recording;
- locating the last visible use of a tool or machine control;
- comparing a current inspection with an earlier one;
- handing a human reviewer a small evidence clip instead of an entire shift.

The current Charades implementation is the first application slice. It searches
the official action annotations and returns the corresponding MP4 windows. The
learned model described here is the next step: it will learn to connect the
visual content of a window with the language used in a question.

## 2. End-to-end architecture

```mermaid
flowchart LR
    A[MP4 videos] --> B[Dataset preparation]
    A2[Action labels / captions] --> B
    B --> C[Temporal windows\n4 s length, 2 s stride]
    C --> D[Sample 16 ordered RGB frames]
    D --> E[Frozen CLIP image encoder]
    E --> F[Frame embedding cache]
    F --> G[Three-head temporal encoder]
    G --> H[Retrieval vector, action scores,\nboundary estimate]
    H --> I[Vector index]
    Q[User question] --> J[CLIP text encoder]
    J --> K[Query embedding]
    K --> I
    I --> L[Ranked and grouped events]
    L --> M[FastAPI application]
    M --> N[Video memory UI]
    N --> O[Timestamped playback and RGB evidence]
    O -. optional .-> VLM[VLM answer with citations]
```

There are two paths:

1. **Offline path:** videos are converted into windows, frames, embeddings, and
   an index. Expensive model work happens once.
2. **Online path:** a user question is embedded, compared with stored window
   embeddings, and rendered as evidence in the UI.

This separation matters in a real deployment. A technician should not wait for
the system to decode and encode an entire video collection every time they ask
the same question.

### Offline and online flow

```text
OFFLINE (build memory)
videos + labels
      ↓
window manifest
      ↓
16 RGB frames per window in the learned pipeline
      ↓
CLIP frame embeddings
      ↓
temporal model
      ↓
window vectors + searchable index

ONLINE (answer a question)
question
      ↓
text embedding
      ↓
nearest windows
      ↓
timestamped MP4 evidence
      ↓
human-readable answer with limits
```

## 3. What is a temporal window?

A **temporal window** is a short consecutive part of a video. In this phase the
initial setting is:

- window duration: (4) seconds;
- new window every (2) seconds;
- overlap between neighboring windows: (2) seconds;
- sampled frames per window: (T=16) for the learned pipeline. The original
  annotation baseline used eight frames and remains useful for comparison.

If a video begins at time (0), the first windows are approximately

```text
0–4 s, 2–6 s, 4–8 s, 6–10 s, ...
```

The overlap reduces boundary misses. An action beginning at 3.8 seconds is
likely to appear in both the 0–4 second and 2–6 second windows instead of being
split across only one badly timed clip.

### Why sixteen frames in the learned pipeline?

Sixteen is an engineering choice, not a law. For a 4-second window, it is
roughly one frame every quarter second, or about 4 frames per second. That gives
the temporal model more evidence for short actions such as:

```text
hand approaches cup → hand grips cup → cup is lifted → person drinks
```

while keeping the first experiment small enough to run on a normal GPU. More
frames improve fine timing but increase decoding, memory, and training cost.

The system can still compare (T=8,16,32) using the same evaluation protocol.

## 4. Data model

Each video and each window receives an explicit record.

### Video record

For video (i), store:

$$
V_i = (\text{video\_id},\ \text{split},\ \text{duration},\ \text{source\_path}).
$$

Example:

```text
video_id: EBtD6
split: test
duration: 30.0 s
source_path: videos/EBtD6.mp4
```

### Window record

For window (j), store:

$$
W_j = (\text{video\_id}, t_s, t_e, F_j, Y_j, O_j),
$$

where (t_s) and (t_e) are the start and end times, (F_j) is the ordered
frame list, (Y_j) contains action labels, and (O_j) contains object labels.

Example:

```text
window_id: EBtD6:0004
video_id: EBtD6
start: 12.0 s
end: 16.0 s
frames: [f0, f1, ..., f7]
actions: [eating a sandwich, taking food]
objects: [dish, food, plate, sandwich]
```

### Action interval

If an annotation says that action (k) is visible from (a_k) to (b_k), store:

$$
A_k = (\text{action\_id},\ \text{name},\ a_k,\ b_k).
$$

The annotation is supervision for training and evaluation. It is not a claim
that every frame in the interval is equally informative.

## 5. What happens to the eight frames?

The text does **not** get copied into each frame. The eight frames first become
eight visual vectors:

$$
\mathbf{z}_t=f_{\text{image}}(x_t),\qquad t=1,\ldots,T,
$$

where (x_t) is frame (t), (f_{\text{image}}) is the CLIP image encoder,
and $\mathbf{z}_t$ is its embedding vector.

For the example “sitting on a chair and drinking coffee,” the ordered sequence
could be interpreted as:

```text
frame 1: person approaches chair
frame 2: person begins sitting
frame 3: person is seated
frame 4: hand moves toward cup
frame 5: cup is held
frame 6: cup reaches mouth
frame 7: person drinks
frame 8: cup moves away
```

The annotation or caption describes the **window as a whole**. In the learned
pipeline $T=16$; the eight-frame list above is only a compact illustration. The
temporal model learns from the order of the visual vectors which parts of the
window support that description.

For finer timing, the future training data can include per-frame or per-subwindow
labels. That enables a second output such as “the action is most likely between
12.5 and 14.0 seconds.”

## 6. Turning a question into a vector

The user’s text is encoded by the CLIP text encoder:

$$
\mathbf{q}=f_{\text{text}}(r),
$$

where $r$ is the question or a normalized search phrase. For example:

```text
question: “When did the person open the door?”
search text: “a video frame showing a person opening a door”
```

The application may keep both forms. The original question is useful for the
user interface; the normalized phrase makes the retrieval intent explicit.

For a labelled window, create a target description such as:

```text
A person is sitting on a chair and drinking coffee.
```

This target text is used during training. It is not attached independently to
all eight frames.

## 7. Learning a whole-window representation

The temporal encoder receives the ordered frame vectors. Bold symbols denote
vectors throughout this document:

$$
\mathbf{h}_1,\ldots,\mathbf{h}_T =
g(\mathbf{z}_1,\ldots,\mathbf{z}_T),
$$

where $g$ is the small Transformer or temporal pooling block. After pooling,
the window vector is

$$
\mathbf{r}=\operatorname{Pool}(\mathbf{h}_1,\ldots,\mathbf{h}_T),
\qquad
\mathbf{v}_{\text{window}}=
\operatorname{Normalize}(W_o\mathbf{r}).
$$

The model has a retrieval output and task-specific outputs:

1. $\mathbf{v}_{\text{window}}$: one vector for searching the entire four-second
   window;
2. action and boundary scores for the requested event.

**Normalize** means divide a vector by its Euclidean length:

$$
\operatorname{Normalize}(\mathbf{x})
=\frac{\mathbf{x}}{\lVert\mathbf{x}\rVert_2}.
$$

The result has length one. For the technician, this makes “opening a cabinet”
compare by direction in the learned space rather than by the raw size of the
stored vector.

The first answers “which clip is relevant?” The second helps answer “when in
that clip did it happen?”

## 8. Training objective

For a batch of $B$ video windows and their matching text descriptions, let

$$
S_{ij}=\frac{\mathbf{v}_i^{\top}\mathbf{q}_j}{\tau}.
$$

be the scaled cosine similarity between video $i$ and text $j$, where $\tau$ is
a temperature parameter. Smaller $\tau$ makes ranking mistakes more costly.

The symmetric contrastive loss is

$$
\mathcal{L}_{\text{contrastive}} =
\frac{1}{2}\left[
\operatorname{CE}(S,\text{video-to-text targets})+
\operatorname{CE}(S^\top,\text{text-to-video targets})
\right].
$$

In plain English: the correct description should be close to its own window
and farther from other windows in the batch.

If action labels $\mathbf{y}_t$ are available for each temporal slice, add a
multi-label classification loss:

$$
\mathcal{L}_{\text{action}} =
\frac{1}{T}\sum_{t=1}^{T}
\operatorname{BCE}(\hat{\mathbf{y}}_t,\mathbf{y}_t).
$$

The combined objective is

$$
\mathcal{L} =
\mathcal{L}_{\text{contrastive}}
 + \lambda\mathcal{L}_{\text{action}}.
$$

The parameter $\lambda$ controls how much the experiment values retrieval
quality versus temporal action localization.

## 9. The three-head temporal model

The learned pipeline keeps CLIP frozen and trains a small temporal model on the
ordered frame vectors. It has one shared temporal backbone and three outputs:

1. **Retrieval head**: a normalized vector used to find relevant windows.
2. **Action head**: one score per Charades action, because a window may contain
   several actions at once.
3. **Boundary head**: two normalized numbers estimating where the requested
   action begins and ends inside the four-second window.

For a window with hidden sequence
$\mathbf{H}=(\mathbf{h}_1,\ldots,\mathbf{h}_T)$, the pooled representation is
$\mathbf{r}=\operatorname{Pool}(\mathbf{H})$. The heads are:

$$
\mathbf{v}=\operatorname{Normalize}(W_r\mathbf{r}),\qquad
\hat{\mathbf{y}}=W_a\mathbf{r}+\mathbf{b}_a,\qquad
\hat{\mathbf{b}}=\sigma(W_b\mathbf{r}+\mathbf{b}_b),
$$

where $\mathbf{v}$ is the retrieval vector, $\hat{\mathbf{y}}$ contains action
logits, and $\hat{\mathbf{b}}=(\hat{s},\hat{e})$ contains normalized start and
end estimates. The
sigmoid keeps the boundary values between zero and one; they are converted back
to seconds using $t_s+(t_e-t_s)\hat{s}$ and
$t_s+(t_e-t_s)\hat{e}$.

Charades action annotations provide the multi-label target
\(\mathbf{y}\in\{0,1\}^C\).
For the boundary target, the best-overlap annotated action is mapped into the
window:

$$
s=\operatorname{clip}\left(\frac{a-t_s}{t_e-t_s},0,1\right),\qquad
e=\operatorname{clip}\left(\frac{b-t_s}{t_e-t_s},0,1\right).
$$

The training objective is:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{retrieval}}
 +\lambda_a\mathcal{L}_{\mathrm{action}}
 +\lambda_b\mathcal{L}_{\mathrm{boundary}},
$$

with binary cross-entropy for the multi-label action head and Smooth L1 for
valid boundary targets. In plain English: the model learns what a window is
about, which named actions are visible, and approximately where the strongest
action occurs. It does **not** learn persistent object identity, depth, or a
guarantee that an action occurred outside the recorded evidence.

## 10. Event grouping and evidence intervals

Retrieval produces overlapping four-second windows. Showing all of them would
make one event look like many events, so the API groups windows from the same
recording when their intervals overlap substantially or share the same action
label. A grouped event stores its member window IDs and an event interval.

The event interval is the model's refined action interval when boundary estimates
are available. The **context interval** is wider and is used only for playback:

$$
I_{\mathrm{context}}=[\max(0,s-\delta),\ \min(T,e+\delta)],
$$

where $\delta=2$ seconds in the current UI. This lets a technician see what
happened immediately before and after without changing the reported event time.

## 11. Retrieval mathematics

At query time, encode the question as a normalized vector $\mathbf{q}$ and
compare it with each stored normalized window vector $\mathbf{v}_j$:

$$
s(\mathbf{q},\mathbf{v}_j)=\mathbf{q}^{\top}\mathbf{v}_j.
$$

Because both vectors are normalized, this dot product is cosine similarity.
The top-$K$ windows are

$$
\operatorname{TopK}(\mathbf{q})=
\operatorname{argsort}_{j}\ s(\mathbf{q},\mathbf{v}_j).
$$

The result contains the video identifier and the time interval, so the UI can
seek directly to the relevant evidence rather than returning an unexplained
number.

The first implementation can use an exact flat index. Later experiments can
compare HNSW, LSH, IVF, and product quantization without changing the user-facing
question flow.

## 12. Application flows

### A. Search the whole video archive

```text
Technician asks: “When did the person open the cabinet?”
        ↓
Text embedding
        ↓
Search all stored windows
        ↓
Return top clips with timestamps
        ↓
Technician watches the evidence and confirms the action
```

### B. Search inside one selected video

```text
User chooses a camera recording
        ↓
Question is restricted to that video
        ↓
Rank only its windows
        ↓
Show the best matching moment and nearby context
```

This avoids an answer from the wrong camera or room when several recordings look
similar.

### C. Inspection hand-off

```text
retrieve a likely moment
        ↓
show the clip and annotation/evidence
        ↓
state confidence and missing coverage
        ↓
ask for manual confirmation when the result is ambiguous
```

The system is an evidence finder, not an autonomous safety certifier.

## 13. API and UI boundary

The application layer exposes a small interface:

```text
GET /api/video-memory?q=when did the person open the door
GET /api/video-memory/videos/{video_id}
```

A result should contain:

```text
window_id
video_id
start_seconds
end_seconds
score
matched_actions
matched_objects
video_url
explanation
limitations
```

The Video memory page presents the question box, candidate moments, playable
clips, timestamps, and a short explanation. It should make clear whether the
result came from annotation matching, learned retrieval, or both.

## 14. Offline artifacts and reproducibility

The expensive and important intermediate outputs are saved explicitly:

```text
outputs/charades/
  subset/manifest.jsonl
  windows/windows.jsonl
  frames/<window_id>.json
  embeddings/<window_id>.npy
  checkpoints/temporal_encoder.pt
  index/window_index.*
  reports/evaluation.json
```

Each artifact records the model name, frame count, window length, stride, split,
and configuration. This prevents a result from being impossible to reproduce
later because the sampling rule silently changed.

### Pilot and full artifacts

The older annotation baseline contains 5,883 windows from a smaller
300-train/100-test preparation. The learned manifest contains 18,994 windows
from 1,000 train and 300 test videos. A 100-video pilot is preserved under
`outputs/charades/learned/pilot/`; the complete run is kept under
`outputs/charades/learned/full/`.

Training and indexing use only the training split. Evaluation queries come only
from the held-out test split. The cache may contain both splits so that one
reproducible artifact stores the complete decoding result, but test windows are
never used to train the temporal head or populate the retrieval index.

The cache is resumable: each video writes an atomic chunk, and an interrupted
run can continue with `--resume` without recomputing completed videos. PyAV
decodes the compressed video on the CPU; the CLIP image and text encoders run on
CUDA when available, with CPU fallback.

### Full-run result: three-head checkpoint

The full cache produced 18,994 windows from 1,300 videos with no failed video
decodes. The three-head training-only index contains 14,824 windows. On 4,170
held-out test queries, the run achieved:

| Metric | Result |
| --- | ---: |
| Recall@1 | 0.6763 |
| Recall@5 | 0.9113 |
| Recall@10 | 0.9321 |
| Mean temporal IoU | 0.2598 |
| Median temporal IoU | 0.1979 |
| Mean boundary error | 7.23 s |
| Median boundary error | 6.50 s |
| Mean duplicate rate | 0.1650 |
| Misses | 283 |

For comparison, the earlier one-head checkpoint reached Recall@1 0.6360,
Recall@5 0.8746, Recall@10 0.9173, and 345 misses. The three-head checkpoint
improves retrieval recall and reduces misses in this run. Temporal IoU and
boundary error change only slightly, so the boundary head is a useful first
experiment rather than a production-quality localizer.

The gap between retrieval recall and temporal IoU is important: the learned
system usually retrieves a semantically relevant activity, but the current
four-second windows are not precise enough to identify the exact action
boundary.

## 15. Evaluation that matters to the application

The primary question is not “did the loss go down?” It is “did the returned
evidence contain the requested moment?”

Useful measures include:

- **Recall@K:** whether a relevant annotated window appears among the first (K)
  results;
- **temporal intersection-over-union:**

  $$
  \operatorname{IoU}_t =
  \frac{|I_{\text{pred}}\cap I_{\text{true}}|}
       {|I_{\text{pred}}\cup I_{\text{true}}|};
  $$

- **boundary error:** difference between predicted and annotated start/end times;
- **duplicate rate:** how often overlapping windows from the same event fill the
  top results;
- **no-result safety:** whether the UI clearly says “not found in indexed
  evidence” rather than “the action never happened”;
- **split discipline:** no video from the test set is used to train the encoder.

For a technician, a result that is second-ranked but within one second of the
true action may be more useful than a visually similar clip from the wrong room.

## 16. Failure boundaries

Every result needs a safe interpretation.

| Failure | Likely cause | Safe interpretation | Next experiment |
|---|---|---|---|
| Similar-looking clip from the wrong activity | visual aliasing | appearance alone was insufficient | train temporal model and add action-aware reranking |
| Correct video, wrong moment | coarse four-second window | video retrieval worked but timing is imprecise | add frame-level action head or shorter windows |
| Action is missed | sparse sampling, occlusion, or weak labels | absence is unproven | increase frames, inspect neighboring windows, improve labels |
| Correct action but duplicate clips dominate | overlapping windows | evidence is redundant, not necessarily wrong | non-maximum suppression over time |
| Question has no lexical annotation match | wording mismatch | baseline has no matching label; event may still exist | learned text-video retrieval |
| Model scores a visually similar action | representation shortcut | similarity is not proof of the requested action | add hard negatives and action-specific evaluation |

The project should say what measurement exposed the problem instead of simply
saying “the model failed.”

## 17. Relationship to the office memory system

The two datasets support different kinds of memory:

```text
ETH Office:     where is this place, and what does it look like?
Charades:       when did this action happen in this recording?
Future system:  where and when did an operational event happen?
```

The office pipeline is mainly spatial/place memory: RGB observations, place
zones, camera pose, object evidence, and cross-visit comparison. Charades adds
temporal/action memory: ordered frames, timestamps, action intervals, and video
retrieval.

Together they suggest a future spatiotemporal memory record:

$$
M = (\text{appearance},\ \text{place},\ \text{time},\ \text{action},\ \text{evidence}).
$$

Depth and 3D are useful when the question depends on physical geometry—distance,
occupancy, object relocation, or whether two views show the same place. They are
not required for the first Charades action-search experiment.

## 18. Future extensions

Once the baseline is understood, the project can grow in controlled steps:

1. train retrieval, action, and boundary heads while keeping CLIP frozen;
2. fine-tune the visual encoder on hard action negatives;
3. compare 8, 16, and 32 frame sampling;
4. add audio when sound carries operational evidence;
5. add object detection and tracking for questions about a particular tool;
6. add depth/3D for geometry and cross-view identity;
7. compare flat, HNSW, LSH, IVF, and product-quantized indexes;
8. replace public action data with a small controlled inspection dataset.

Each extension should preserve the same application contract: return evidence,
show the time, explain the match, and state what the system cannot establish.

## 19. Current status and boundary

Implemented now:

- deterministic Charades subset and temporal-window manifest;
- 4-second windows with 2-second stride;
- deterministic frame timestamp manifests;
- cached frozen CLIP frame/text embeddings;
- trainable temporal retrieval head and checkpoint export;
- three-head temporal model for retrieval, multi-label action scoring, and boundary regression;
- exact learned temporal index;
- learned API retrieval with annotation-based fallback;
- retrieval-mode labels in the Video memory UI;
- Recall@K, temporal IoU, boundary-error, duplicate-rate, and miss reporting;
- timestamped MP4 playback in the Video memory UI;
- a PyTorch temporal encoder and contrastive-loss implementation.

Not implemented yet:

- final CLIP vision-block fine-tuning from raw video;
- VLM-generated Charades captions;
- VLM-grounded video answer synthesis from sampled RGB evidence is now
  available as an optional post-retrieval step;
- reliable production-quality frame-level action boundaries;
- audio, depth, or 3D fusion.

That distinction is important in an interview. The current application is a
working, evidence-linked temporal retrieval system with a frozen-CLIP training
path. The three-head model is a first learned localization experiment and must
still be evaluated against the annotation baseline. The VLM explains selected
evidence; it is not the source of temporal ground truth.

## 20. VLM synthesis after retrieval

The application separates event retrieval from answer writing. The learned
temporal model identifies a candidate event and estimates its interval. The
VLM receives a small set of timestamped RGB frames from that interval, the
candidate action labels, and the evidence IDs. It then produces a readable
answer with citations and limitations.

The VLM does not search the whole recording, invent timestamps, or replace the
event interval. If it is unavailable, the application falls back to the
official Charades annotations and labels that answer as a dataset-grounded
fallback.

The application uses two synthesis moments:

1. a short preview for the strongest retrieved event;
2. a detailed explanation after the user selects an event.

This keeps the technician-facing answer natural while preserving a traceable
path from question to temporal event to RGB evidence.

The synthesis endpoint is:

```text
POST /api/video-memory/synthesize
```

It receives a video ID, question, event interval, and evidence window IDs. The
server samples six RGB frames from that interval and sends only those frames,
the official action labels, and the evidence IDs to the VLM. The response must
contain an answer, confidence, evidence citations, and limitations. If the API
key is missing or the call fails, the request returns an annotation-grounded
fallback rather than inventing an answer.

### Rebuilding the learned artifacts

The learned run can be rebuilt with sixteen frames per window:

```powershell
uv run visual-memory-lab build-charades-frames `
  --manifest outputs/charades/learned/windows/windows.jsonl `
  --output outputs/charades/learned/frames16 `
  --frames-per-window 16

uv run visual-memory-lab build-charades-video-cache `
  --manifest outputs/charades/learned/frames16/frames.jsonl `
  --output outputs/charades/learned/full16/cache `
  --device cuda

uv run visual-memory-lab train-charades-video `
  --cache outputs/charades/learned/full16/cache `
  --output outputs/charades/learned/full16/train `
  --device cuda

uv run visual-memory-lab index-charades-video `
  --cache outputs/charades/learned/full16/cache `
  --checkpoint outputs/charades/learned/full16/train/temporal_multitask.pt `
  --output outputs/charades/learned/full16/index `
  --device cuda
```

The index contains only training windows. Held-out windows remain available for
evaluation and are never used to train the temporal heads.
