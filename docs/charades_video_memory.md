# Charades video memory: implementation, data contracts, and evaluation

This is the implementation-grounded reference for the Charades video-memory
pipeline. It answers four questions precisely:

1. What comes from the dataset, and how is it represented?
2. What is the input and output of each preparation and model stage?
3. How does a natural-language question become a timestamped result?
4. Which measurements tell us whether retrieval and localization worked?

The application goal is deliberately modest: a user asks about a recording and
receives a small set of playable time ranges. The answer remains tied to the
video and to the official annotations. A fluent explanation is not treated as
ground truth.

## 0. The whole system in one picture

There are two timelines in this project. The **offline timeline** prepares and
learns from the recordings. The **online timeline** answers one user question.
Keeping them separate matters: the expensive video decoding and CLIP encoding
should happen once, while a question should return quickly from saved vectors.

```mermaid
flowchart LR
    A[Charades MP4 files] --> B[Official CSV annotations]
    A --> C[Four-second overlapping windows]
    B --> C
    C --> D[16 RGB timestamps per window]
    D --> E[PyAV decode and CLIP ViT-B/32]
    E --> F[16 frame vectors, each 512-D]
    F --> G[Temporal Transformer]
    G --> H[Window vector plus action and boundary heads]
    H --> I[Exact train-only index]

    Q[User question] --> J[CLIP text vector]
    Q --> K[LLM label normalization]
    K --> I
    J --> I
    I --> L[Rank and group candidate windows]
    L --> M[Playable interval and timestamped frames]
    M --> N[Optional VLM explanation]
```

In plain language: the model does not search raw pixels every time the user
asks a question. It searches a memory of previously computed window vectors,
then returns the original video as evidence.

## 1. Source data and provenance

The local dataset is `data/Charades_v1_480`. Preparation code lives in
`src/visual_memory_lab/charades.py`; learned video preparation, training, and
evaluation live in `src/visual_memory_lab/learned_video.py`; the temporal model
is in `src/visual_memory_lab/temporal.py`.

The Charades CSV provides several different annotation sources. They are kept
separate rather than concatenated into one pseudo-caption:

| Source field | Meaning | How the pipeline uses it |
| --- | --- | --- |
| `id` | Recording identifier, for example `X226B` | Locates the MP4 and identifies evidence |
| `length` | Full recording duration in seconds | Builds valid time windows |
| `script` | One dataset-written summary/script | Recording context and UI display |
| `descriptions` | One or more dataset-written descriptions | Context text; not model-generated |
| `objects` | Video-level object list | Context and object hints; not per-frame truth |
| `actions` | Action class IDs plus start/end seconds | Window labels and evaluation reference |
| `split` | Official train or test assignment | Prevents train/test leakage |

An action string such as `c077 21.00 27.20` is parsed into an action ID and
interval, then mapped through `Charades_v1_classes.txt` to a name such as
`Putting a pillow somewhere`. The semicolon in `descriptions` separates
independent dataset descriptions. It does not indicate two retrieval passes,
and no VLM generated those sentences.

One prepared recording has this conceptual schema:

```text
video = {
  video_id, split, video_path, subject, scene, length_s,
  script, descriptions[], objects[],
  actions[{action_id, action_name, start_s, end_s}]
}
```

Objects are recording-level hints. If `laptop` is listed, that does not mean a
laptop is visible in every frame or every temporal window.

## 2. Window construction and labels

The preparation command first builds four-second windows with a two-second
stride. For recording duration $T$, window length $W=4$ and stride $S=2$:

$$
I_i = [s_i,e_i],\qquad
s_i=iS,\qquad e_i=\min(s_i+W,T).
$$

Thus a 30-second recording produces approximately $[0,4]$, $[2,6]$,
$[4,8]$, and so on. Windows overlap intentionally: an action crossing a
boundary can be represented in more than one window.

Action $A_k=[a_k,b_k]$ is attached to a window when the intervals overlap:

$$
b_k>s_i\quad\text{and}\quad a_k<e_i.
$$

This produces a multi-label problem. For example, if a window overlaps
`Holding a laptop` and `Someone is running somewhere`, both labels are valid;
neither is removed just because the other exists.

### Formula beside a video example

| Mathematical view | What the video looks like |
| --- | --- |
| Window $I_i=[2,6]$ | The player is showing seconds 2 through 6. |
| Action $A_1=[3,5]$ | The person holds a laptop from seconds 3 through 5. |
| Action $A_2=[4,7]$ | The person is also running from seconds 4 through 7. |
| $5>2$ and $3<6$ | `Holding a laptop` belongs to the window. |
| $7>2$ and $4<6$ | `Someone is running somewhere` also belongs to it. |
| Target $\mathbf{y}=[1,1,0,\ldots]$ | Two action entries are on at the same time. |

The labels are not saying that the whole four seconds are identical. They say
that the window contains evidence for those actions somewhere inside it. This
is why overlapping windows and later boundary estimation are necessary.

The preparation flow is:

```mermaid
flowchart TD
    A[One CSV row] --> B[Parse video metadata]
    B --> C[Parse action IDs and start/end seconds]
    C --> D[Create 4 s windows every 2 s]
    D --> E{Does an action overlap this window?}
    E -- yes --> F[Attach every overlapping action]
    E -- no --> G[Keep description or object fallback text]
    F --> H[Window JSONL record]
    G --> H
```

Each window record contains:

```text
window = {
  video_id, split, start_s, end_s, actions[], objects[],
  description, text, timestamps_s[16]
}
```

The deterministic training text is made by `window_text`:

1. If structured action labels exist, use `A person is ...` followed by the
   action names.
2. Otherwise use the full recording description.
3. Otherwise use the object list.
4. Otherwise use a generic fallback.

This text is a stable CLIP target, not a generated caption. The original
`script`, descriptions, actions, and objects remain available as separate
metadata fields for display and audit.

### What becomes a label, and what does not?

| Item | Training/evaluation role | Example from a recording |
| --- | --- | --- |
| Action interval | Ground-truth event and time range | `Opening a closet/cabinet`, 9.5-17.0 s |
| Action name | Multi-label target and query vocabulary | `Holding a laptop` |
| Script/description | Context text and optional CLIP text target | “A person walks through a doorway...” |
| Object list | Recording-level context only | `laptop`, `doorway`, `vacuum` |
| VLM sentence | Optional post-retrieval explanation | “The laptop is visible near the doorway.” |

The last row is deliberately not used as temporal ground truth. A VLM may be
helpful for a human-readable explanation, but the official interval remains the
measurement reference.

## 3. RGB sampling and preprocessing

The current reference experiment uses 16 RGB samples per four-second window.
For a window $[s,e]$ and $F=16$ samples, timestamp $t_j$ is the center of its
equal sub-interval:

$$
t_j=s+\left(j+\frac{1}{2}\right)\frac{e-s}{F},
\qquad j=0,\ldots,F-1.
$$

For a full four-second window this is $0.125,0.375,\ldots,3.875$ seconds
after the window start. Sampling at sub-interval centers avoids repeatedly
using the exact boundary frame.

PyAV decodes the original MP4 at each timestamp. Each decoded RGB image is
passed through the Hugging Face preprocessing for `openai/clip-vit-base-patch32`:
resize/crop to the model's 224 by 224 input and convert pixels to the CLIP
normalized floating-point tensor. The image encoder is frozen and produces a
512-dimensional vector per frame:

$$
\mathbf{x}_{i,j}\in\mathbb{R}^{512},
\qquad X_i=[\mathbf{x}_{i,1},\ldots,\mathbf{x}_{i,16}]
\in\mathbb{R}^{16\times512}.
$$

The cache stores these arrays and the window metadata. It does not copy a new
video for each window. The reference cache is
`outputs/phase11/frames16/cache-v2` and records 18,994 windows from 1,300
videos, with no failed videos.

```mermaid
flowchart LR
    A[Window 2.0-6.0 s] --> B[16 center timestamps]
    B --> C[PyAV decodes RGB frames]
    C --> D[Resize and center crop to 224x224]
    D --> E[CLIP pixel normalization]
    E --> F[CLIP image encoder]
    F --> G[16 vectors of length 512]
    G --> H[Saved cache: arrays plus JSONL metadata]
```

### Formula beside the video example

For the window from 2.0 to 6.0 seconds, $e-s=4$ and $F=16$:

| Math | Video interpretation |
| --- | --- |
| $t_0=2+(0.5)(4/16)=2.125$ | The first snapshot is just after 2 seconds. |
| $t_7=2+(7.5)(4/16)=3.875$ | The eighth snapshot is just before 4 seconds. |
| $t_{15}=2+(15.5)(4/16)=5.875$ | The last snapshot is just before 6 seconds. |
| $\mathbf{x}_{i,7}\in\mathbb{R}^{512}$ | CLIP's visual description of the frame at 3.875 s. |
| $X_i\in\mathbb{R}^{16\times512}$ | The complete ordered visual record for seconds 2-6. |

If a person reaches for a laptop at 2.5 seconds and leaves at 5.5 seconds,
the middle samples capture that progression. A single still image could miss
the reach; the ordered set gives the temporal model several chances to see it.

## 4. Temporal model: exact input and output contract

The trainable model receives one cached window tensor
$X_i\in\mathbb{R}^{16\times512}$. CLIP remains frozen. The temporal module is
`TemporalWindowEncoder`:

1. A learned linear projection maps each 512-vector to 256 dimensions.
2. A learned position embedding is added to preserve frame order.
3. Two Transformer encoder layers, each with four attention heads, exchange
   information between the 16 ordered frames.
4. The contextualized frame vectors are mean-pooled.
5. A projection and LayerNorm produce a normalized 512-dimensional window
   vector.

For one window, the contract is therefore:

```text
input:  float tensor [16, 512]
output: normalized retrieval vector [512]
        action logits [157]
        boundary logits [2]
```

The current training split contains 157 distinct action IDs. The number is a
property of this prepared subset, not a universal Charades constant.

```mermaid
flowchart TD
    A[16 x 512 frozen CLIP frame matrix]
      --> B[Linear projection: 512 -> 256]
    B --> C[Add learned position embeddings]
    C --> D[Transformer encoder: 2 layers, 4 heads]
    D --> E[Mean pool 16 contextual frame vectors]
    E --> F[Projection and LayerNorm]
    F --> G[Shared representation r]
    G --> H[Retrieval head: normalized z, 512-D]
    G --> I[Action head: 157 logits]
    G --> J[Boundary head: start/end logits]
```

In equations, the shared sequence representation is:

$$
\mathbf{h}_j=\mathbf{W}_{in}\mathbf{x}_j+\mathbf{b}_{in}+\mathbf{p}_j,
\qquad
H'=\mathrm{Transformer}(\mathbf{h}_1,\ldots,\mathbf{h}_{16}),
$$

$$
\mathbf{r}=\mathrm{LayerNorm}\left(
\mathbf{W}_{out}\frac{1}{16}\sum_{j=1}^{16}\mathbf{h}'_j
+\mathbf{b}_{out}\right).
$$

The three heads are:

$$
\mathbf{z}=\mathrm{Normalize}(\mathbf{W}_r\mathbf{r}+\mathbf{b}_r),
\qquad
\hat{\mathbf{y}}=\mathbf{W}_a\mathbf{r}+\mathbf{b}_a,
\qquad
\hat{\mathbf{b}}=\sigma(\mathbf{W}_b\mathbf{r}+\mathbf{b}_b).
$$

Here **bold symbols are vectors**. $\mathbf{z}$ is used for retrieval,
$\hat{\mathbf{y}}$ is a 157-dimensional multi-label action prediction, and
$\hat{\mathbf{b}}=[\hat{u}_{start},\hat{u}_{end}]$ predicts normalized action
boundaries inside the current window.

### Formula beside a technician-style video example

| Model operation | What it means for the video |
| --- | --- |
| $\mathbf{x}_1,\ldots,\mathbf{x}_{16}$ | Sixteen CLIP descriptions of a technician's ordered views. |
| $\mathbf{h}_j=\mathbf{W}_{in}\mathbf{x}_j+\mathbf{b}_{in}+\mathbf{p}_j$ | Put every frame into the model's working space and mark its position in time. |
| Transformer attention | Let the “reaching” frame use the later “holding” frame as context. |
| $\frac{1}{16}\sum_j\mathbf{h}'_j$ | Summarize the entire four-second inspection moment. |
| $\mathbf{z}$ | One searchable memory for “reach, hold, and turn away.” |
| $\hat{\mathbf{y}}$ | Scores actions such as `Holding a laptop` or `Running`. |
| $\hat{\mathbf{b}}$ | Estimates where the action starts and ends inside this window. |

The temporal model is not a detector. It does not draw a box around a laptop.
It learns a compact representation of the ordered event and provides auxiliary
action and boundary predictions that can improve ranking and localization.

## 5. Target preparation and loss

For each window, the action target is a 157-dimensional multi-hot vector
$\mathbf{y}$. If `Holding a laptop` and `Someone is running somewhere` overlap
the window, both corresponding entries of $\mathbf{y}$ equal 1.

For the action with the greatest overlap, the boundary target is normalized
relative to the window:

$$
\mathbf{b}=
\left[
\mathrm{clip}\left(\frac{a_k-s_i}{e_i-s_i},0,1\right),
\mathrm{clip}\left(\frac{b_k-s_i}{e_i-s_i},0,1\right)
\right].
$$

Example: an annotated action from 1.0 to 3.0 seconds inside a window from 0.0
to 4.0 seconds has target $\mathbf{b}=[0.25,0.75]$. A window can have action
labels without a boundary target when no suitable single action interval is
available; the reference run has 13,567 boundary-labelled training windows.

### Formula beside a video example

Imagine the window is 0.0-4.0 seconds and the annotation says:
`Holding a laptop` from 1.0-3.0 seconds.

| Target | Calculation | Meaning |
| --- | --- | --- |
| Start | $(1.0-0.0)/(4.0-0.0)=0.25$ | The action begins one quarter of the way through the window. |
| End | $(3.0-0.0)/(4.0-0.0)=0.75$ | The action ends three quarters of the way through the window. |
| $\mathbf{b}$ | $[0.25,0.75]$ | The model should point to seconds 1-3, not claim the whole window. |

At inference time, convert normalized coordinates back to seconds:

$$
\widehat{a}=s_i+\hat{u}_{start}(e_i-s_i),
\qquad
\widehat{b}=s_i+\hat{u}_{end}(e_i-s_i).
$$

So a prediction of $[0.25,0.75]$ for a 2.0-6.0 second window becomes
3.0-5.0 seconds in the original video.

The objective combines three terms:

$$
\mathcal{L}=\mathcal{L}_{retrieval}
+\lambda_a\mathcal{L}_{action}
+\lambda_b\mathcal{L}_{boundary}.
$$

The retrieval term is symmetric contrastive loss between normalized temporal
vectors and normalized CLIP text vectors. The action term is multi-label binary
cross-entropy. The boundary term is Smooth L1 between the predicted sigmoid
coordinates and $\mathbf{b}$. The reference run uses
$\lambda_a=1.0$ and $\lambda_b=2.0$.

The model has 1,929,119 trainable parameters. The reference three-epoch CUDA
run produced:

```text
epoch 1: total 3.2478 | retrieval 2.4475 | action 0.6209 | boundary 0.0897
epoch 2: total 2.3197 | retrieval 1.7432 | action 0.4637 | boundary 0.0564
epoch 3: total 1.7963 | retrieval 1.3761 | action 0.3477 | boundary 0.0363
```

These are optimization diagnostics, not evidence that timestamps are accurate.

The three losses answer different questions:

| Loss | Question being trained | Technician interpretation |
| --- | --- | --- |
| Retrieval | Is this window related to the text description? | “Does this short clip look like the requested event?” |
| Action | Which official actions occur in the window? | “Is laptop-holding present, even if running is present too?” |
| Boundary | Where inside the window is the strongest action interval? | “Which part should the technician inspect first?” |

## 6. Online retrieval and answer flow

The user-facing path is:

```text
question
  -> CLIP text embedding \mathbf{q}
  -> optional text-only LLM maps wording to exact available action labels
  -> filter to supported labels in the selected recording
  -> search learned temporal vectors
  -> boundary-aware temporal reranking
  -> group overlapping windows into distinct events
  -> show RGB frames, timestamps, and playable evidence
  -> optional VLM explanation scoped to selected evidence
```

The LLM is a label-normalization aid, not an evidence source. It may map
“when did they carry the laptop?” to the exact supplied label `Holding a laptop`.
If no supplied action is compatible, the application returns a safe no-result;
it does not substitute a nearby cabinet or clothing action.

With normalized vectors, the retrieval score for query **q** and stored window
**z**$_i$ is cosine similarity:

$$
s_i=\cos(\mathbf{q},\mathbf{z}_i)
=\mathbf{q}^{\mathsf{T}}\mathbf{z}_i,
\qquad
\lVert\mathbf{q}\rVert_2=\lVert\mathbf{z}_i\rVert_2=1.
$$

The current index uses exact dot-product search over the 14,824 train windows.
The browser then plays the original MP4 using the selected interval. Overlap
grouping changes the displayed list, not the stored vectors or annotations.

The VLM is invoked only after retrieval. It receives timestamped RGB frames and
the selected event metadata, and can say that an event is supported, partially
visible, unclear, or not visibly confirmed. It cannot repair a bad retrieval or
turn a video-level object list into frame-level truth.

### Formula beside the user flow

| Step | Math or data operation | What the user sees |
| --- | --- | --- |
| Ask | Text question becomes **q** | “When did someone carry a laptop?” |
| Search | $s_i=\mathbf{q}^{\mathsf{T}}\mathbf{z}_i$ | Candidate windows ranked by compatibility. |
| Restrict | Keep windows with the normalized action label | Unrelated cabinet clips are rejected. |
| Refine | Apply predicted $\hat{\mathbf{b}}$ | A narrower event interval is proposed. |
| Group | Merge intervals with substantial time overlap | One event card instead of five duplicate cards. |
| Explain | VLM sees selected timestamped frames only | A readable answer with visible limitations. |

The full online flow is easier to audit when drawn as a sequence:

```mermaid
sequenceDiagram
    participant U as User
    participant API as Video API
    participant L as Label resolver
    participant IDX as Temporal index
    participant V as Original video
    participant M as Optional VLM

    U->>API: Ask a question for one recording
    API->>L: Normalize wording against available labels
    L-->>API: Exact label or unsupported
    API->>IDX: Embed question and rank compatible windows
    IDX-->>API: Scores, labels, intervals, boundary estimates
    API->>V: Decode selected interval and context frames
    V-->>API: Playable clip and timestamped RGB evidence
    API-->>U: Distinct candidate event cards
    U->>M: Optional explanation request
    M-->>U: Evidence-scoped answer and limitations
```

The LLM in this diagram can clarify that “carry a laptop” is close to the
supplied action label `Holding a laptop`. It cannot create a new timestamp. The
timestamp comes from the stored annotation, the learned boundary prediction,
and the original playable video.

## 7. Evaluation and what the numbers mean

The held-out evaluation used 4,170 test queries and the matching 16-frame
manifest. A query is correct for retrieval when a returned window shares an
official action label and overlaps its official interval.

For a predicted interval $P=[p_s,p_e]$ and official interval
$G=[g_s,g_e]$, temporal intersection-over-union is:

$$
\mathrm{IoU}(P,G)=
\frac{\max(0,\min(p_e,g_e)-\max(p_s,g_s))}
{\max(p_e,g_e)-\min(p_s,g_s)}.
$$

| Formula | Video example |
| --- | --- |
| $G=[10,14]$ | The official annotation says the action lasts from 10 to 14 s. |
| $P=[9,13]$ | The system returns a clip from 9 to 13 s. |
| Intersection $=[10,13]$ | Three seconds agree. |
| Union $=[9,14]$ | Five seconds are covered by either interval. |
| $\mathrm{IoU}=3/5=0.60$ | The result overlaps well, but it starts too early. |

Recall@k asks a different question: among the first $k$ returned cards, did at
least one have the right label and overlap the official interval? This is why a
system can have high Recall@10 while still having modest temporal IoU: it often
finds the right event somewhere in ten candidates, but does not yet place it
tightly in time.

| Metric | Result | Interpretation |
| --- | ---: | --- |
| Recall@1 | 0.6604 | Correct event appears first 66.04% of the time |
| Recall@5 | 0.9070 | Correct event appears in five candidates 90.70% of the time |
| Recall@10 | 0.9326 | Correct event appears in ten candidates 93.26% of the time |
| Mean temporal IoU | 0.2606 | Retrieved time ranges are still coarse |
| Median temporal IoU | 0.2010 | Typical overlap is lower than the recall suggests |
| Mean boundary error | 7.305 s | Absolute start/end error summary; large values reflect broad windows and fallback cases |
| Mean normalized boundary error | 0.5710 | Error relative to the annotated interval scale |
| Duplicate rate | 0.1753 | Fraction of displayed candidates that repeat an overlapping moment |
| Misses | 281 / 4,170 | No top-k candidate met the label-and-overlap criterion |

The important engineering conclusion is that the model retrieves the right
event family fairly well at top-k, but temporal localization is not yet precise.
Recall is therefore not a timestamp guarantee. The files are:

```text
outputs/phase11/frames16/cache-v2/summary.json
outputs/phase11/frames16/training/summary.json
outputs/phase11/frames16/evaluation/metrics.json
outputs/phase11/frames16/evaluation/report.md
```

## 8. Reproduce the reference run

```powershell
uv run visual-memory-lab build-charades-frames `
  --manifest outputs/charades/learned/windows/windows.jsonl `
  --output outputs/phase11/frames16 `
  --frames-per-window 16

uv run visual-memory-lab build-charades-video-cache `
  --manifest outputs/phase11/frames16/frames.jsonl `
  --output outputs/phase11/frames16/cache-v2 `
  --device auto --workers 4 --batch-size 16

uv run visual-memory-lab train-charades-video `
  --cache outputs/phase11/frames16/cache-v2 `
  --output outputs/phase11/frames16/training `
  --device auto --boundary-weight 2.0 --action-weight 1.0

uv run visual-memory-lab index-charades-video `
  --cache outputs/phase11/frames16/cache-v2 `
  --checkpoint outputs/phase11/frames16/training/temporal_multitask.pt `
  --output outputs/phase11/frames16/index --device auto --split train

uv run visual-memory-lab evaluate-charades-video `
  --index outputs/phase11/frames16/index `
  --test-manifest outputs/phase11/frames16/frames.jsonl `
  --output outputs/phase11/frames16/evaluation --device auto
```

The earlier `outputs/charades/learned` and `outputs/phase11` artifacts used
eight-frame windows and remain as historical comparisons. The 16-frame
`frames16` run is the current reference for future experiments.

## 9. Worked technician interpretation

Suppose a maintenance recording contains a person running through a doorway
with a laptop. A user asks, “When did someone carry the laptop through the
door?” The system may normalize this to `Holding a laptop` plus a nearby doorway
relation, retrieve a window, and show its annotated interval and context.

The technician can inspect the frames and playback. If the laptop is visible in
only two frames, the correct result is “partially visible,” not a confident
claim that an identified laptop was tracked. If the recording has no compatible
action, the correct result is “no matching annotated event was found.” This
separation between dataset label, learned retrieval, visual review, and final
wording is the main reliability boundary of the application.

## 10. Current boundaries

This reference system does not yet provide persistent object identity, depth,
audio understanding, true frame-level segmentation, or production-grade event
boundaries. Those are separate extensions. The current contribution is a
reproducible path from official video annotations to learned temporal retrieval,
with explicit tensor contracts, evidence provenance, and held-out metrics.

## 11. Phase 12: frame-level temporal evidence

The earlier three-head model predicted one start and one end value from the
pooled four-second window. Phase 12 adds a small evidence head to the same
temporal encoder. CLIP remains frozen; only the temporal model is trained.

For each sampled frame embedding $\mathbf{x}_{i,j}$, the encoder produces a
hidden vector $\mathbf{h}_{i,j}$. A linear head predicts relevance, start
proximity, and end proximity:

$$
\mathbf{z}_{i,j}=W_f\mathbf{h}_{i,j}+\mathbf{b}_f\in\mathbb{R}^3.
$$

The labels come from the official action interval. A frame inside the interval
gets relevance label $1$; the closest sampled frame to each boundary gets the
corresponding boundary label $1$. The extra loss is:

$$
\mathcal{L}_{frame}=\mathrm{BCEWithLogits}(\mathbf{z},\mathbf{y}).
$$

The Phase 12 objective is:

$$
\mathcal{L}=\mathcal{L}_{retrieval}+\lambda_a\mathcal{L}_{action}
+\lambda_b(\mathcal{L}_{boundary}+\mathcal{L}_{frame}).
$$

At indexing time, frame probabilities become timestamps using weighted
averages. If $p^s_j$ and $p^e_j$ are the start and end probabilities at
timestamp $t_j$:

$$
\hat{s}=\frac{\sum_jt_jp^s_j}{\sum_jp^s_j},\qquad
\hat{e}=\frac{\sum_jt_jp^e_j}{\sum_jp^e_j}.
$$

The estimates are clamped to the retrieved window. If they are invalid, the
pooled boundary estimate is retained. This is interval refinement, not object
tracking and not proof that every frame visibly proves the action.

The UI keeps three intervals separate:

| Field | Meaning |
| --- | --- |
| Matched action interval | The refined interval used for review |
| Dataset annotation | The official Charades interval used for supervision |
| Context shown | A padded playable interval, currently two seconds on either side |

For example, “holding a laptop” may produce a refined interval of $0.8$--$6.9$
seconds, an official annotation of $0.6$--$7.8$ seconds, and a context clip of
$0.0$--$9.8$ seconds. The user can distinguish the model’s localization from
the annotation and from extra playback context. The index also stores sampled
timestamps and refinement confidence for later frame-strip review.

Phase 12 does not add audio, depth, persistent object identity, or VLM-generated
labels. It improves temporal evidence while preserving the frozen-CLIP baseline
for comparison.

### Phase 12 status and metrics

The Phase 12 implementation is present on the `phase/12-trustworthy-temporal-evidence`
branch, but its experiment artifacts have not yet been generated. In practical
terms, the new frame-refinement head has been coded and tested for shape,
training, checkpoint loading, API propagation, and UI compilation. We have not
yet run the full Phase 12 training job, built its refined index, or evaluated it
on the held-out queries.

Therefore, the metrics reported earlier in this document and in the README are
Phase 11 baseline metrics. They must not be described as Phase 12 results. The
Phase 12 comparison will be reported only after these files exist:

```text
outputs/phase12/frames16/training/summary.json
outputs/phase12/frames16/training/history.jsonl
outputs/phase12/frames16/index/summary.json
outputs/phase12/frames16/evaluation/metrics.json
```

The comparison will use the same held-out test queries and report Recall@1,
Recall@5, Recall@10, temporal IoU, start/end boundary error, duplicate-window
rate, and unsupported-query behavior. Until then, the correct claim is:

> Phase 12 adds a tested frame-level temporal refinement method; its numerical
> improvement over the Phase 11 baseline is not yet measured.
