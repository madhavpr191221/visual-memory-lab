# Phase 5: Cross-Traversal Revisit Memory

## Objective

Phase 5 asks a narrower question than change detection:

> Given a camera observation from one traversal, can the memory recover a
> geometrically comparable observation from a designated reference traversal?

Before a system can say what changed since an earlier inspection, it must first
find the correct earlier view. Comparing the wrong desk, the wrong side of the
room, or a view that never saw the target area will produce a meaningless
change report.

Phase 3 searched all 6,000 stored Office frames at once. Phase 5 selects one
reference traversal and asks the retrieval system to find the best matching
view inside it. This exposes two independent questions:

1. Did the selected traversal record a comparable view at all?
2. If it did, did CLIP retrieve that view?

The first is a route-coverage problem. The second is a visual-retrieval problem.

## Technician interpretation

Imagine that a technician is standing in front of a workstation during the
current inspection. They select one earlier inspection recording and ask:

> Show me the view from that recording that is most comparable to what I am
> looking at now.

The earlier technician may have walked through the office along a different
route. There are two possible outcomes:

- The earlier recording contains a view of the workstation from a sufficiently
  similar position and direction. The retrieval system should recover it.
- The earlier recording never covered that viewpoint. The system should report
  that the required evidence was unavailable rather than treating retrieval as
  the only failure.

7-Scenes does not provide verified dates or maintenance events. Its sequences
are treated as separate camera traversals, not as calendar-ordered visits. The
experiment measures revisit alignment without claiming temporal change.

## The simple version

Imagine that six videos were recorded while walking around the same office on
six separate rounds. Today, someone is standing in front of a desk. They choose
one old recording and ask:

> Find the moment in this recording when the camera was looking at this same
> desk.

Before Phase 5, the system searched every stored office image together. It
might retrieve the correct desk, but the result could come from any recording.
Phase 5 lets the user select one recording and searches only inside it:

```text
current office image
        |
        v
choose one reference recording
        |
        v
search only that recording
        |
        v
return its most comparable views
```

Camera pose acts as the answer key. CLIP can say that two images look similar,
but pose tells us whether they were captured near the same physical location
and while facing approximately the same direction.

The selected recording may never have captured the requested location. That is
a coverage failure:

```text
Did the selected recording contain a comparable view?
        |
        +-- no  -> the evidence was never recorded
        |
        +-- yes -> test whether CLIP retrieves it
```

This prevents us from blaming the retrieval model for failing to return an
image that does not exist. In a real inspection system, the result also says
something about the recording route: a better model cannot compensate for a
technician who never visited or photographed the relevant area.

Phase 5 does not yet report that a mug disappeared or a chair moved. It builds
the prerequisite steps:

```text
find the same place in another recording
        |
        v
make sure the two views are comparable
        |
        v
future phase: compare their contents and report actual change
```

## What the 6,000/4,000 train-test split means

The word **train** is slightly misleading in this project. Phase 5 does not
train or fine-tune CLIP on the 6,000 Office images. CLIP remains frozen.

The split is better understood as:

```text
6,000 training-side frames = stored visual memory
4,000 test-side frames     = new observations used to question that memory
```

The Office scene has ten sequences with 1,000 frames each. Its six official
training sequences form the stored memory, while its four official test
sequences provide held-out queries.

Keeping complete sequences on different sides is important. If individual
frames were randomly mixed, the memory could contain frames immediately before
or after a query frame from the same camera recording. Those neighbouring
frames would be almost duplicates. Retrieval would look excellent without
showing that the system can recognize a place during a different traversal.

The sequence-level split therefore creates a fairer test:

```text
build memory from six camera traversals
        |
        v
present images from four different traversals
        |
        v
measure whether the stored memory recognizes the same physical areas
```

For Phase 3, this tests whether a held-out image can retrieve the correct place
from the complete 6,000-frame memory. For Phase 5, each held-out image is
searched separately against each of the six stored traversals. The same split
therefore supports both the broad place-retrieval experiment and the more
specific cross-traversal experiment.

This is evaluation, not supervised learning. The test images do not update the
encoder, the embeddings, the zone definitions, or any model parameters.

## Dataset protocol

The experiment reuses the official Phase 3 split:

- query traversals: `seq-02`, `seq-06`, `seq-07`, and `seq-09`;
- reference traversals: `seq-01`, `seq-03`, `seq-04`, `seq-05`, `seq-08`, and
  `seq-10`;
- 1,000 RGB observations per traversal;
- 4,000 query observations and 6,000 stored observations.

Every query traversal is paired with every reference traversal. This produces

$$
4 \times 6 = 24
$$

source-target traversal pairs and

$$
4{,}000 \times 6 = 24{,}000
$$

query-target evaluations.

The experiment evaluates every pair instead of inventing an order such as
`seq-01` before `seq-03`. Sequence numbers are identifiers, not timestamps.

## One query-target evaluation

For query observation $q$ and designated target traversal $v$, Phase 5:

1. keeps only the 1,000 candidate observations belonging to $v$;
2. calculates which candidates are geometrically relevant using camera pose;
3. ranks the candidates by exact cosine similarity between normalized CLIP
   embeddings;
4. checks whether the top 1, 5, or 10 results contain a pose-relevant frame;
5. repeats the check with a fixed-seed random ranking from the same traversal.

Candidate restriction is important. The task is not to guess the traversal ID.
The user has already selected the recording they want to inspect. The task is
to find the correct view inside it.

No images are re-encoded. Phase 5 loads the frozen Phase 3 ViT-B/32 indexes and
uses their existing 512-dimensional normalized embeddings.

## Pose-grounded relevance and coverage

Let query pose $P_q$ and candidate pose $P_i$ contain a camera position and
orientation. A candidate is relevant when both its translation error and
rotation error fall within a threshold.

The thresholds are unchanged from Phase 3:

| Criterion | Translation | Rotation |
|---|---:|---:|
| Strict | at most 0.25 m | at most 30 degrees |
| Relaxed | at most 0.50 m | at most 30 degrees |

A query-target evaluation is **eligible** when at least one frame in the target
traversal satisfies the chosen threshold. Coverage is

$$
\text{Coverage}
=
\frac{\text{eligible query-target evaluations}}
{\text{all query-target evaluations}}.
$$

Coverage is an oracle ceiling derived from pose. If no relevant candidate
exists, neither CLIP nor a more advanced ranker can retrieve one from that
traversal.

## Metrics

For eligible query-target evaluations, Hit@$k$ is

$$
\text{Covered Hit@}k
=
\frac{\text{eligible evaluations with a relevant result in the top }k}
{\text{eligible evaluations}}.
$$

The experiment also reports an all-query-target rate:

$$
\text{All-query Hit@}k
=
\frac{\text{all evaluations with a relevant result in the top }k}
{\text{all query-target evaluations}}.
$$

The covered rate isolates ranking quality. The all-query rate combines route
coverage and ranking quality. Reporting both prevents missing observations from
being mistaken for weak retrieval.

Results are aggregated in two ways:

- micro results pool all 24,000 query-target evaluations;
- macro-pair results calculate every traversal pair separately and then average
  the 24 pair-level values.

The evaluator also records top-1 translation and rotation error and a random
baseline sampled from the same target traversal with seed 42.

## Measured results

The acceptance run used all 10,000 Office frames and completed 24,000
query-target evaluations across all 24 traversal pairs.

| Criterion | Coverage | CLIP Hit@1 | CLIP Hit@5 | CLIP Hit@10 | Random Hit@1 | Random Hit@5 | Random Hit@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strict | 17.1% | 36.8% | 52.1% | 60.6% | 3.6% | 15.1% | 27.1% |
| Relaxed | 41.6% | 53.7% | 67.2% | 73.1% | 7.0% | 27.9% | 44.4% |

These Hit@$k$ values use only eligible query-target evaluations. The strict
criterion contains 4,097 eligible evaluations; the relaxed criterion contains
9,995.

The all-query-target rates are much lower because most designated traversals do
not contain a qualifying view:

| Criterion | Hit@1 | Hit@5 | Hit@10 |
|---|---:|---:|---:|
| Strict | 6.3% | 8.9% | 10.3% |
| Relaxed | 22.4% | 28.0% | 30.5% |

The median top-1 pose error across all query-target evaluations is 0.804 m and
25.5 degrees. The 90th-percentile errors are 1.564 m and 85.2 degrees.

## What the result means

CLIP clearly beats random selection when comparable evidence exists. Under the
strict criterion, its Hit@1 is roughly ten times the random Hit@1. Increasing
$k$ also recovers additional valid views.

The larger limitation is coverage. Only 17.1% of designated traversal pairs
contain a strict pose match for a query. Coverage also varies sharply by route:
`seq-09 -> seq-08` has 52.0% strict coverage, while `seq-09 -> seq-10` has none.
This is not a CLIP failure. The two recordings simply do not provide the same
strict viewpoint coverage.

That gives the project a useful operational rule:

> Check whether comparable evidence exists before interpreting a retrieval
> failure or attempting scene comparison.

## Artifacts

The command writes two local, ignored artifacts:

- `metrics.json` contains protocol metadata, micro metrics, macro-pair metrics,
  pose-error summaries, and all 24 pair summaries;
- `per_query_target.jsonl` contains every query-target decision and its ten
  retrieved observations, scores, pose errors, relevance counts, and hit flags.

The detailed file is intentionally local because it is about 55 MB and points
to licensed dataset observations. The repository records the protocol and
aggregate results without redistributing images, embeddings, or dataset-derived
bulk output.

## Reproduction

After producing the Phase 3 train and test indexes, run:

```powershell
uv run visual-memory-lab evaluate-traversal-memory `
  --memory-index outputs/phase3/train-index `
  --query-index outputs/phase3/test-index `
  --output outputs/phase5/traversal-evaluation `
  --seed 42
```

The output directory must be absent or empty. Source fingerprints are verified
before evaluation so the recorded embeddings cannot silently drift away from
their manifests.

## Limits and next phase

Phase 5 does not determine which traversal happened first. It does not report
that an object appeared, disappeared, moved, opened, closed, or changed
condition. Artificial dates would add metadata but would not create real scene
change.

The next phase needs repeated observations with known ordering and annotated
state transitions. With that ground truth, the system can evaluate three stages
separately:

1. retrieve the correct place;
2. retrieve the correct previous visit;
3. compare the two aligned memories and report the actual change.

7-Scenes remains useful for the first alignment problem. A controlled simulator,
game environment, or deliberately captured public-safe scene is required for
the final state-change benchmark.

## References

- Microsoft Research, [RGB-D Dataset 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).
- Jamie Shotton et al., [Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images](https://www.microsoft.com/en-us/research/publication/scene-coordinate-regression-forests-for-camera-relocalization-in-rgb-d-images-2/), CVPR 2013.
- Alec Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html), ICML 2021.
