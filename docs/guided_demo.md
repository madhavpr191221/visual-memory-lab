# Guided Office Inspection Demo

## The 90-second story

Visual Memory Lab is an evidence-first assistant for a technician revisiting an office. The guided case asks:

> Is this the workstation beside the window?

The page presents real office-memory frames, compares two retrieved views, explains the safe conclusion, and recommends a manual check. It is a deterministic presentation of the current artifacts, not a claim that the public recordings contain calendar-dated inspections.

Open `/app/demo` locally, or choose **Watch the guided case** on the landing page.

## What the viewer sees

1. **The question** — a practical location question a facilities technician might ask.
2. **Evidence retrieval** — CLIP searches the prepared office memory index and returns diverse views rather than a wall of adjacent frames.
3. **Side-by-side views** — two real office images are shown with their zone, sequence, frame, and similarity score.
4. **Safe conclusion** — the system describes the shared visual evidence without claiming persistent object identity.
5. **Manual check** — the technician receives an actionable next step: verify monitor power, cable routing, and desk stability on site.

## What each subsystem contributes

```text
CLIP                    finds visually relevant office views
zone metadata           gives the views a human-readable area label
source metadata         preserves sequence, frame, and provenance
claim boundary          prevents similarity from becoming identity
technician report       turns evidence into a practical inspection step
```

The Research workspace then exposes the measurements behind the presentation: pose-grounded retrieval, coverage, detector and mask outputs, RGB-D evidence, association candidates, and failure cases.

## Metrics to discuss in an interview

The Research workspace reports:

- held-out query count and strict/relaxed coverage;
- hit@1, hit@5, and hit@10;
- median and 90th-percentile translation and rotation error;
- technician-question evidence recall;
- object prediction and mask coverage;
- RGB-D point coverage;
- cross-visit candidate counts and uncertainty.

Coverage is reported separately from hit rate. If the memory contains no sufficiently nearby reference view, retrieval cannot succeed even if the embedding model is good.

## Retrieval evaluation ownership

For a normalized query vector `q` and stored image vector `x_i`, exact retrieval ranks cosine similarity:

```math
s(q,x_i) = \frac{q \cdot x_i}{\lVert q \rVert_2\lVert x_i \rVert_2}
```

The pose-grounded evaluation then asks whether a retrieved frame is physically close enough to the query. A strict result uses the project’s strict translation and rotation thresholds; a relaxed result uses wider thresholds. `hit@k` asks whether at least one of the first `k` results meets the threshold.

The important failure decomposition is:

```text
no nearby reference exists → coverage failure
nearby reference exists but is not retrieved → retrieval failure
wrong-looking place retrieved → perceptual aliasing
right place but wrong visit → temporal ambiguity
```

This is why a similarity score alone is not enough for a physical-world memory system.

## Cross-visit association ownership

For candidate detections `a` and `b` from different logical visits, the current ranking combines several signals:

```math
S(a,b) = w_a A(a,b) + w_s S_{\mathrm{shape}}(a,b) + w_g G(a,b) + w_e E(a,b)
```

The terms represent appearance, visible shape, approximate shared-room geometry, and evidence quality. The output is a candidate label such as `likely_same`, `possible_match`, or `uncertain`.

This is deliberately not a tracker. A high score means “inspect this pair first,” not “the system proved this is the same chair.” A position difference may support a movement hypothesis, but viewpoint, occlusion, and incomplete reconstruction can produce the same pattern.

## Failure examples

| Failure | Likely cause | Evidence | Safe interpretation | Next experiment |
| --- | --- | --- | --- | --- |
| Similar workstation from wrong area | perceptual aliasing | visual match but large pose error | appearance alone is insufficient | add pose or zone filtering |
| Correct place, wrong visit | temporal ambiguity | place metrics pass, visit order is unavailable | place retrieval worked; visit retrieval did not | add verified repeated visits |
| Object not detected | occlusion or detector miss | neighboring frames show the object | absence is unproven | inspect more frames and coverage |
| Candidate positions disagree | viewpoint or reconstruction noise | geometry differs but RGB evidence is weak | possible mismatch or move | compare more views and quality scores |
| VLM disagrees with geometry | incomplete evidence or model error | cached judgment conflicts with measured artifact | pseudo-audit is not ground truth | retain both evidence and uncertainty |

The interview answer should be: failure → likely cause → evidence → safe interpretation → next experiment. “The model failed” is not a useful diagnosis because it hides which assumption was violated.

## What the demo does not claim

- The public sequences provide logical order, not calendar dates.
- A visually similar workstation is not a persistent object identity.
- A missing box or mask is not proof that an object disappeared.
- Recorded point clouds show visible geometry, not hidden surfaces.
- A VLM report supports inspection; it does not replace a technician or create ground truth.

## Video-memory UI walkthrough

The primary application is now the prepared-video memory workflow. Open
`/app/video`, choose a recording, and use **Find an event** when you want the
system to locate a moment. The page shows the recording summary but hides the
action list before retrieval, so the question tests the memory system rather
than revealing the answer.

For example:

```text
choose recording 08F85
    -> read its summary
    -> ask “When did the person hold some medicine?”
    -> review candidate event cards
    -> inspect the action interval and separate context interval
    -> inspect timestamped RGB frames and playable evidence
    -> ask a follow-up about the selected event
    -> save the finding with a review status
```

The **Review the timeline** tab is different: it exposes the official Charades
action intervals directly for audit. Those action labels come from the dataset
manifest. The optional VLM is called only after an event is selected and sees
the supplied evidence frames, not the entire archive.

The current UI uses prepared Charades recordings. Arbitrary uploaded-video
inference is a future extension described in the system architecture document;
it is not silently performed by this demo.
