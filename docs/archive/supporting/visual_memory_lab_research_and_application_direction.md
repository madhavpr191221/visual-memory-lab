# Visual Memory Lab: Research and Application Direction

## Purpose of this document

Visual Memory Lab is not intended to become a collection of unrelated computer
vision demonstrations. It is one research and application programme built
around a specific problem:

> How can a system remember physical places across repeated observations,
> recover the correct prior evidence, and distinguish real changes in the world
> from changes caused by viewpoint, visibility, lighting, and incomplete
> recording coverage?

This problem is useful in the real world and technically non-trivial. It brings
together visual representation learning, multimodal retrieval, camera geometry,
temporal memory, object identity, uncertainty, and human inspection.

The project should remain grounded in an actual user. A technician, inspector,
engineer, analyst, or ordinary person does not primarily care which embedding
model produced a result. They care whether the system can recover trustworthy
evidence, explain what it could and could not observe, and avoid inventing a
change when the available images do not support one.

## The real-world problem

Many organisations already collect large numbers of photographs and videos:

- maintenance technicians record equipment during inspection rounds;
- construction teams photograph progress and safety conditions;
- utilities record poles, pipes, meters, cabinets, and substations;
- property inspectors document rooms before and after occupancy;
- insurance assessors collect evidence before and after an incident;
- warehouse and retail teams record storage areas, displays, and access routes;
- laboratories and hospitals inspect instruments, rooms, and controlled areas.

The recordings usually become an unstructured archive. Finding useful evidence
later depends on filenames, folders, timestamps, or a person remembering where
to look. Even when image search is available, visual similarity alone is not
enough. Two workstations may look similar while occupying different locations,
and two images of the same workstation may look very different because the
camera moved.

A useful visual memory should support questions such as:

- What did this workstation look like during the previous inspection?
- Which earlier recording gives the clearest comparable view of this panel?
- When was this obstruction first visible?
- Was this exit clear during the last three inspection rounds?
- Did this object move, or was it outside the later camera view?
- Is this the same damaged component reported previously?
- Which required locations were not adequately recorded today?
- What evidence supports the claim that a condition changed?

These are not ordinary image-search questions. They require memory about place,
visit, viewpoint, visibility, object identity, and world state.

## A concrete inspection example

Consider a technician who walks through a facility once per week while recording
video. A month later, a cardboard box is found obstructing an electrical panel.
The supervisor asks:

> When did the obstruction first appear, and which recordings prove it?

The system cannot answer responsibly by retrieving images containing any box or
any electrical panel. It needs to:

1. identify the correct physical panel;
2. find recordings from earlier inspection rounds;
3. determine whether each round actually captured a comparable view;
4. align or select observations that show the same area;
5. determine whether the box was visible, absent, or unobservable;
6. report the earliest supported appearance;
7. show the original evidence and any uncertainty.

If one inspection never recorded the panel, the system should say that the
state is unknown for that visit. It must not silently interpret missing evidence
as evidence that the box was absent.

This distinction has practical value. It separates three statements that are
often confused:

```text
The object was not present.
The object was not visible.
The location was not recorded.
```

Only the first is a world-state conclusion. The other two describe limitations
of the observation process.

## Why an image archive is not yet a memory

An image is an observation of the world, not the world state itself. A useful
model is

$$
I_{\ell,v,k}
=
g\left(
S_{\ell,v},
P_{\ell,v,k},
L_{\ell,v,k},
O_{\ell,v,k},
C_{\ell,v,k},
\epsilon_{\ell,v,k}
\right),
$$

where:

- $I_{\ell,v,k}$ is frame $k$ captured at physical location $\ell$ during
  visit or traversal $v$;
- $S_{\ell,v}$ is the physical state of the location during that visit;
- $P_{\ell,v,k}$ is the camera pose;
- $L_{\ell,v,k}$ represents lighting and exposure;
- $O_{\ell,v,k}$ represents occlusion and visibility;
- $C_{\ell,v,k}$ represents camera properties such as intrinsics and
  distortion;
- $\epsilon_{\ell,v,k}$ represents sensor noise, blur, and other unmodelled
  effects.

Two images can differ even when the physical state is unchanged:

$$
S_{\ell,v_1}=S_{\ell,v_2},
\qquad
I_{\ell,v_1,k_1}\neq I_{\ell,v_2,k_2},
$$

because the camera pose, lighting, or visibility changed.

The opposite can also happen. Two images can be visually similar even though
one important condition changed:

$$
P_{\ell,v_1,k_1}\approx P_{\ell,v_2,k_2},
\qquad
S_{\ell,v_1}\neq S_{\ell,v_2}.
$$

For example, a small safety pin may be removed while the rest of a large machine
remains visually identical. The system must detect the task-relevant change
without treating every pixel difference as a world change.

## The central research problem

The long-term task is to estimate world-state change while accounting for the
observation process:

$$
\Delta_{v_1\rightarrow v_2}^{(\ell)}
=
\mathcal{C}\left(
S_{\ell,v_1},
S_{\ell,v_2}
\right),
$$

where $\mathcal{C}$ is a structured comparison operation. It should return
claims such as:

```text
added:
  cardboard box

removed:
  red warning tag

moved:
  rolling chair: desk front -> aisle

state changed:
  panel door: closed -> open

unchanged:
  electrical panel
  wall sign

unknown:
  fire extinguisher: not visible in the later observation
```

The change operator should not be treated as literal subtraction between two
arbitrary embedding vectors. The underlying state is structured: objects have
identities, attributes, spatial relationships, visibility, and uncertainty.

## The research decomposition

An end-to-end answer can fail for several different reasons. Visual Memory Lab
should preserve a decomposition in which each stage can be measured separately.

### 1. Recording coverage

Before retrieval, ask whether the selected traversal contains any observation
capable of answering the question.

For query $q$, reference traversal $v$, and comparability threshold $\tau$,
define

$$
\operatorname{Covered}(q,v;\tau)
=
\mathbb{1}\left[
\exists i\in v:
d_P(P_q,P_i)\leq\tau
\right],
$$

where $d_P$ may combine translation, rotation, field-of-view overlap, and later
3D visibility.

Coverage is an oracle ceiling. A retrieval model cannot return evidence that
was never captured.

### 2. Place retrieval

Given query representation $z_q$ and stored representations $z_i$, the basic
retrieval stage ranks memories by

$$
i^*
=
\arg\max_i
\operatorname{sim}(z_q,z_i).
$$

The highest visual similarity is not necessarily the correct place. Repeated
workstations, doors, corridors, panels, and shelves create perceptual aliasing.
Pose and spatial ground truth are therefore used to distinguish resemblance
from place correctness.

### 3. Traversal-conditioned retrieval

If a user selects one reference recording $v$, search must be restricted to
that recording:

$$
i_v^*
=
\arg\max_{i:\,\operatorname{visit}(i)=v}
\operatorname{sim}(z_q,z_i).
$$

This is the capability implemented in Phase 5. It asks whether a comparable
view can be recovered inside a designated traversal and reports separately when
that traversal never covered the query pose.

### 4. Observation comparability

Two frames should not be compared merely because they received similar
embeddings. A comparability function can incorporate pose, overlap, visibility,
and scene content:

$$
a_{ij}
=
f_{\text{align}}\left(
P_i,P_j,D_i,D_j,F_i,F_j
\right),
$$

where $D$ represents depth and $F$ represents local or semantic features.

The system should compare state only when $a_{ij}$ is sufficiently high. If no
pair is reliable, it should abstain.

### 5. Object correspondence

Detection alone produces categories, not persistent physical identities.
Tracking usually maintains identity across adjacent frames in one video. A
last-seen query across separate visits requires cross-visit association:

$$
p(o_i\equiv o_j\mid
\text{appearance, geometry, context, time}).
$$

The system must distinguish:

- the same object viewed from another angle;
- two different objects of the same category;
- one object that moved;
- one object that disappeared;
- an object that is merely occluded.

### 6. State comparison

Given a correct place, correct visit, comparable observations, and associated
objects, the system can estimate change:

$$
\widehat{\Delta}_{v_1\rightarrow v_2}^{(\ell)}
=
f_{\text{change}}\left(
I_{\ell,v_1},
I_{\ell,v_2},
A,
V
\right),
$$

where $A$ contains alignment or correspondence information and $V$ represents
visibility.

### 7. Temporal selection

When real visit order is available, the system must retrieve the immediately
previous relevant visit rather than an arbitrary old observation:

$$
v^-
=
\max\left\{
v_i : t(v_i)<t(v_q),\
\operatorname{place}(v_i)=\ell
\right\}.
$$

This cannot be evaluated honestly using artificial dates attached to unordered
recordings. A temporal benchmark requires verified visit order.

### 8. Evidence-grounded answer and abstention

The final answer should cite the observations that support each claim. It
should also expose uncertainty and abstain when the evidence is insufficient.

For confidence threshold $\gamma$, a simple decision rule is

$$
\widehat{y}
=
\begin{cases}
f(x), & p_{\max}(x)\geq\gamma,\\
\text{insufficient evidence}, & p_{\max}(x)<\gamma.
\end{cases}
$$

In this application, a well-calibrated refusal can be more valuable than a
confident false change report.

## The application system

The long-term product flow is:

```text
inspection photos or video
        |
        v
frame selection and quality checks
        |
        v
visit, time, camera, and provenance metadata
        |
        v
visual and spatial memory index
        |
        v
place and reference-visit retrieval
        |
        v
coverage and comparable-view checks
        |
        v
object association and state comparison
        |
        v
evidence, uncertainty, and missing-coverage report
        |
        v
human review
```

The interface should remain evidence-first. A generated summary is useful only
after the user can inspect the source images, retrieval scores, visit metadata,
and stated limitations.

### Ingestion

The application should accept an inspection round as a coherent unit rather
than as unrelated images. A visit can contain:

- source video or photographs;
- capture timestamps;
- device and camera metadata;
- optional route, GPS, or pose estimates;
- inspection type and site identifier;
- operator-provided notes;
- immutable provenance linking every derived memory to its source.

### Memory creation

The system selects useful frames, removes exact or near duplicates when
appropriate, computes representations, records temporal order, and stores
spatial and object-level metadata. A memory record may eventually be represented
as

$$
m_i=
\left(
I_i,z_i,v_i,t_i,P_i,\mathcal{O}_i,\mathcal{V}_i,\pi_i
\right),
$$

where $\mathcal{O}_i$ contains observed objects, $\mathcal{V}_i$ contains
visibility information, and $\pi_i$ records provenance.

### Query and evidence selection

Queries may arrive as text, an image, a current video frame, a location, an
object instance, or a structured inspection question. Retrieval should return
a small evidence set rather than immediately generating an answer.

### Human review

The application should clearly distinguish:

- retrieved evidence;
- mechanically measured facts;
- model-generated interpretations;
- unsupported or unknown conditions.

This distinction is essential for maintenance, safety, insurance, and other
settings where a wrong statement can be costly.

## Real-world application areas

### Facility maintenance and safety

Possible questions:

- Was the electrical panel accessible during the previous round?
- When did the leak stain first become visible?
- Was the emergency exit obstructed?
- Which machines were not photographed clearly?

Potential value:

- less time manually reviewing inspection archives;
- earlier recognition of recurring conditions;
- evidence for maintenance prioritisation;
- measurable route and recording completeness.

### Construction progress and compliance

Possible questions:

- What changed in this work area since the previous site walk?
- Was the safety barrier present before work began?
- Which rooms have no comparable progress images?
- Did installed equipment move after inspection?

The system should distinguish genuine construction progress from viewpoint
change, temporary occlusion, and incomplete site coverage.

### Utilities and field service

Possible questions:

- Is this the same cabinet serviced during the last visit?
- Which component was replaced?
- Was corrosion already visible?
- Did the technician capture the required meter and connector views?

Here, stable asset identity and offline or edge-capable operation may be more
important than conversational sophistication.

### Property and insurance evidence

Possible questions:

- Was this damage visible in the earlier inspection?
- Which images provide the closest comparable viewpoint?
- Is a missing item unsupported because the room was not recorded?
- What evidence changed between check-in and check-out?

The system must preserve provenance and avoid converting absence of evidence
into evidence of absence.

### Warehouse and retail operations

Possible questions:

- When did this aisle become obstructed?
- Did the display layout change?
- Was the required safety equipment visible?
- Which storage zones were skipped during the walk-through?

The commercial value comes from faster review and better coverage, not from
claiming that every operational decision can be automated.

## Research track and application track

The repository can support two connected tracks without becoming two unrelated
projects.

### Research track

The research track asks:

- Can the same physical area be recovered across traversals?
- Does a selected traversal contain comparable evidence?
- Can viewpoint change be separated from physical state change?
- Can object identity persist across separate visits?
- Which component caused an end-to-end failure?
- When should the system abstain?

Each experiment should have a defined dataset contract, ground truth, baseline,
metric, failure taxonomy, and exit criterion.

### Application track

The application track asks:

- Can a user ingest a complete inspection?
- Can they select a location, object, or prior visit?
- Can they understand why evidence was selected?
- Are missing coverage and uncertainty visible?
- Can results be traced to original observations?
- Is latency acceptable for the intended workflow?

The persistent `demo` branch is the application showcase. Research phases are
developed and validated independently. A capability is added to the demo only
after it produces stable artifacts and a useful human workflow.

## The role of model training

Training a neural network is a method, not the purpose of Visual Memory Lab.
The project should not add a detector, segmenter, tracker, or fine-tuned encoder
only to collect model names.

The preferred sequence is:

```text
establish a simple baseline
        |
        v
measure failures on a defined task
        |
        v
identify the real bottleneck
        |
        v
train a component designed for that bottleneck
        |
        v
compare against the frozen baseline and perform ablations
```

Examples of justified training include:

- pose-supervised metric learning when similar workstations cause place
  confusion;
- cross-view object re-identification when category detection cannot preserve
  physical identity;
- visibility-aware change classification when viewpoint changes create false
  alarms;
- learned local matching when global embeddings retrieve the right area but
  fail to align observations;
- temporal representation learning when isolated frames miss gradual state
  transitions.

A trained component should answer a documented failure. The research claim is
then not merely that a model was fine-tuned, but that a specific intervention
improved a specific failure mode under a controlled protocol.

## The role of 3D computer vision

3D geometry can connect images to physical space. With camera intrinsics $K$,
depth $d$, pixel coordinates $(u,v)$, and camera-to-world pose $T_{wc}$, a pixel
can be back-projected into the camera frame:

$$
\mathbf{x}_c
=
dK^{-1}
\begin{bmatrix}
u\\v\\1
\end{bmatrix},
$$

then transformed into world coordinates:

$$
\widetilde{\mathbf{x}}_w
=
T_{wc}
\begin{bmatrix}
\mathbf{x}_c\\1
\end{bmatrix}.
$$

This enables memories from different viewpoints to refer to a shared spatial
frame. It can support:

- 3D overlap and visibility checks;
- spatially anchored objects;
- reasoning about whether an object moved;
- separating camera motion from world motion;
- more reliable comparison of different viewpoints.

Ground-truth depth and pose should first be treated as an oracle. After the
spatial-memory logic is validated, estimated depth and pose can replace the
oracle and the resulting degradation can be measured. This keeps geometry
failure separate from memory failure.

## Dataset strategy

No single dataset needs to support every research question.

| Dataset role | Required properties | Supported claims |
|---|---|---|
| 7-Scenes Office | Real RGB-D traversals and supplied camera poses | Real-image place retrieval, route coverage, and cross-traversal alignment |
| ETH Office | Real RGB images, coloured point clouds, and recorded transforms | Object localization, visible geometry, and cautious cross-visit association |
| Controlled repeated-visit scene | Verified visit order, repeated viewpoints, annotated changes | Previous-visit retrieval and real state-change evaluation |
| Operational pilot data | Real workflow, capture constraints, and user questions | Usability, latency, coverage, and domain value |

7-Scenes should continue to be used for the questions it can answer. Its
sequence identifiers should not be treated as verified chronology, and its
images do not provide controlled object-change labels.

A controlled change dataset does not need to contain private household images.
It can use a neutral tabletop, mock workstation, or lab corner with public-safe
objects and scripted changes.

## Evaluation framework

### Retrieval and coverage

- Place Hit@$k$: does top $k$ contain the correct physical area?
- Traversal-conditioned Hit@$k$: does top $k$ contain a comparable observation
  from the selected traversal?
- Coverage: did the selected traversal contain any valid evidence?
- Translation and rotation error: how far is the retrieved camera pose from the
  target pose?
- Random and simple-feature baselines: how much value comes from the visual
  representation?

### Temporal retrieval

When real visit order exists, define Previous-Visit Hit@$k$:

$$
\operatorname{PVHit@}k
=
\frac{1}{N}
\sum_{q=1}^{N}
\mathbb{1}\left[
\text{top-}k(q)\cap\mathcal{R}_{v^-(q)}\neq\varnothing
\right],
$$

where $\mathcal{R}_{v^-(q)}$ is the set of relevant observations from the
immediately previous visit.

### State-change quality

For reported change claims:

$$
\operatorname{Precision}
=
\frac{\text{correct reported changes}}
{\text{all reported changes}},
$$

$$
\operatorname{Recall}
=
\frac{\text{correct reported changes}}
{\text{all true changes}},
$$

and

$$
F_1
=
2\frac{\operatorname{Precision}\operatorname{Recall}}
{\operatorname{Precision}+\operatorname{Recall}}.
$$

The evaluation should distinguish change categories such as added, removed,
moved, attribute changed, unchanged, occluded, and unknown.

### Oracle versus end-to-end evaluation

Report state-comparison quality under two conditions:

$$
F_1^{\text{oracle pair}}
$$

uses the correct pair of comparable observations, while

$$
F_1^{\text{retrieved pair}}
$$

uses the observations selected by the full retrieval pipeline.

The difference between them measures how much end-to-end performance is lost
before the change model receives its inputs.

### Uncertainty and abstention

Useful metrics include:

- false change rate under viewpoint-only variation;
- false absence rate under occlusion;
- risk-coverage curves for abstention thresholds;
- calibration error of confidence estimates;
- proportion of questions returned as unsupported because evidence is missing.

### Operational metrics

Research accuracy is not sufficient for an application. Also measure:

- ingestion time per minute of video;
- index size per inspection hour;
- query latency;
- peak memory and accelerator use;
- percentage of required locations adequately covered;
- manual review time saved;
- number of unsupported claims correctly rejected.

## Failure atlas

Failures should remain first-class research artifacts.

| Failure | Meaning |
|---|---|
| Wrong place | A visually similar but physically different location was retrieved |
| Wrong traversal | The correct place came from an unintended visit |
| Missing coverage | The selected visit never recorded usable evidence |
| Poor alignment | Both frames show the place but not comparable content |
| Viewpoint false change | Camera motion was interpreted as world change |
| Lighting false change | Exposure or illumination was interpreted as state change |
| Occlusion false absence | A hidden object was reported as removed |
| Identity confusion | Two similar objects were treated as one physical instance |
| Moved versus removed | An object changed position but was reported as missing |
| Temporal boundary error | A nearby but incorrect visit was selected |
| Unsupported certainty | The system made a claim despite insufficient evidence |

Every end-to-end error should be assigned to the earliest stage that made the
correct outcome impossible. This avoids blaming the final model for failures
caused by missing data or incorrect retrieval.

## Current evidence from the repository

Phases 1 through 5 already establish several parts of this programme:

- reproducible simulated observations with exact state metadata;
- frozen CLIP visual memory with exact cosine search;
- real-image evaluation on 10,000 7-Scenes Office frames;
- camera-pose-grounded relevance rather than similarity-only scoring;
- held-out sequence evaluation;
- VLM-assisted semantic place zones;
- an evidence-first local application;
- cross-traversal evaluation that separates route coverage from ranking quality.

The Phase 5 experiment evaluated 24,000 query-target combinations. Under the
strict pose criterion, only 17.1% contained a comparable observation in the
selected traversal. On those covered cases, CLIP achieved 36.8% Hit@1 compared
with 3.6% for random selection within the same traversal.

This result is operationally meaningful: the recording route is a major part
of memory quality. A stronger model cannot retrieve evidence that was never
captured.

## Candidate research progression

### Current direction: object-aware office memory

- adopted four logically ordered, aligned RGB-D/3D observations from the ETH
  ASL Change Detection Office dataset;
- produced a 96-frame visual audit and compared all six mesh pairs;
- measured bidirectional point-to-point and point-to-plane disagreement;
- generated 917 geometric candidate clusters at the 5 cm baseline;
- reviewed the 72 largest candidates with a strict VLM schema and retained 47
  medium/high-confidence candidates in a pseudo-reference;
- documented fragmentation, reconstruction boundaries, missing coverage, and
  VLM uncertainty without claiming human-labelled accuracy.

The current implementation starts with frozen RGB object localization and
inspectable evidence. Depth, 3D placement, and temporal change reasoning will
be added only when they answer a concrete user question.

### Later learned improvement for a measured bottleneck

- freeze the current evidence and reports as the baseline;
- select cluster association, correspondence, or RGB-D candidate
  classification based on the measured failure distribution;
- train on a separately labelled or synthetic source;
- evaluate transfer to ETH Office and compare against the frozen baseline.

### Phase 7: Object-centric memory

- detect or segment task-relevant objects;
- maintain within-visit tracks;
- evaluate cross-visit instance association;
- support where-seen and last-seen queries only where identity and chronology
  are defensible.

### Phase 8: 3D-aware memory

- back-project RGB-D observations into a shared coordinate system;
- measure spatial overlap and visibility;
- anchor objects and observations in world coordinates;
- distinguish object motion from camera motion using geometry.

### Phase 9: Learned improvement for a measured bottleneck

- select one failure exposed by the previous baselines;
- train a representation, matcher, or change component for that failure;
- compare with frozen and geometric baselines;
- run ablations and extend the failure atlas.

### Phase 10: Inspection-video and deployment study

- ingest continuous video;
- select useful frames and preserve temporal context;
- profile latency, storage, and memory;
- test ONNX or another optimized runtime only if deployment measurements justify
  it;
- expose the stable workflow through the demo application.

This sequence is a research direction, not a commitment to add infrastructure
before the underlying questions are measured.

## Commercial value without exaggerated claims

The commercial value is not that the system contains an embedding model or a
chat interface. It comes from reducing costly human effort and improving the
quality of recorded evidence:

- faster review of large inspection archives;
- easier comparison with the correct earlier recording;
- early discovery of missing route coverage;
- traceable evidence for maintenance and compliance decisions;
- fewer false change reports caused by viewpoint or occlusion;
- clearer identification of questions the recordings cannot answer.

The system should initially assist human review rather than claim autonomous
inspection. Human confirmation remains appropriate when consequences are
financial, legal, medical, or safety-critical.

## Claims the project should not make prematurely

Visual Memory Lab should not claim that:

- sequence identifiers are real timestamps;
- a missing detection proves that an object was absent;
- visual similarity proves physical identity;
- a generated explanation is ground truth;
- temporal retrieval alone is a learned world model;
- supplied camera pose demonstrates a working localisation system;
- a benchmark result automatically transfers to field conditions;
- one domain-specific pilot establishes universal deployment readiness.

Clear boundaries make the positive results more credible.

## Positioning

The project can be described as:

> An evidence-grounded visual memory system for repeated real-world
> inspections, studying how to retrieve comparable prior observations and
> distinguish physical state change from viewpoint, visibility, and recording
> coverage variation.

This positioning connects real application value with a non-trivial research
problem. Detection, tracking, model training, 3D geometry, multimodal retrieval,
and efficient inference are supporting methods. They should be introduced when
they help answer the central question, not as disconnected demonstrations.

## Decision principles

Future work should follow these rules:

1. Begin with a real question a user would ask.
2. Define what evidence is required to answer it.
3. Separate missing evidence from model failure.
4. Use the simplest credible baseline first.
5. Add model training only for a measured bottleneck.
6. Use geometry where appearance alone is ambiguous.
7. Evaluate oracle components and the end-to-end pipeline separately.
8. Preserve uncertainty, provenance, and abstention.
9. Promote research work to the demo only after it becomes stable and useful.
10. Prefer one coherent research programme over a collection of unrelated
    features.

## References

- Microsoft Research, [RGB-D Dataset 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).
- Jamie Shotton et al., [Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images](https://www.microsoft.com/en-us/research/publication/scene-coordinate-regression-forests-for-camera-relocalization-in-rgb-d-images-2/), CVPR 2013.
- Alec Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html), ICML 2021.
- [Phase 5: Cross-Traversal Revisit Memory](../phases/05_cross_traversal_memory.md).
