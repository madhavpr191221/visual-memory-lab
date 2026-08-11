# What Can a User Ask the Visual Memory Lab?

This document describes the questions the Visual Memory Lab is intended to
answer. It separates capabilities that already exist from questions that need
object-aware memory or additional data.

The system is not meant to be a general office chatbot. Its job is narrower:
retrieve inspectable visual evidence, compare observations of the same physical
place, and state clearly when the evidence is insufficient.

## Capability summary

| Question family | Status | Required evidence |
| --- | --- | --- |
| Place retrieval | Available | RGB embeddings and place-zone metadata |
| Visible scene description | Available with evidence review | Retrieved RGB frames and optional VLM analysis |
| Earlier-visit retrieval | Available | Visit order, pose, and RGB memory |
| Coarse geometric change | Available in Phase 6A | Aligned 3D reconstructions |
| Object localization | Phase 6B target | Detector and segmentation masks |
| Object identity across visits | Phase 6B target | Appearance, geometry, and association |
| Added, removed, or moved objects | Phase 6B target | Identity, visibility, and 3D displacement |
| Long-term object history | Later extension | Persistent object records across visits |
| Cause, person, and real calendar time | Unsupported | Evidence is absent from the datasets |

The status labels have precise meanings:

- **Available** means the current repository can retrieve or compute the
  evidence, although a VLM answer may still require explicit user approval.
- **Phase 6B target** means the question motivates the next object-aware
  research phase; it is not a claim about the current system.
- **Later extension** means the question requires a longer object history or
  another capability beyond Phase 6B.
- **Unsupported** means the available recordings cannot establish the answer.

## 1. Finding places

**Status:** Available now.

These questions use the real-office visual memory built in Phases 3 and 4.
They return ranked RGB evidence and the place-zone labels associated with the
retrieved memories.

- Where is the workstation beside the exterior window?
- Where is the workstation beside the bookshelf?
- Where are the paired desks?
- Where is the central aisle?
- Where is the interior window?
- Which observations show the bookshelf?
- Which observations show the dual-monitor desk?
- Show me the clearest view of the window-side workstation.
- Show me the clearest view of the central aisle.
- Which office zone does this image show?
- Have we seen this part of the office before?
- Which stored views are most similar to this photograph?
- Show other viewpoints of this workstation.
- Which earlier visit contains the closest view of this location?
- Was this area observed during more than one visit?

**Returned evidence:** ranked images, similarity scores, sequence and frame
identifiers, pose metadata, and place-zone agreement.

**Boundary:** a strong visual match is evidence of a similar place or view. It
does not prove that two visually identical objects are the same physical
instance.

## 2. Understanding what is visible

**Status:** Available now with evidence review.

The application retrieves relevant images locally. A user may then select a
small evidence set and explicitly request VLM analysis.

- What is visible around this workstation?
- What furniture is visible in this area?
- What is immediately beside the bookshelf?
- What is between the two workstations?
- Is a chair visible in this image?
- Is a waste bin visible near the desk?
- Is a box visible in this area?
- Are two monitors visible on this desk?
- Does the aisle appear obstructed?
- Does the desk appear cluttered?
- Which image gives the best view of the floor?
- Which image gives the best view beneath the desk?
- Which observations support this description?
- Do the retrieved images agree about the location?
- Are the available images too ambiguous to answer?

**Returned evidence:** selected RGB frames, cited observation identifiers, an
evidence-grounded answer, evidence strength, and stated limitations.

**Boundary:** the answer describes only what is visible. An object missing from
one image may be outside the field of view or hidden behind something else.

## 3. Retrieving earlier visits

**Status:** Available now.

Phase 5 treats each sequence as an ordered visit. It tests whether the memory
can retrieve the same place from a different traversal before attempting to
reason about change.

- Show the closest matching observation from an earlier visit.
- What did this location look like during the previous visit?
- Which previous visit contains the same office area?
- Retrieve the most recent earlier view of this workstation.
- Show earlier views taken from a similar camera position.
- Show earlier views even if the camera position changed.
- Did retrieval find the correct place but the wrong visit?
- Is this result from the immediately previous visit or an older one?
- Which earlier observation is spatially closest to this view?
- How much did the camera viewpoint change between these observations?

**Returned evidence:** current and retrieved frames, logical visit order,
camera poses, spatial error, and retrieval rank.

**Boundary:** sequence order is a logical visit order. It is not a real date or
calendar timestamp.

## 4. Asking about coarse 3D change

**Status:** Available in Phase 6A.

Phase 6A compares already aligned ETH Office reconstructions. It finds physical
regions whose reconstructed surfaces do not have a nearby counterpart in the
other observation.

- Which physical regions differ between these two office scans?
- Where does the later scan contain surfaces absent from the earlier scan?
- Where does the earlier scan contain surfaces absent from the later scan?
- Which 3D difference region is largest?
- Does the RGB evidence show something near this 3D difference?
- Is this difference compact enough to resemble an object?
- Could this difference be caused by incomplete reconstruction?
- Could this difference be caused by missing coverage?
- Does the geometry support the visible RGB change?
- Do the RGB and geometry disagree?
- Which candidate changes are likely reconstruction artifacts?
- Which regions require human inspection?

**Returned evidence:** before-and-after RGB samples, 3D difference plots,
cluster statistics, and VLM-supported pseudo-reference judgments.

**Boundary:** Phase 6A locates changed geometry. It does not reliably name the
responsible object or establish persistent object identity.

## 5. Locating movable objects

**Status:** Phase 6B target.

Phase 6B will initially concentrate on movable office chairs, waste bins, and
boxes. Detection will locate an object in an RGB frame; segmentation will
separate its pixels from the surrounding room.

- Where is the black office chair?
- Where is the waste bin?
- Where is the cardboard box?
- Show every observation containing an office chair.
- Show every view of this waste bin.
- Which visit contains this box?
- Which workstation is this chair closest to?
- Is the chair beside the desk or beside the window?
- What is the estimated 3D position of this object?
- Which detections from this visit belong to one physical object?
- Which image provides the clearest view of the object?
- Is the object partially occluded?
- Was the object detected confidently?
- Which object classes were searched for but not detected?

**Expected evidence:** model-generated boxes and masks, confidence scores,
frame identifiers, depth support, and an object location in the shared 3D
coordinate system.

**Boundary:** a missing detection does not prove absence. Detector failure,
occlusion, poor lighting, or missing camera coverage remain possible.

## 6. Establishing object identity across visits

**Status:** Phase 6B target.

This is more difficult than category detection. Two images may both contain an
office chair without showing the same physical chair. Phase 6B will combine
appearance, 3D position, and 3D shape to estimate cross-visit identity.

- Is this the same chair seen during the previous visit?
- Are these two chair detections the same physical chair?
- Which earlier observation most likely contains this object?
- Have we seen this particular bin before?
- Is this one box viewed from two angles or two different boxes?
- Which observations were consolidated into this object record?
- Does appearance support the identity match?
- Does 3D shape support the identity match?
- Does spatial position support or contradict the match?
- Are there several equally plausible matches?
- Is the identity uncertain because two objects look alike?
- Is the identity uncertain because the object is poorly visible?

For an earlier object observation $i$ and later observation $j$, a baseline
association cost can combine appearance, position, and shape:

$$
C_{ij}
=
\lambda_a\left(1-\cos(\mathbf e_i,\mathbf e_j)\right)
+
\lambda_p\lVert\mathbf x_i-\mathbf x_j\rVert_2
+
\lambda_s d_{\mathrm{shape}}(i,j).
$$

Here, $\mathbf e$ is a visual embedding, $\mathbf x$ is a 3D object position,
and $d_{\mathrm{shape}}$ measures disagreement between object shapes.

**Expected evidence:** paired crops, masks, compact 3D object crops, component
scores, and an association confidence.

**Boundary:** the system should say **likely the same object** unless persistent
identity has been independently verified.

## 7. Asking whether an object moved

**Status:** Phase 6B target.

- Did this chair move between the two visits?
- Where was this chair during the earlier visit?
- Where is it during the later visit?
- How far did the object move in 3D?
- In which direction did it move?
- Did it move from the desk to the window area?
- Was the object moved within the same office zone?
- Did it move into another office zone?
- Which movable objects changed position?
- Which objects remained in approximately the same position?
- Is the apparent movement explained only by camera motion?
- Does the RGB evidence support the estimated 3D displacement?
- Could pose or reconstruction error explain the displacement?
- Is there enough evidence to classify the object as moved?

For a matched object with positions $\mathbf x_{t-1}$ and $\mathbf x_t$,

$$
d_{\mathrm{move}}
=
\left\lVert\mathbf x_t-\mathbf x_{t-1}\right\rVert_2.
$$

A movement claim requires both a credible identity match and displacement above
a validated uncertainty threshold. Pixel displacement alone is not movement:
the camera may simply have moved.

## 8. Asking whether an object was added or removed

**Status:** Phase 6B target.

- Which objects appear in the later visit but not the earlier visit?
- Was a box added to this area?
- Was the waste bin removed?
- Which objects disappeared from this workstation?
- Which objects are new in the current scan?
- Was this chair removed or merely moved elsewhere?
- Could the missing object be hidden behind another object?
- Was the earlier location observed clearly in the later visit?
- Did the detector miss an object that is visibly present?
- Is the object absent, or do we lack sufficient coverage?
- Does later-only geometry support an added object?
- Does earlier-only geometry support a removed object?
- Do an earlier-only and later-only region form a plausible moved-object pair?
- Is this classification supported by both RGB and 3D evidence?

**Expected evidence:** object detections from both visits, visibility and
coverage information, cross-visit association results, and nearby 3D change.

**Boundary:** “not detected” and “not present” are different conclusions. A
responsible system must preserve that distinction.

## 9. Supporting inspection and maintenance work

**Status:** some retrieval questions are available now; object-level change
questions are Phase 6B targets.

- What changed in this work area since the previous inspection?
- Which movable items require review?
- Is the aisle still clear?
- Has a chair been left in the walkway?
- Has a box appeared near the workstation?
- Has the waste bin changed location?
- Which changes are supported by strong evidence?
- Which changes are uncertain?
- Which areas were not adequately observed?
- Where should the technician take another photograph?
- Which object needs another viewpoint before making a decision?
- Which result is most likely a reconstruction artifact?
- Show the before-and-after evidence for this reported change.
- Explain why the system believes the object moved.
- What evidence contradicts the proposed change?
- Can this result be safely accepted without another inspection?

A useful answer is not merely “chair moved.” It should provide the earlier and
later RGB views, highlight the relevant object, show its 3D evidence, report
confidence, and describe alternative explanations.

## 10. Asking about longer object histories

**Status:** Later extension.

These questions become meaningful after the system maintains persistent object
records across several visits.

- Where was this object last seen?
- During which logical visit was it last observed?
- Show the object's location history across all visits.
- How many times has it changed position?
- What was its most common location?
- Has it repeatedly moved between the same two areas?
- How long has it been missing in logical visit order?
- Which objects change location most frequently?
- Which objects normally remain stationary?
- What was this area like during each recorded visit?
- When did this state first appear in the recorded sequence?
- Has the area returned to a previously observed state?

These are logical-history questions. Answering them with real times requires a
dataset that records trustworthy calendar timestamps.

## 11. Questions the system must refuse or qualify

The available evidence cannot responsibly answer the following questions:

- Who moved the chair?
- Why was the object moved?
- At what real date or time was it moved?
- What happened between two scans when no camera recorded it?
- Is the object definitely stolen?
- Is an undetected object definitely absent?
- Is this definitely the same physical chair?
- What is inside a closed box?
- Is equipment operational when its state is not visibly measurable?
- Is the office legally or professionally safety-compliant?
- What happened outside the recorded field of view?
- Did a person perform a particular action?

The correct response is not a guess. It should identify the missing evidence,
show what is known, and recommend another observation when that would resolve
the uncertainty.

## A realistic interaction

A facilities technician asks:

> Did the black chair move between the previous and current inspection?

An object-aware system should perform the following reasoning:

1. Retrieve observations of the same office area from two consecutive visits.
2. Detect and segment chair candidates in both visits.
3. Combine multiple views of each candidate within its visit.
4. Project the chair observations into the common 3D coordinate system.
5. Estimate whether the two observations show the same physical chair.
6. Measure the displacement only if the identity match is credible.
7. Check whether RGB evidence and 3D change evidence agree.
8. Return the before-and-after evidence and an uncertainty-aware conclusion.

A suitable result might be:

> The observations likely show the same black office chair. Its estimated
> location changed by 1.2 m from the monitor desk to the window area. Appearance
> and 3D shape support the match, but the dataset does not provide a persistent
> object ID, so the identity remains probable rather than proven.

That distinction between visible evidence, model judgment, and verified fact is
central to the Visual Memory Lab.
