# Phase 6.1: Object-Aware Change Memory

## The idea in one sentence

Phase 6.1 is about helping a technician remember **physical objects**, not just
similar-looking images.

For example:

> “Was the black office chair still beside the workstation during the latest
> inspection, or had somebody moved it near the window?”

To answer that responsibly, the system needs to know:

1. what object appears in each image;
2. which pixels belong to that object;
3. where those pixels are in the room;
4. whether observations from two visits may refer to the same object;
5. whether the evidence is strong enough to describe a change.

The project is building these capabilities in stages. Phase 6.1.1 implements the
first two for real office images. The later stages are the research plan, not
completed results.

## A technician's complete workflow

Imagine a facilities technician carrying an RGB-D camera through an office.
The camera records colour, depth, and its position as it moves.

During the first inspection:

```text
The chair is detected beside the desk.
Its visible pixels are segmented.
Its visible surface is placed approximately in the room's 3D coordinates.
```

During the second inspection:

```text
The chair is detected near the window.
Its visible pixels are segmented again.
Its new visible surface is placed in the same room coordinates.
```

The system can then present:

> “The two observations are consistent with a possible chair relocation from
> the desk area to the window area. The camera coverage and object identity are
> not strong enough to call this certain.”

That last sentence matters. A visual-memory system should show evidence and
uncertainty, not turn every difference between two images into a confident
claim.

## What is implemented now?

### Phase 6.1.1: RGB object localization

Phase 6.1.1 processes 384 selected ETH Office RGB frames. For each frame:

- Grounding DINO predicts chairs, waste bins, and boxes;
- SAM 2.1 predicts the pixels belonging to each predicted object;
- the artifact stores the image, box, mask, confidence, and camera pose;
- the Objects UI lets us inspect the predictions and failures.

Example output for one frame:

```text
object: chair
detector box: x=381..1138, y=575..716 pixels
detector score: 0.37
segmentation mask: 3.5% of the image
```

This says “the model sees something that may be a chair here.” It does not yet
say “this is the same chair seen in Visit 1.”

### Later Phase 6.1 stages

The planned progression is:

1. turn RGB-D masks into visible 3D object points;
2. compare object observations in the shared room frame;
3. associate likely observations of the same physical object;
4. compare position, shape, and visibility;
5. report possible movement, addition, removal, or uncertainty;
6. train a component only after measuring a real failure mode.

## Three coordinate systems in simple language

When a camera sees a chair, we have to be clear about **where** a point is
being described.

### Image coordinates

An image point is a pixel location:

```text
(u, v) = (640, 380)
```

This means “640 pixels across and 380 pixels down in the image.”

### Camera coordinates

After using the depth value, that same pixel becomes a 3D point relative to the
camera:

```text
the chair point is 1.8 metres in front of the camera,
0.2 metres to its left,
and 0.7 metres above its optical centre.
```

### Room or global coordinates

The camera itself is moving. We therefore transform the camera-relative point
into a fixed room coordinate system:

```text
Visit 1 camera position → room coordinates
Visit 2 camera position → the same room coordinates
```

Now a chair can be compared across visits even when the technician held the
camera from a different position.

## How one RGB-D pixel becomes a 3D point

This is the core mathematical operation.

Suppose the technician's camera observes a chair pixel at image location
$(u,v)$. The depth sensor says that the point is $z$ metres away. The camera
calibration tells us how image pixels correspond to rays leaving the camera.

The calibration matrix is

$$
K=
\begin{bmatrix}
f_x & 0 & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix}.
$$

Here:

- $f_x$ and $f_y$ describe the camera's focal lengths;
- $(c_x,c_y)$ is the image centre, called the principal point.

The pixel is back-projected into camera-relative 3D coordinates with

$$
\mathbf p_C(u,v,z)
=zK^{-1}
\begin{bmatrix}u\\v\\1\end{bmatrix}.
$$

In ordinary language:

> Start with the pixel's viewing ray and extend it by the measured depth.

Written out component by component:

$$
X_C=(u-c_x)\frac{z}{f_x},
\qquad
Y_C=(v-c_y)\frac{z}{f_y},
\qquad
Z_C=z.
$$

### Numerical example

Suppose:

```text
pixel:       u = 640, v = 380
depth:       z = 2.0 m
focal length: fx = fy = 600 pixels
image centre: cx = 640, cy = 360
```

Then:

$$
X_C=(640-640)\frac{2.0}{600}=0.00\text{ m},
$$

$$
Y_C=(380-360)\frac{2.0}{600}\approx0.067\text{ m},
$$

$$
Z_C=2.0\text{ m}.
$$

So this pixel lies approximately two metres in front of the camera and 6.7 cm
below the image centre's horizontal ray.

## How the camera-relative point becomes a room point

The camera pose is recorded as `T_G_C`. We use the notation
$T_{G\leftarrow C}$ to say:

> transform a point from Camera coordinates into Global/room coordinates.

The pose contains a rotation $R_{G\leftarrow C}$ and a translation
$\mathbf t_{G\leftarrow C}$:

$$
T_{G\leftarrow C}=
\begin{bmatrix}
R_{G\leftarrow C} & \mathbf t_{G\leftarrow C}\\
\mathbf 0^\top & 1
\end{bmatrix}.
$$

The transformation is

$$
\mathbf p_G
=R_{G\leftarrow C}\mathbf p_C
+\mathbf t_{G\leftarrow C}.
$$

In plain English:

> Rotate the camera-relative point according to the camera's orientation, then
> add the camera's room position.

### Technician example

On Visit 1, a chair pixel becomes:

```text
camera coordinates: (0.0, 0.07, 2.0) metres
camera pose: camera beside the desk, facing the room
room coordinates: (2.1, 0.8, 0.6) metres
```

On Visit 2, the technician approaches from the opposite side. The same physical
chair may have completely different image coordinates, but after applying the
second camera pose its room-relative points can still land near:

```text
room coordinates: (2.1, 0.8, 0.6) metres
```

That is how different-looking images can still describe the same physical
location.

## From an object mask to a 3D object cloud

This sounds complicated, but the operation is straightforward:

1. Find every pixel marked as chair by the segmentation mask.
2. Look up the depth value at that pixel.
3. Ignore pixels with missing or invalid depth.
4. Convert every remaining pixel into a 3D room point.

Let $M(u,v)$ be the mask:

```text
M(u,v) = 1  → the pixel belongs to the predicted chair
M(u,v) = 0  → it does not
```

Let $D(u,v)$ be the depth image. The visible 3D chair points are

$$
\mathcal P_G(M,D)=
\left\{
\mathbf p_G(u,v,D(u,v))
\;\middle|\;
M(u,v)=1,\ D(u,v)>0
\right\}.
$$

In English:

> Keep only pixels that the mask calls “chair” and for which the depth sensor
> returned a valid distance. Convert those pixels to room-relative 3D points.

### What does the cloud look like?

It is a collection of points sampled from the visible chair surface:

```text
        • • • •       ← visible chair back
      • • • • • •
          •          ← support
     • • • • •        ← seat
   •     •     •      ← visible wheels
```

It is not a complete CAD model. The camera cannot see the hidden side of the
chair, the underside of the seat, or the part blocked by a desk.

### What if some depth values are bad?

Depth cameras sometimes return missing values or isolated measurements that are
far too large. We can summarize the visible points with a robust centroid:

$$
\bar{\mathbf p}_G
=\operatorname{median}_{\mathbf p\in\mathcal P_G}\mathbf p.
$$

This means “take the middle coordinate value rather than letting a few extreme
measurements pull the result away.”

For example:

```text
valid chair-point x coordinates: 2.0, 2.1, 2.1, 2.2, 9.8
ordinary mean:                  3.64
median:                         2.1
```

The value `9.8` is probably a bad depth measurement. The median gives a much
more sensible approximate chair location.

Other useful summaries are:

- the 3D width, height, and depth of the visible points;
- the number of valid depth pixels;
- the fraction of the mask with usable depth;
- the viewpoints from which the object was observed.

## Comparing two visits

Let the earlier visible chair points be $\mathcal P_G^{(a)}$ and the later
points be $\mathcal P_G^{(b)}$.

If both clouds use the same room coordinate system, we can compare their
approximate centres:

$$
\Delta_{\text{position}}
=
\left\|
\bar{\mathbf p}_G^{(b)}-
\bar{\mathbf p}_G^{(a)}
\right\|_2.
$$

In English:

> Measure the straight-line distance between the earlier and later estimated
> chair positions.

### Numerical example

```text
earlier chair centre: (2.1, 0.8, 0.6) m
later chair centre:   (3.3, 1.2, 0.6) m
```

Then:

$$
\Delta_{\text{position}}
=\sqrt{(3.3-2.1)^2+(1.2-0.8)^2+(0.6-0.6)^2}
\approx1.26\text{ m}.
$$

A 1.26-metre displacement is worth investigating. It is not automatically
proof of movement: the two detections might be different chairs, or one cloud
may be badly reconstructed.

We can also compare surfaces. For an earlier point $\mathbf q$, find its nearest
point in the later cloud:

$$
d(\mathbf q,\mathcal P_G^{(b)})
=\min_{\mathbf p\in\mathcal P_G^{(b)}}
\left\|\mathbf q-\mathbf p\right\|_2.
$$

In English:

> For every earlier surface point, ask how far away the closest later surface
> point is.

Large distances may indicate a moved or missing surface. They may also indicate
occlusion, noisy depth, or incomplete reconstruction, so the result needs RGB
and coverage evidence.

## Why “not detected” does not mean “removed”

Suppose the chair appeared in Visit 1 but not Visit 2. There are at least three
possibilities:

```text
1. The chair was removed.
2. The chair is still there but hidden behind a cabinet.
3. The detector failed on the later image.
```

The system therefore needs coverage information. A simple coverage ratio can be
written as

$$
C(\Omega)=
\frac{\text{observed samples from region }\Omega}
{\text{samples expected to be visible in region }\Omega}.
$$

If the old chair location was clearly visible in the later inspection and no
chair was found, removal becomes more plausible. If that area was hidden or
never scanned, the correct result is `uncertain`.

Technician-facing wording:

```text
Good coverage + earlier chair + no later counterpart
    → possible removal

Poor coverage + earlier chair + no later counterpart
    → cannot determine whether it was removed
```

## Associating observations of the same object

Two chairs that look alike are not automatically the same physical chair. A
future association stage can combine several clues:

$$
S(i,j)=
w_eS_{\text{appearance}}(i,j)
+w_xS_{\text{position}}(i,j)
+w_sS_{\text{shape}}(i,j)
+w_zS_{\text{zone}}(i,j).
$$

In English:

> Give a higher match score when two observations have similar appearance,
> plausible room positions, compatible shape, and a consistent office zone.

Example:

```text
same black chair appearance:       strong evidence
similar visible 3D shape:           supporting evidence
desk area → window area:             plausible displacement
poor later coverage:                 uncertainty penalty
```

This score is only a matching aid until calibrated on labelled object pairs. It
is not automatically a probability of identity.

## Technician-facing outcomes

The eventual interface should report one interpretable outcome.

### Possible move

The same object class and appearance are visible in both visits, the estimated
3D regions are displaced, and both locations have reasonable coverage.

Example:

> “Possible chair move: desk area to window area. RGB appearance and visible 3D
> shape are compatible, but persistent identity is not proven.”

### Possible addition

The later visit contains a well-supported object, no comparable earlier object is
found, and the relevant earlier region had adequate coverage.

Example:

> “Possible added storage box near the printer. The earlier view covers this
> shelf, but no box was detected then.”

### Possible removal

The earlier object is well supported, its old location is visible later, and no
later counterpart is found.

Example:

> “Possible chair removal from the workstation. The later inspection visibly
> covers the old chair area, but absence is not certain.”

### Uncertain

The object is occluded, the mask is weak, the depth is missing, the view is too
different, or two similar objects cannot be distinguished.

Example:

> “Uncertain: the later camera view does not cover the space behind the cabinet.”

## What is and is not claimed

Phase 6.1.1 currently provides automatic RGB boxes and masks. The full
object-aware interpretation remains
future work.

The project does not currently claim:

- persistent object IDs;
- reliable object tracking across visits;
- calibrated object-level precision and recall;
- complete 3D object reconstruction from one mask;
- physically verified movement, addition, or removal;
- knowledge of why a technician moved an object.

The equations in this document define how the next stages can work. They should
not be mistaken for completed evaluation results.

## Why this matters in practice

Once evaluated properly, this system could support questions such as:

- “Which workstation had its chair moved?”
- “Was the emergency route blocked during the last inspection?”
- “Where was the storage box last seen?”
- “Did the replacement component appear after the service visit?”
- “What changed in this room since the previous inspection?”

The goal is not merely to retrieve a similar picture. It is to maintain
evidence-grounded memory of physical objects in a changing environment.
