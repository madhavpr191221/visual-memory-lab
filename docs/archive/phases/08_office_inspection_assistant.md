# Phase 8: Office Inspection Assistant

Phase 8 turns the existing evidence explorer into a technician-style inspection
workflow. A user can ask a question or upload a current office image, review
five retrieved memories (expandable to ten), optionally choose an earlier view,
and save the result locally.

After saving, the user can choose one retrieved memory as the earlier view.
The assistant then shows the current view and earlier view side by side, with
their sequence, frame, and zone metadata. An uploaded current photograph is
stored under the Phase 8 output directory; a text-only inspection uses the
selected retrieved frame as its current view. The comparison remains cautious:
it explains what is visible, but does not turn appearance or geometry into a
guaranteed object identity or movement claim.

The first version uses SQLite for inspection metadata. CLIP embeddings remain in
the existing NumPy index; PostgreSQL/pgvector is a future scaling option, not a
new dependency here.

The assistant always returns a plain-language status, but it distinguishes a
supported answer from a possible candidate, missing evidence, and a question
that needs manual review. It does not prove movement, identity, or absence.

## Technician inspection report

When a current image is uploaded, the application can request a short visual
summary. The summary lists visible objects, visible conditions, and limitations
such as blur or occlusion. It describes the current photograph only; it does
not claim what happened earlier.

After the user chooses an earlier memory, **Generate inspection report** combines
the current image, the earlier evidence, and the maintenance question. The
structured report contains a cautious status, visible objects and conditions,
comparison observations, evidence citations, limitations, and a recommended
manual check. Reports are saved with the inspection metadata in SQLite. If the
cloud VLM is unavailable, local retrieval and side-by-side comparison still
work, but the summary/report sections remain unavailable.
