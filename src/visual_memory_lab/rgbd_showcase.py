"""Read-only API payload for the Phase 6.1.2 RGB-D comparison view."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@dataclass(frozen=True)
class RgbdShowcase:
    payload: dict[str, object]

    @classmethod
    def load(cls, *, evidence: Path, localization: Path) -> "RgbdShowcase":
        root = evidence.resolve()
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        rows = _jsonl(root / "evidence.jsonl")
        frames = {str(item["frame_id"]): item for item in _jsonl(localization / "frames.jsonl")}
        detections = {str(item["detection_id"]): item for item in _jsonl(localization / "detections.jsonl")}
        by_key: dict[tuple[int, str], list[dict[str, object]]] = {}
        for row in rows:
            detection = detections.get(str(row["detection_id"]))
            frame = frames.get(str(row["frame_id"]))
            if detection is None or frame is None:
                continue
            enriched = {
                **row,
                "score": detection["score"],
                "mask_score": detection["mask_score"],
                "mask_area_fraction": detection["mask_area_fraction"],
                "image_url": f"/api/objects/images/{str(row['frame_id']).replace(':', '-')}-raw",
                "mask_url": f"/api/objects/images/{str(row['detection_id']).replace(':', '-')}-mask",
                "message_index": frame["message_index"],
                "width": frame["width"],
                "height": frame["height"],
            }
            by_key.setdefault((int(row["observation"]), str(row["canonical_class"])), []).append(enriched)

        comparisons: list[dict[str, object]] = []
        classes = sorted({str(row["canonical_class"]) for row in rows})
        for earlier in range(3):
            for later in range(earlier + 1, 4):
                for object_class in classes:
                    old = by_key.get((earlier, object_class), [])
                    new = by_key.get((later, object_class), [])
                    if not old or not new:
                        continue
                    comparisons.append({
                        "id": f"{earlier}-to-{later}-{object_class}",
                        "earlier_observation": earlier,
                        "later_observation": later,
                        "object_class": object_class,
                        "earlier": max(old, key=lambda item: int(item["point_count"])),
                        "later": max(new, key=lambda item: int(item["point_count"])),
                        "interpretation": "The visible evidence differs between these visits. This phase does not establish persistent identity or prove movement.",
                    })
        return cls(payload={
            "phase": "6.1.2",
            "dataset": run["dataset"],
            "claim_boundary": run["claim_boundary"],
            "metrics": run,
            "classes": classes,
            "comparisons": comparisons,
        })
