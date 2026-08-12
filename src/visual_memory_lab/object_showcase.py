"""Read-only Phase 6B1 artifacts for the local React application."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


@dataclass(frozen=True)
class ObjectShowcase:
    payload: dict[str, object]
    images: dict[str, Path]

    @classmethod
    def load(cls, *, localization: Path, audit: Path | None = None) -> "ObjectShowcase":
        root = localization.resolve()
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        frames = _jsonl(root / "frames.jsonl")
        detections = _jsonl(root / "detections.jsonl")
        detection_by_frame: dict[str, list[dict[str, object]]] = {}
        for item in detections:
            detection_by_frame.setdefault(str(item["frame_id"]), []).append(item)

        audit_summary: dict[str, object] | None = None
        frame_audits: dict[str, dict[str, object]] = {}
        detection_audits: dict[str, dict[str, object]] = {}
        if audit is not None and (audit / "summary.json").is_file():
            audit_summary = json.loads((audit / "summary.json").read_text(encoding="utf-8"))
            for item in _jsonl(audit / "frame-audits.jsonl"):
                frame_audits[str(item["frame_id"])] = item
                for judgment in item["detections"]:
                    detection_audits[str(judgment["detection_id"])] = judgment

        images: dict[str, Path] = {}
        payload_frames: list[dict[str, object]] = []
        for frame in frames:
            frame_id = str(frame["frame_id"])
            slug = frame_id.replace(":", "-")
            raw_id = f"{slug}-raw"
            overlay_id = f"{slug}-overlay"
            images[raw_id] = (root / str(frame["image_path"])).resolve()
            images[overlay_id] = (root / str(frame["overlay_path"])).resolve()
            payload_detections: list[dict[str, object]] = []
            for detection in detection_by_frame.get(frame_id, []):
                detection_id = str(detection["detection_id"])
                mask_id = f"{detection_id.replace(':', '-')}-mask"
                images[mask_id] = (root / str(detection["mask_path"])).resolve()
                payload_detections.append(
                    {
                        **detection,
                        "mask_url": f"/api/phase6b1/images/{mask_id}",
                        "audit": detection_audits.get(detection_id),
                        "audit_status": (
                            str(detection_audits[detection_id]["verdict"])
                            if detection_id in detection_audits
                            else "unreviewed"
                        ),
                    }
                )
            frame_audit = frame_audits.get(frame_id)
            payload_frames.append(
                {
                    **frame,
                    "image_url": f"/api/phase6b1/images/{raw_id}",
                    "overlay_url": f"/api/phase6b1/images/{overlay_id}",
                    "detections": payload_detections,
                    "audit_status": "reviewed" if frame_audit is not None else "unreviewed",
                    "missed_visible_classes": (
                        frame_audit.get("missed_visible_classes", []) if frame_audit else []
                    ),
                    "audit_limitations": (
                        frame_audit.get("overall_limitations", []) if frame_audit else []
                    ),
                }
            )

        for path in images.values():
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"Phase 6B1 image escapes artifact root: {path}") from error
            if not path.is_file():
                raise FileNotFoundError(path)

        payload = {
            "dataset": "ETH ASL Change Detection: Office",
            "claim_boundary": run["claim_boundary"],
            "metrics": {
                "frame_count": run["frame_count"],
                "detection_count": run["detection_count"],
                "frames_with_detections": run["frames_with_detections"],
                "empty_frame_count": run["empty_frame_count"],
                "class_counts": run["class_counts"],
                "frames_per_observation": run["frames_per_observation"],
            },
            "method": {
                "prompt": run["prompt"],
                "box_threshold": run["box_threshold"],
                "text_threshold": run["text_threshold"],
                "nms_iou": run["nms_iou"],
                "detector": run["detector"],
                "segmenter": run["segmenter"],
            },
            "audit": audit_summary,
            "frames": payload_frames,
        }
        return cls(payload=payload, images=images)

    def image_path(self, image_id: str) -> Path:
        try:
            return self.images[image_id]
        except KeyError as error:
            raise KeyError(f"unknown Phase 6B1 image ID: {image_id}") from error
