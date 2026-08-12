"""Tests for the read-only Phase 6B1 showcase."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from visual_memory_lab.object_showcase import ObjectShowcase


def make_object_showcase(root: Path) -> ObjectShowcase:
    localization = root / "localization"
    audit = root / "audit"
    localization.mkdir(parents=True)
    audit.mkdir()
    raw = localization / "raw.jpg"
    overlay = localization / "overlay.jpg"
    mask = localization / "mask.png"
    Image.new("RGB", (20, 12), "white").save(raw)
    Image.new("RGB", (20, 12), "gray").save(overlay)
    Image.new("RGBA", (20, 12), (0, 0, 255, 100)).save(mask)
    frame_id = "eth-office:0:000001"
    detection_id = f"{frame_id}:det-00"
    run = {
        "schema_version": 1,
        "claim_boundary": "Predictions are not ground truth.",
        "frame_count": 1,
        "detection_count": 1,
        "frames_with_detections": 1,
        "empty_frame_count": 0,
        "class_counts": {"chair": 1},
        "frames_per_observation": {"0": 1},
        "prompt": "office chair.",
        "box_threshold": 0.25,
        "text_threshold": 0.2,
        "nms_iou": 0.5,
        "detector": {"model_id": "fake-detector"},
        "segmenter": {"model_id": "fake-segmenter"},
    }
    frame = {
        "frame_id": frame_id,
        "observation": 0,
        "message_index": 1,
        "timestamp_ns": 1,
        "width": 20,
        "height": 12,
        "pose": {"frame": "T_G_C", "translation_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]},
        "image_path": raw.name,
        "overlay_path": overlay.name,
        "detection_ids": [detection_id],
        "detection_count": 1,
    }
    detection = {
        "detection_id": detection_id,
        "frame_id": frame_id,
        "canonical_class": "chair",
        "phrase": "office chair",
        "score": 0.9,
        "box_xyxy": [1, 1, 10, 10],
        "box_normalized": [0.05, 0.08, 0.5, 0.83],
        "mask_path": mask.name,
        "mask_score": 0.8,
        "mask_area_fraction": 0.2,
        "warnings": [],
    }
    (localization / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (localization / "frames.jsonl").write_text(json.dumps(frame) + "\n", encoding="utf-8")
    (localization / "detections.jsonl").write_text(json.dumps(detection) + "\n", encoding="utf-8")
    (audit / "summary.json").write_text(
        json.dumps(
            {
                "frame_count": 1,
                "reviewed_detection_count": 1,
                "verdict_counts": {"supported": 1},
                "mask_quality_counts": {"good": 1},
                "missed_visible_class_counts": {},
                "high_confidence_pseudo_support_rate": 1.0,
                "claim_boundary": "A pseudo-audit, not ground truth.",
                "model_requested": "fake-vlm",
                "response_models": ["fake-vlm"],
            }
        ),
        encoding="utf-8",
    )
    (audit / "frame-audits.jsonl").write_text(
        json.dumps(
            {
                "frame_id": frame_id,
                "detections": [
                    {
                        "detection_id": detection_id,
                        "verdict": "supported",
                        "category_correct": "yes",
                        "mask_quality": "good",
                        "explanation": "Visible chair.",
                    }
                ],
                "missed_visible_classes": [],
                "overall_limitations": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return ObjectShowcase.load(localization=localization, audit=audit)


def test_showcase_exposes_only_allowlisted_images_and_public_metadata(tmp_path: Path) -> None:
    showcase = make_object_showcase(tmp_path)
    serialized = json.dumps(showcase.payload)
    assert "C:\\" not in serialized
    assert showcase.payload["metrics"]["detection_count"] == 1  # type: ignore[index]
    frame = showcase.payload["frames"][0]  # type: ignore[index]
    assert frame["detections"][0]["audit_status"] == "supported"
    assert showcase.image_path("eth-office-0-000001-raw").is_file()
    try:
        showcase.image_path("../../secret")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown image ID should be rejected")
