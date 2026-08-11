"""Tests for the sampled VLM pseudo-audit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from visual_memory_lab.object_audit import FrameAudit, audit_eth_object_localization


def make_localization(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "run.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    frames = []
    detections = []
    for observation in range(4):
        image = root / f"raw-{observation}.jpg"
        overlay = root / f"overlay-{observation}.jpg"
        Image.new("RGB", (20, 12), "white").save(image)
        Image.new("RGB", (20, 12), "gray").save(overlay)
        frame_id = f"eth-office:{observation}:000000"
        frames.append(
            {
                "frame_id": frame_id,
                "observation": observation,
                "message_index": 0,
                "image_path": image.name,
                "overlay_path": overlay.name,
            }
        )
        detections.append(
            {
                "detection_id": f"{frame_id}:det-00",
                "frame_id": frame_id,
                "canonical_class": "chair",
                "phrase": "office chair",
                "score": 0.9,
            }
        )
    (root / "frames.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in frames), encoding="utf-8"
    )
    (root / "detections.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in detections), encoding="utf-8"
    )
    return root


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        prompt = kwargs["input"][0]["content"][0]["text"]  # type: ignore[index]
        frame_id = re.search(r"Frame ID: (.+)", prompt).group(1)  # type: ignore[union-attr]
        detection_id = re.search(r"- (.+): predicted=", prompt).group(1)  # type: ignore[union-attr]
        parsed = FrameAudit.model_validate(
            {
                "frame_id": frame_id,
                "detections": [
                    {
                        "detection_id": detection_id,
                        "verdict": "supported",
                        "category_correct": "yes",
                        "mask_quality": "good",
                        "explanation": "The chair and mask are visible.",
                    }
                ],
                "missed_visible_classes": [],
                "overall_limitations": [],
            }
        )
        return SimpleNamespace(output_parsed=parsed, model="fake-vlm")


def test_audit_samples_every_observation_and_caches_responses(tmp_path: Path) -> None:
    localization = make_localization(tmp_path / "localization")
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    cache = tmp_path / "cache"

    first = audit_eth_object_localization(
        localization=localization,
        output=tmp_path / "audit-1",
        cache_dir=cache,
        frames_per_observation=1,
        model="fake-vlm",
        client=client,
    )
    second = audit_eth_object_localization(
        localization=localization,
        output=tmp_path / "audit-2",
        cache_dir=cache,
        frames_per_observation=1,
        model="fake-vlm",
        client=client,
    )

    assert first["frame_count"] == 4
    assert first["verdict_counts"] == {"supported": 4}
    assert first["high_confidence_pseudo_support_rate"] == 1.0
    assert second["cache_hits"] == 4
    assert responses.calls == 4
    assert len((tmp_path / "audit-1" / "frame-audits.jsonl").read_text().splitlines()) == 4
