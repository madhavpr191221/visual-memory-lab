from __future__ import annotations

from PIL import Image

from visual_memory_lab.video_object_evidence import VideoObjectEvidence


class FakeDetector:
    provenance = {"model_id": "fake", "device": "cpu"}

    def detect(self, image: Image.Image) -> list[dict[str, object]]:
        del image
        return [{"phrase": "laptop", "score": 0.8, "box_xyxy": [1, 2, 8, 9]}]


class FakeSegmenter:
    def segment(self, image: Image.Image, boxes: list[list[float]]) -> list[tuple[object, float]]:
        import numpy as np
        return [(np.ones((image.height, image.width), dtype=bool), 0.9) for _ in boxes]


def test_object_evidence_reports_frame_coverage_and_boxes() -> None:
    service = VideoObjectEvidence(detector=FakeDetector(), segmenter=FakeSegmenter())
    frames = [
        ("f0", 1.0, Image.new("RGB", (10, 10))),
        ("f1", 2.0, Image.new("RGB", (10, 10))),
    ]
    result = service.inspect(frames, object_prompts=["laptop"])
    assert result["status"] == "detected"
    assert result["objects"][0]["status"] == "supported"
    assert result["objects"][0]["frames_visible"] == 2
    assert result["frames"][0]["detections"][0]["box_normalized"] == [0.1, 0.2, 0.8, 0.9]
    assert result["frames"][0]["detections"][0]["track_id"] == result["frames"][1]["detections"][0]["track_id"]
    assert result["frames"][0]["detections"][0]["mask_available"] is True
