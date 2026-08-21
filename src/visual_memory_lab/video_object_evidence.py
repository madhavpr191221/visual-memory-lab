"""On-demand object evidence for a retrieved video event.

This module deliberately runs only after a user selects an event.  It reuses
the project's Grounding DINO and SAM adapters, but keeps the result small and
traceable: every detection is tied to a frame timestamp and every conclusion
is marked as visual evidence rather than ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from visual_memory_lab.object_localization import (
    HuggingFaceGroundingDino,
    HuggingFaceSam2,
)


def _normalise_label(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split()).strip(" \t\r\n.,;:!?()[]{}\"'")


def _status(frame_count: int, visible_count: int) -> str:
    if visible_count == 0:
        return "not_visibly_confirmed"
    if visible_count == frame_count:
        return "supported"
    if visible_count >= max(1, frame_count // 2):
        return "partially_visible"
    return "unclear"


def _tighten_box_from_mask(detection: dict[str, object], mask: object) -> None:
    """Use a smaller, valid SAM mask box when it improves the detector box."""
    array = np.asarray(mask).astype(bool)
    if array.ndim != 2 or not array.any():
        return
    ys, xs = np.where(array)
    height, width = array.shape
    candidate = [
        float(xs.min()) / width,
        float(ys.min()) / height,
        float(xs.max() + 1) / width,
        float(ys.max() + 1) / height,
    ]
    current = [float(value) for value in detection["box_normalized"]]
    candidate_area = max(0.0, candidate[2] - candidate[0]) * max(0.0, candidate[3] - candidate[1])
    current_area = max(0.0, current[2] - current[0]) * max(0.0, current[3] - current[1])
    if 0.0 < candidate_area < current_area * 0.95:
        detection["box_normalized"] = candidate
def _iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _assign_tracks(frame_results: list[dict[str, object]], threshold: float = 0.25) -> None:
    """Give nearby same-label detections local, non-identity track ids."""
    previous: dict[str, list[tuple[str, list[float]]]] = {}
    counters: dict[str, int] = {}
    for frame in frame_results:
        current: dict[str, list[tuple[str, list[float]]]] = {}
        detections = frame.get("detections", [])
        for detection in detections if isinstance(detections, list) else []:
            label = str(detection.get("label", "object"))
            box = [float(value) for value in detection.get("box_xyxy", [])]
            best_id = None
            best_iou = threshold
            for track_id, previous_box in previous.get(label, []):
                overlap = _iou(box, previous_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_id = track_id
            if best_id is None:
                counters[label] = counters.get(label, 0) + 1
                best_id = f"{label}-{counters[label]}"
            detection["track_id"] = best_id
            current.setdefault(label, []).append((best_id, box))
        previous = current


@dataclass
class VideoObjectEvidence:
    """Lazy Grounding DINO/SAM service used by the video UI."""

    device: str = "auto"
    detector: object | None = None
    detector_prompt: str | None = None
    segmenter: object | None = None

    def _detector(self, prompt: str):
        if self.detector is not None and self.detector_prompt is None:
            return self.detector
        if self.detector is None or self.detector_prompt != prompt:
            if self.detector is not None:
                close = getattr(self.detector, "close", None)
                if callable(close):
                    close()
            self.detector = HuggingFaceGroundingDino(
                device=self.device,
                prompt=prompt,
                box_threshold=0.25,
                text_threshold=0.20,
            )
            self.detector_prompt = prompt
        return self.detector

    def inspect(
        self,
        frames: Sequence[tuple[str, float, Image.Image]],
        *,
        object_prompts: Sequence[str],
    ) -> dict[str, object]:
        prompts = sorted({_normalise_label(value) for value in object_prompts if _normalise_label(value)})
        if not frames:
            return {"status": "unavailable", "frames": [], "objects": [], "limitations": ["No RGB frames were available for inspection."]}
        if not prompts:
            return {"status": "unavailable", "frames": [], "objects": [], "limitations": ["No object terms could be derived from the question or event."]}

        prompt = ". ".join(f"{value}." for value in prompts)
        segmentation_limitation = ""
        try:
            detector = self._detector(prompt)
            frame_results: list[dict[str, object]] = []
            object_frames: dict[str, list[int]] = {value: [] for value in prompts}
            object_scores: dict[str, list[float]] = {value: [] for value in prompts}
            for frame_index, (frame_id, timestamp, image) in enumerate(frames):
                raw = detector.detect(image)
                detections: list[dict[str, object]] = []
                for item in raw:
                    label = _normalise_label(str(item.get("phrase", "")))
                    if not label:
                        continue
                    box = [float(value) for value in item.get("box_xyxy", [])]
                    if len(box) != 4:
                        continue
                    detections.append({
                        "label": label,
                        "phrase": str(item.get("phrase", label)),
                        "score": float(item.get("score", 0.0)),
                        "box_xyxy": box,
                        "box_normalized": [
                            box[0] / max(1, image.width), box[1] / max(1, image.height),
                            box[2] / max(1, image.width), box[3] / max(1, image.height),
                        ],
                        "mask_available": False,
                    })
                # The showcase renders one representative box per target label.
                best_by_label: dict[str, dict[str, object]] = {}
                for detection in detections:
                    label = str(detection["label"])
                    if float(detection["score"]) > float(best_by_label.get(label, {}).get("score", -1.0)):
                        best_by_label[label] = detection
                detections = list(best_by_label.values())
                if detections:
                    try:
                        if self.segmenter is None:
                            self.segmenter = HuggingFaceSam2(device=self.device)
                        predictions = self.segmenter.segment(
                            image, [item["box_xyxy"] for item in detections]
                        )
                        for detection, (mask, mask_score) in zip(detections, predictions, strict=True):
                            detection["mask_available"] = True
                            detection["mask_score"] = float(mask_score)
                            detection["mask_area_fraction"] = float(mask.mean())
                            _tighten_box_from_mask(detection, mask)
                    except Exception as error:
                        segmentation_limitation = (
                            f"Segmentation was unavailable: {type(error).__name__}: {error}"
                        )
                for detection in detections:
                    label = str(detection["label"])
                    for prompt_value in prompts:
                        if prompt_value in label or label in prompt_value:
                            object_frames[prompt_value].append(frame_index)
                            object_scores[prompt_value].append(float(detection["score"]))
                frame_results.append({
                    "frame_id": frame_id,
                    "timestamp_s": float(timestamp),
                    "detections": detections,
                })
            _assign_tracks(frame_results)
            objects = [
                {
                    "label": label,
                    "frames_visible": len(set(object_frames[label])),
                    "frame_count": len(frames),
                    "max_score": max(object_scores[label], default=0.0),
                    "status": _status(len(frames), len(set(object_frames[label]))),
                }
                for label in prompts
            ]
            return {
                "status": "detected",
                "detector": getattr(detector, "provenance", {}),
                "frames": frame_results,
                "objects": objects,
                "limitations": [
                    "Boxes are frozen-model predictions, not ground-truth annotations.",
                    "A missing detection does not prove that an object was absent.",
                    *([segmentation_limitation] if segmentation_limitation else []),
                ],
            }
        except Exception as error:  # model downloads and local runtimes vary by machine
            return {
                "status": "unavailable",
                "frames": [],
                "objects": [],
                "limitations": [f"Object inspection was unavailable: {type(error).__name__}: {error}"],
            }

    def close(self) -> None:
        for model in (self.detector, self.segmenter):
            close = getattr(model, "close", None)
            if callable(close):
                close()
        self.detector = None
        self.detector_prompt = None
        self.segmenter = None
