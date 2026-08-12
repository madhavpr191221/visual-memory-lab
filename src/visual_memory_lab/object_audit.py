"""Cached VLM pseudo-audit for Phase 6B1 localization predictions."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

PROMPT_VERSION = "phase6b1-object-audit-v1"
DEFAULT_MODEL = "gpt-5.6-terra"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DetectionJudgment(StrictModel):
    detection_id: str
    verdict: Literal["supported", "uncertain", "unsupported"]
    category_correct: Literal["yes", "no", "uncertain"]
    mask_quality: Literal["good", "partial", "excessive", "uncertain"]
    explanation: str


class FrameAudit(StrictModel):
    frame_id: str
    detections: list[DetectionJudgment]
    missed_visible_classes: list[Literal["chair", "waste_bin", "box"]]
    overall_limitations: list[str]


def _records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _image_part(path: Path) -> tuple[dict[str, str], str]:
    data = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return (
        {
            "type": "input_image",
            "image_url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}",
            "detail": "high",
        },
        hashlib.sha256(data).hexdigest(),
    )


def _cache_path(
    *, cache_dir: Path, model: str, prompt: str, image_hashes: list[str]
) -> Path:
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "schema": FrameAudit.model_json_schema(),
            "prompt": prompt,
            "image_hashes": image_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return cache_dir / f"{hashlib.sha256(payload).hexdigest()}.json"


def _validate_audit(
    audit: FrameAudit, *, frame_id: str, detection_ids: list[str]
) -> None:
    if audit.frame_id != frame_id:
        raise ValueError(f"VLM returned frame {audit.frame_id!r}; expected {frame_id!r}")
    actual = [item.detection_id for item in audit.detections]
    if len(actual) != len(set(actual)) or set(actual) != set(detection_ids):
        raise ValueError("VLM audit must review every supplied detection exactly once")
    if len(audit.missed_visible_classes) != len(set(audit.missed_visible_classes)):
        raise ValueError("missed visible classes must be unique")


def _call_audit(
    *,
    client: object,
    model: str,
    frame: dict[str, object],
    detections: list[dict[str, object]],
    raw_path: Path,
    overlay_path: Path,
    cache_dir: Path,
) -> tuple[FrameAudit, str, bool]:
    frame_id = str(frame["frame_id"])
    detection_lines = "\n".join(
        f"- {item['detection_id']}: predicted={item['canonical_class']}, "
        f"phrase={item['phrase']!r}, score={float(item['score']):.3f}"
        for item in detections
    ) or "- No detections were produced."
    prompt = (
        "You are auditing frozen-model predictions on a public ETH Office RGB frame. "
        "The first image is raw. The second contains model-generated boxes and translucent masks. "
        "Review every listed detection exactly once. supported means the predicted object and mask "
        "are visibly plausible; unsupported means visibly wrong; uncertain covers occlusion, blur, "
        "or insufficient evidence. Report target classes that are clearly visible in the raw image "
        "but missed entirely. Target classes are chair, waste_bin, and box. Do not infer object "
        "identity, movement, absence outside the view, human activity, or ground truth. Keep each "
        "explanation short.\n\n"
        f"Frame ID: {frame_id}\nDetections:\n{detection_lines}"
    )
    raw_part, raw_hash = _image_part(raw_path)
    overlay_part, overlay_hash = _image_part(overlay_path)
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_text", "text": "Raw RGB frame:"},
        raw_part,
        {"type": "input_text", "text": "Prediction overlay:"},
        overlay_part,
    ]
    cache_path = _cache_path(
        cache_dir=cache_dir,
        model=model,
        prompt=prompt,
        image_hashes=[raw_hash, overlay_hash],
    )
    detection_ids = [str(item["detection_id"]) for item in detections]
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        audit = FrameAudit.model_validate(cached["parsed"])
        _validate_audit(audit, frame_id=frame_id, detection_ids=detection_ids)
        return audit, str(cached["response_model"]), True

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = getattr(client, "responses").parse(
                model=model,
                input=[{"role": "user", "content": content}],
                text_format=FrameAudit,
                store=False,
            )
            audit = response.output_parsed
            if not isinstance(audit, FrameAudit):
                raise ValueError("VLM response did not match the audit schema")
            _validate_audit(audit, frame_id=frame_id, detection_ids=detection_ids)
            response_model = str(getattr(response, "model", model))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "prompt_version": PROMPT_VERSION,
                        "model_requested": model,
                        "response_model": response_model,
                        "image_hashes": [raw_hash, overlay_hash],
                        "parsed": audit.model_dump(mode="json"),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return audit, response_model, False
        except Exception as error:  # SDK and transport exception types vary.
            last_error = error
            if attempt < 2:
                time.sleep(2**attempt)
    assert last_error is not None
    raise RuntimeError(f"VLM object audit failed after three attempts: {last_error}") from last_error


def _sample_frames(
    frames: list[dict[str, object]], per_observation: int
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for observation in sorted({int(item["observation"]) for item in frames}):
        group = [item for item in frames if int(item["observation"]) == observation]
        actual = min(per_observation, len(group))
        positions = np.rint(np.linspace(0, len(group) - 1, actual)).astype(int)
        selected.extend(group[int(index)] for index in positions)
    return selected


def audit_eth_object_localization(
    *,
    localization: Path,
    output: Path,
    cache_dir: Path,
    frames_per_observation: int = 12,
    model: str = DEFAULT_MODEL,
    client: object | None = None,
) -> dict[str, object]:
    """Audit a fixed, detector-independent frame sample with a VLM."""

    if frames_per_observation < 1:
        raise ValueError("frames per observation must be positive")
    root = localization.resolve()
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    frames = _records(root / "frames.jsonl")
    detections = _records(root / "detections.jsonl")
    detection_by_frame: dict[str, list[dict[str, object]]] = {}
    detection_by_id: dict[str, dict[str, object]] = {}
    for item in detections:
        detection_by_frame.setdefault(str(item["frame_id"]), []).append(item)
        detection_by_id[str(item["detection_id"])] = item
    selected = _sample_frames(frames, frames_per_observation)
    resolved = output.resolve()
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise FileExistsError(f"output path is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    load_dotenv()
    if client is None:
        from openai import OpenAI

        client = OpenAI()

    reviews: list[dict[str, object]] = []
    response_models: set[str] = set()
    cache_hits = 0
    for frame in selected:
        frame_detections = detection_by_frame.get(str(frame["frame_id"]), [])
        audit, response_model, cached = _call_audit(
            client=client,
            model=model,
            frame=frame,
            detections=frame_detections,
            raw_path=root / str(frame["image_path"]),
            overlay_path=root / str(frame["overlay_path"]),
            cache_dir=cache_dir.resolve(),
        )
        response_models.add(response_model)
        cache_hits += int(cached)
        reviews.append(
            {
                **audit.model_dump(mode="json"),
                "observation": int(frame["observation"]),
                "message_index": int(frame["message_index"]),
                "cached": cached,
            }
        )

    verdicts: Counter[str] = Counter()
    mask_quality: Counter[str] = Counter()
    class_verdicts: dict[str, Counter[str]] = {}
    high_confidence: Counter[str] = Counter()
    missed: Counter[str] = Counter()
    for review in reviews:
        for class_name in review["missed_visible_classes"]:
            missed[str(class_name)] += 1
        for judgment in review["detections"]:
            detection = detection_by_id[str(judgment["detection_id"])]
            verdict = str(judgment["verdict"])
            class_name = str(detection["canonical_class"])
            verdicts[verdict] += 1
            mask_quality[str(judgment["mask_quality"])] += 1
            class_verdicts.setdefault(class_name, Counter())[verdict] += 1
            if float(detection["score"]) >= 0.35:
                high_confidence[verdict] += 1
    decided_high = high_confidence["supported"] + high_confidence["unsupported"]
    summary = {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "source_schema_version": run.get("schema_version"),
        "model_requested": model,
        "response_models": sorted(response_models),
        "frame_count": len(reviews),
        "frames_per_observation": frames_per_observation,
        "reviewed_detection_count": sum(verdicts.values()),
        "verdict_counts": dict(sorted(verdicts.items())),
        "mask_quality_counts": dict(sorted(mask_quality.items())),
        "missed_visible_class_counts": dict(sorted(missed.items())),
        "class_verdict_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(class_verdicts.items())
        },
        "high_confidence_verdict_counts": dict(sorted(high_confidence.items())),
        "high_confidence_pseudo_support_rate": (
            high_confidence["supported"] / decided_high if decided_high else None
        ),
        "cache_hits": cache_hits,
        "claim_boundary": (
            "These are VLM pseudo-audit judgments, not human annotations or ground-truth accuracy."
        ),
    }
    (resolved / "frame-audits.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in reviews),
        encoding="utf-8",
    )
    (resolved / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
