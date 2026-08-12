"""Optional VLM pseudo-audit for top cross-visit candidate pairs."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from PIL import Image


class PairJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pair_id: str
    verdict: Literal["same", "different", "uncertain"]
    explanation: str


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _crop_data(frame: dict[str, object], detection: dict[str, object], root: Path) -> str:
    with Image.open(root / str(frame["image_path"])) as image:
        image = image.convert("RGB")
        x1, y1, x2, y2 = [float(value) for value in detection["box_xyxy"]]
        crop = image.crop((max(0, int(x1)), max(0, int(y1)), min(image.width, int(x2)), min(image.height, int(y2))))
        buffer = io.BytesIO()
        crop.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def audit_associations(*, associations: Path, localization: Path, output: Path, cache_dir: Path, model: str = "gpt-5.6-terra", limit: int = 200) -> dict[str, object]:
    load_dotenv()
    from openai import OpenAI

    root = localization.resolve()
    pairs = _jsonl(associations.resolve() / "associations.jsonl")[:limit]
    frames = {str(item["frame_id"]): item for item in _jsonl(root / "frames.jsonl")}
    detections = {str(item["detection_id"]): item for item in _jsonl(root / "detections.jsonl")}
    client = OpenAI()
    judgments: list[dict[str, object]] = []
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        cache = cache_dir / f"{pair_id.replace(':', '_').replace('__', '--')}.json"
        if cache.is_file():
            judgments.append(json.loads(cache.read_text(encoding="utf-8")))
            continue
        earlier_id, later_id = str(pair["earlier_detection_id"]), str(pair["later_detection_id"])
        prompt = ("Compare these two RGB crops from different logical office visits. "
                  "Decide only whether the visible object appears to be the same physical object, "
                  "a different object, or uncertain. Do not infer movement or calendar time. "
                  f"Predicted class: {pair['object_class']}. Pair ID: {pair_id}.")
        response = client.responses.parse(model=model, input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_text", "text": "Earlier crop:"},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{_crop_data(frames[str(detections[earlier_id]['frame_id'])], detections[earlier_id], root)}", "detail": "high"},
            {"type": "input_text", "text": "Later crop:"},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{_crop_data(frames[str(detections[later_id]['frame_id'])], detections[later_id], root)}", "detail": "high"},
        ]}], text_format=PairJudgment, store=False)
        judgment = response.output_parsed
        if not isinstance(judgment, PairJudgment) or judgment.pair_id != pair_id:
            raise ValueError(f"invalid VLM judgment for {pair_id}")
        payload = judgment.model_dump(mode="json")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        judgments.append(payload)
    summary = {"phase": "6.1.3", "pair_count": len(pairs), "reviewed_count": len(judgments), "model": model, "claim_boundary": "VLM verdicts are pseudo-labels for analysis, not ground truth."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "judgments.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in judgments), encoding="utf-8")
    return summary
