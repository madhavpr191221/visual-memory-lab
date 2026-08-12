"""Read-only payload for Phase 6.1.3 candidate association review."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@dataclass(frozen=True)
class AssociationShowcase:
    payload: dict[str, object]

    @classmethod
    def load(cls, *, associations: Path, localization: Path, audit: Path | None = None) -> "AssociationShowcase":
        root = associations.resolve()
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        frames = {str(item["frame_id"]): item for item in _jsonl(localization / "frames.jsonl")}
        detections = {str(item["detection_id"]): item for item in _jsonl(localization / "detections.jsonl")}
        judgments = {str(item["pair_id"]): item for item in _jsonl(audit / "judgments.jsonl")} if audit is not None and (audit / "judgments.jsonl").is_file() else {}
        pairs = []
        for pair in _jsonl(root / "associations.jsonl"):
            enriched = dict(pair)
            enriched["vlm_audit"] = judgments.get(str(pair["pair_id"]))
            for side in ("earlier", "later"):
                detection_id = str(pair[f"{'earlier' if side == 'earlier' else 'later'}_detection_id"])
                detection = detections[detection_id]
                frame = frames[str(detection["frame_id"])]
                enriched[side] = {
                    **detection,
                    "observation": frame["observation"],
                    "message_index": frame["message_index"],
                    "image_url": f"/api/phase6b1/images/{str(frame['frame_id']).replace(':', '-')}-raw",
                    "mask_url": f"/api/phase6b1/images/{detection_id.replace(':', '-')}-mask",
                }
            pairs.append(enriched)
        return cls(payload={"phase": "6.1.3", "claim_boundary": run["claim_boundary"], "metrics": run, "pairs": pairs, "classes": list(run["classes"])})
