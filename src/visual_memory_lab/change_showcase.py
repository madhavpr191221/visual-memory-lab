"""Read-only catalog for presenting Phase 6A artifacts in the local UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChangeShowcase:
    payload: dict[str, object]
    images: dict[str, Path]

    @classmethod
    def load(cls, *, audit: Path, baseline: Path, review: Path) -> "ChangeShowcase":
        manifest = json.loads((audit / "manifest.json").read_text(encoding="utf-8"))
        run = json.loads((baseline / "run.json").read_text(encoding="utf-8"))
        pairs = [
            json.loads(line)
            for line in (baseline / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        reviews_payload = json.loads((review / "reviews.json").read_text(encoding="utf-8"))
        review_by_pair = {str(item["pair_id"]): item for item in reviews_payload["pairs"]}
        images: dict[str, Path] = {}
        observations: list[dict[str, object]] = []
        for observation in manifest["observations"]:
            index = int(observation["logical_order"])
            bag = observation["bag"]
            frames = []
            for frame in bag["rgb_frames"]:
                image_id = f"observation-{index}-frame-{int(frame['message_index']):06d}"
                images[image_id] = Path(frame["path"]).resolve()
                frames.append(
                    {
                        "message_index": int(frame["message_index"]),
                        "timestamp_ns": int(frame["timestamp_ns"]),
                        "image_url": f"/api/phase6a/images/{image_id}",
                    }
                )
            sheet_id = f"observation-{index}-contact-sheet"
            vlm_sheet_id = f"observation-{index}-vlm-contact-sheet"
            images[sheet_id] = Path(bag["contact_sheet"]).resolve()
            images[vlm_sheet_id] = Path(bag["vlm_contact_sheet"]).resolve()
            observations.append(
                {
                    "observation_id": str(observation["observation_id"]),
                    "logical_order": index,
                    "frame_count": len(frames),
                    "frames": frames,
                    "contact_sheet_url": f"/api/phase6a/images/{sheet_id}",
                    "vlm_contact_sheet_url": f"/api/phase6a/images/{vlm_sheet_id}",
                }
            )

        pair_payloads: list[dict[str, object]] = []
        for pair in pairs:
            pair_id = str(pair["pair_id"])
            current_id = f"pair-{pair_id}-current-only"
            earlier_id = f"pair-{pair_id}-earlier-only"
            images[current_id] = (baseline / pair_id / "current-only.png").resolve()
            images[earlier_id] = (baseline / pair_id / "earlier-only.png").resolve()
            review_item = review_by_pair.get(pair_id, {})
            pair_payloads.append(
                {
                    "pair_id": pair_id,
                    "earlier_observation": int(pair["earlier_observation"]),
                    "current_observation": int(pair["current_observation"]),
                    "consecutive": bool(pair["consecutive"]),
                    "current_only_candidate_count": int(pair["current_only_candidate_count"]),
                    "earlier_only_candidate_count": int(pair["earlier_only_candidate_count"]),
                    "current_only_projection_url": f"/api/phase6a/images/{current_id}",
                    "earlier_only_projection_url": f"/api/phase6a/images/{earlier_id}",
                    "changed_fraction": pair["changed_fraction"],
                    "point_to_point": pair["point_to_point"],
                    "reviewed_candidates": review_item.get("candidates", []),
                    "review_limitations": review_item.get("overall_limitations", []),
                }
            )

        payload: dict[str, object] = {
            "dataset": manifest["dataset"],
            "logical_order_note": manifest["logical_order_note"],
            "claim_boundary": reviews_payload["summary"]["claim_boundary"],
            "metrics": {
                "observation_count": len(observations),
                "rgb_sample_count": sum(int(item["frame_count"]) for item in observations),
                "pair_count": int(run["pair_count"]),
                "geometric_candidate_count": int(run["candidate_count"]),
                "reviewed_candidate_count": int(reviews_payload["summary"]["reviewed_candidate_count"]),
                "accepted_pseudo_reference_count": int(
                    reviews_payload["summary"]["accepted_pseudo_reference_count"]
                ),
                "verdict_counts": reviews_payload["summary"]["verdict_counts"],
            },
            "method": {
                "voxel_size_m": float(run["voxel_size_m"]),
                "primary_threshold_m": float(run["primary_threshold_m"]),
                "distance_thresholds_m": run["distance_thresholds_m"],
                "min_cluster_voxels": int(run["min_cluster_voxels"]),
            },
            "observations": observations,
            "pairs": pair_payloads,
        }
        for image_id, path in images.items():
            if not path.is_file():
                raise FileNotFoundError(f"Phase 6A image is missing for {image_id}: {path}")
        return cls(payload=payload, images=images)

    def image_path(self, image_id: str) -> Path:
        try:
            return self.images[image_id]
        except KeyError as error:
            raise KeyError(f"unknown Phase 6A image: {image_id}") from error
