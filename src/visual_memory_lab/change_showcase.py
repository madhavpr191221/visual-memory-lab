"""Catalog and small presentation artifacts for the Phase 6A local UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from plyfile import PlyData

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


_CURATED_CASES: dict[str, dict[str, object]] = {
    "0-to-1": {
        "earlier_frame": 472,
        "current_frame": 53,
        "earlier_box": {"x": 0.57, "y": 0.38, "width": 0.42, "height": 0.61},
        "current_box": {"x": 0.32, "y": 0.10, "width": 0.38, "height": 0.88},
        "outcome": "object_moved",
        "outcome_label": "Object likely moved",
        "headline": "The black office chair moved from the desk edge to the window area",
        "confidence": "medium",
        "explanation": (
            "The earlier view records the chair tucked against the monitor desk. "
            "The later visit records the same chair type standing in front of the window."
        ),
        "limitation": (
            "The chair appearance and 3D shape agree, but this dataset does not provide a persistent object ID."
        ),
        "earlier_cluster": 0,
        "current_cluster": 0,
    },
    "1-to-2": {
        "earlier_frame": 53,
        "current_frame": 453,
        "earlier_box": {"x": 0.32, "y": 0.10, "width": 0.38, "height": 0.88},
        "current_box": {"x": 0.29, "y": 0.0, "width": 0.42, "height": 0.99},
        "outcome": "object_moved",
        "outcome_label": "Object likely moved",
        "headline": "The chair moved again, from the window area to the workstation",
        "confidence": "medium",
        "explanation": (
            "Visit 1 shows the chair centered in front of the window. Visit 2 shows a matching chair pulled "
            "up beside the workstation."
        ),
        "limitation": (
            "The visual and geometric evidence support a relocation, but cannot prove that two similar chairs were not exchanged."
        ),
        "earlier_cluster": 0,
        "current_cluster": 0,
    },
    "2-to-3": {
        "earlier_frame": 453,
        "current_frame": 1390,
        "earlier_box": {"x": 0.29, "y": 0.0, "width": 0.42, "height": 0.99},
        "current_box": {"x": 0.35, "y": 0.0, "width": 0.34, "height": 0.92},
        "outcome": "object_moved",
        "outcome_label": "Object likely moved",
        "headline": "The chair returned to the window-side desk",
        "confidence": "medium",
        "explanation": (
            "Visit 2 records the chair beside a workstation. Visit 3 records a matching chair positioned at "
            "the window-side desk."
        ),
        "limitation": (
            "This is a strong change-of-position example, but object identity remains an inference because several chairs may look alike."
        ),
        "earlier_cluster": 1,
        "current_cluster": 1,
    },
}


def _cluster_points(path: Path, cluster_index: int | None) -> np.ndarray:
    if cluster_index is None or not path.is_file():
        return np.empty((0, 3), dtype=np.float64)
    vertex = PlyData.read(path)["vertex"]
    names = set(vertex.data.dtype.names or ())
    if "cluster_index" not in names:
        return np.empty((0, 3), dtype=np.float64)
    selected = np.asarray(vertex["cluster_index"]) == cluster_index
    return np.column_stack([np.asarray(vertex[axis])[selected] for axis in ("x", "y", "z")])


def _render_cluster_crop(
    *,
    earlier_ply: Path,
    current_ply: Path,
    earlier_cluster: int | None,
    current_cluster: int | None,
    output: Path,
) -> bool:
    earlier = _cluster_points(earlier_ply, earlier_cluster)
    current = _cluster_points(current_ply, current_cluster)
    available = [points for points in (earlier, current) if len(points)]
    if not available:
        return False
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
    for axis, (a, b, labels) in zip(
        axes,
        ((0, 1, ("left / right", "front / back")), (0, 2, ("left / right", "height"))),
        strict=True,
    ):
        if len(earlier):
            axis.scatter(earlier[:, a], earlier[:, b], s=5, alpha=0.75, color="#c9773d", label="earlier")
        if len(current):
            axis.scatter(current[:, a], current[:, b], s=5, alpha=0.75, color="#2f688d", label="later")
        combined = np.concatenate(available, axis=0)
        low = combined[:, [a, b]].min(axis=0)
        high = combined[:, [a, b]].max(axis=0)
        margin = np.maximum((high - low) * 0.16, 0.12)
        axis.set_xlim(low[0] - margin[0], high[0] + margin[0])
        axis.set_ylim(low[1] - margin[1], high[1] + margin[1])
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        axis.grid(alpha=0.16)
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle("Coarse 3D difference region")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, facecolor="white")
    plt.close(figure)
    return True


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
        frame_paths: dict[tuple[int, int], Path] = {}
        frame_urls: dict[tuple[int, int], str] = {}
        for observation in manifest["observations"]:
            index = int(observation["logical_order"])
            bag = observation["bag"]
            frames = []
            for frame in bag["rgb_frames"]:
                image_id = f"observation-{index}-frame-{int(frame['message_index']):06d}"
                images[image_id] = Path(frame["path"]).resolve()
                frame_paths[(index, int(frame["message_index"]))] = images[image_id]
                frame_urls[(index, int(frame["message_index"]))] = f"/api/phase6a/images/{image_id}"
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
        cases: list[dict[str, object]] = []
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
            spec = _CURATED_CASES.get(pair_id)
            if bool(pair["consecutive"]) and spec is not None:
                earlier_observation = int(pair["earlier_observation"])
                current_observation = int(pair["current_observation"])
                earlier_key = (earlier_observation, int(spec["earlier_frame"]))
                current_key = (current_observation, int(spec["current_frame"]))
                if earlier_key in frame_paths and current_key in frame_paths:
                    crop_id = f"pair-{pair_id}-focus-3d"
                    crop_path = (baseline / "showcase-cache" / f"{pair_id}.png").resolve()
                    rendered = _render_cluster_crop(
                        earlier_ply=(baseline / pair_id / "earlier-only.ply").resolve(),
                        current_ply=(baseline / pair_id / "current-only.ply").resolve(),
                        earlier_cluster=spec["earlier_cluster"],  # type: ignore[arg-type]
                        current_cluster=spec["current_cluster"],  # type: ignore[arg-type]
                        output=crop_path,
                    )
                    geometry_url = f"/api/phase6a/images/{crop_id}" if rendered else f"/api/phase6a/images/{current_id}"
                    images[crop_id] = crop_path if rendered else images[current_id]
                    cases.append(
                        {
                            "pair_id": pair_id,
                            "earlier_observation": earlier_observation,
                            "current_observation": current_observation,
                            "earlier_image_url": frame_urls[earlier_key],
                            "current_image_url": frame_urls[current_key],
                            "earlier_frame": int(spec["earlier_frame"]),
                            "current_frame": int(spec["current_frame"]),
                            "earlier_box": spec["earlier_box"],
                            "current_box": spec["current_box"],
                            "outcome": spec["outcome"],
                            "outcome_label": spec["outcome_label"],
                            "headline": spec["headline"],
                            "confidence": spec["confidence"],
                            "explanation": spec["explanation"],
                            "limitation": spec["limitation"],
                            "geometry_url": geometry_url,
                            "geometry_note": (
                                "This is the strongest nearby coarse difference region, not an object segmentation. "
                                "It supports where geometry changed but cannot prove object identity."
                            ),
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
            "cases": cases,
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
