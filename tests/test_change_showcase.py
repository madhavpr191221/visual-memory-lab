"""Tests for the read-only Phase 6A showcase catalog."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from visual_memory_lab.change_showcase import ChangeShowcase


def make_change_showcase(tmp_path: Path) -> ChangeShowcase:
    audit = tmp_path / "audit"
    baseline = tmp_path / "baseline"
    review = tmp_path / "review"
    audit.mkdir(parents=True)
    baseline.mkdir()
    review.mkdir()
    observations = []
    for index in range(2):
        frame = audit / f"frame-{index}.jpg"
        sheet = audit / f"sheet-{index}.jpg"
        Image.new("RGB", (20, 12), "white").save(frame)
        Image.new("RGB", (20, 12), "gray").save(sheet)
        observations.append(
            {
                "observation_id": f"eth-office:{index}",
                "logical_order": index,
                "bag": {
                    "rgb_frames": [
                        {"message_index": index, "timestamp_ns": index, "path": str(frame)}
                    ],
                    "contact_sheet": str(sheet),
                    "vlm_contact_sheet": str(sheet),
                },
            }
        )
    (audit / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "ETH Office fixture",
                "logical_order_note": "Logical order only.",
                "observations": observations,
            }
        ),
        encoding="utf-8",
    )
    (baseline / "run.json").write_text(
        json.dumps(
            {
                "pair_count": 1,
                "candidate_count": 2,
                "voxel_size_m": 0.02,
                "primary_threshold_m": 0.05,
                "distance_thresholds_m": [0.02, 0.05, 0.1],
                "min_cluster_voxels": 20,
            }
        ),
        encoding="utf-8",
    )
    pair_dir = baseline / "0-to-1"
    pair_dir.mkdir()
    current = pair_dir / "current-only.png"
    earlier = pair_dir / "earlier-only.png"
    Image.new("RGB", (20, 12), "red").save(current)
    Image.new("RGB", (20, 12), "blue").save(earlier)
    pair = {
        "pair_id": "0-to-1",
        "earlier_observation": 0,
        "current_observation": 1,
        "consecutive": True,
        "current_only_candidate_count": 1,
        "earlier_only_candidate_count": 1,
        "changed_fraction": {"0.050": {"current_only": 0.1, "earlier_only": 0.2}},
        "point_to_point": {},
    }
    (baseline / "pairs.jsonl").write_text(json.dumps(pair) + "\n", encoding="utf-8")
    reviewed = {
        "candidate_id": "eth-office:0-to-1:current-only:cluster-000",
        "verdict": "supported",
        "interpretation": "current_only",
        "description": "A visible difference.",
        "confidence": "high",
        "evidence_ids": ["projection"],
        "limitations": [],
        "related_candidate_id": None,
    }
    (review / "reviews.json").write_text(
        json.dumps(
            {
                "summary": {
                    "claim_boundary": "Pseudo-reference only.",
                    "reviewed_candidate_count": 1,
                    "accepted_pseudo_reference_count": 1,
                    "verdict_counts": {"supported": 1, "uncertain": 0, "unsupported": 0},
                },
                "pairs": [
                    {
                        "pair_id": "0-to-1",
                        "candidates": [reviewed],
                        "overall_limitations": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return ChangeShowcase.load(audit=audit, baseline=baseline, review=review)


def test_change_showcase_has_no_local_paths_and_allowlists_images(tmp_path: Path) -> None:
    showcase = make_change_showcase(tmp_path)
    serialized = json.dumps(showcase.payload)
    assert "C:\\" not in serialized
    assert showcase.payload["metrics"]["pair_count"] == 1  # type: ignore[index]
    assert showcase.image_path("pair-0-to-1-current-only").is_file()
    try:
        showcase.image_path("../../secret")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown image ID should be rejected")
