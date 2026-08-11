"""Tests for the strict ETH change pseudo-reference review."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from visual_memory_lab.change_review import CandidateReview, PairReview, review_eth_changes


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, **kwargs: object) -> object:
        self.calls += 1
        input_value = kwargs["input"]
        prompt = input_value[0]["content"][0]["text"]  # type: ignore[index]
        pair_id = re.search(r"Pair: ([^\n]+)", prompt).group(1)  # type: ignore[union-attr]
        candidate_ids = re.findall(r"^- (eth-office:[^:]+:[^:]+:cluster-\d+):", prompt, re.MULTILINE)
        evidence_ids = re.search(r"Evidence IDs: ([^\n]+)", prompt).group(1).split(", ")  # type: ignore[union-attr]
        parsed = PairReview(
            pair_id=pair_id,
            candidates=[
                CandidateReview(
                    candidate_id=candidate_id,
                    verdict="supported",
                    interpretation="current_only",
                    description="A visible geometric difference.",
                    confidence="medium",
                    evidence_ids=[evidence_ids[2]],
                    limitations=["This is not human ground truth."],
                    related_candidate_id=None,
                )
                for candidate_id in candidate_ids
            ],
            overall_limitations=["Public evidence only."],
        )
        return SimpleNamespace(output_parsed=parsed, model="test-model")


def test_review_is_cached_and_writes_pseudo_reference(tmp_path: Path) -> None:
    audit = tmp_path / "audit"
    baseline = tmp_path / "baseline"
    audit.mkdir()
    baseline.mkdir()
    observations = []
    for index in range(2):
        image_path = audit / f"contact-{index}.jpg"
        Image.new("RGB", (40, 30), "white").save(image_path)
        observations.append(
            {"logical_order": index, "bag": {"vlm_contact_sheet": str(image_path)}}
        )
    (audit / "manifest.json").write_text(json.dumps({"observations": observations}), encoding="utf-8")
    current_png = baseline / "current.png"
    earlier_png = baseline / "earlier.png"
    Image.new("RGB", (40, 30), "red").save(current_png)
    Image.new("RGB", (40, 30), "blue").save(earlier_png)
    pair = {
        "pair_id": "0-to-1",
        "earlier_observation": 0,
        "current_observation": 1,
        "evidence": {
            "current_only_png": str(current_png),
            "earlier_only_png": str(earlier_png),
        },
    }
    (baseline / "pairs.jsonl").write_text(json.dumps(pair) + "\n", encoding="utf-8")
    candidate = {
        "candidate_id": "eth-office:0-to-1:current-only:cluster-000",
        "pair_id": "0-to-1",
        "direction": "current-only",
        "centroid_m": [0, 0, 0],
        "voxel_count": 25,
    }
    (baseline / "candidates.jsonl").write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    cache = tmp_path / "cache"
    first = review_eth_changes(
        baseline=baseline,
        audit=audit,
        output=tmp_path / "review-1",
        cache_dir=cache,
        model="test-model",
        client=client,
    )
    second = review_eth_changes(
        baseline=baseline,
        audit=audit,
        output=tmp_path / "review-2",
        cache_dir=cache,
        model="test-model",
        client=client,
    )
    assert first["accepted_pseudo_reference_count"] == 1
    assert second["accepted_pseudo_reference_count"] == 1
    assert responses.calls == 1
