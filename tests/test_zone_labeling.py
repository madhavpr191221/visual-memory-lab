"""Tests for deterministic VLM-assisted zone curation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

from visual_memory_lab.seven_scenes import prepare_office_dataset
from visual_memory_lab.zone_labeling import (
    DiscoveryBatch,
    FrameObservation,
    FrameAssignment,
    VerificationBatch,
    ZoneDefinition,
    ZoneOntology,
    ZonePrompts,
    label_zones,
)

from tests.test_seven_scenes import make_office_fixture


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, *, model: str, input: list[dict[str, object]], text_format: type, store: bool) -> object:
        self.calls += 1
        prompt = input[0]["content"][0]["text"]
        ids = re.findall(r"office:seq-\d{2}:\d{6}", prompt)
        if text_format is DiscoveryBatch:
            parsed = DiscoveryBatch(
                frames=[
                    FrameObservation(
                        observation_id=observation_id,
                        suggested_zone="monitor desk",
                        stable_landmarks=["monitors"],
                        confidence="high",
                    )
                    for observation_id in ids
                ]
            )
        elif text_format is ZoneOntology:
            parsed = ZoneOntology(
                zones=[
                    ZoneDefinition(
                        slug=f"zone-{index}",
                        name=f"Zone {index}",
                        description="An office area",
                        stable_landmarks=["desk"],
                        prompts=ZonePrompts(
                            name=f"zone {index}",
                            landmarks="an office desk",
                            technician_question="Where is this office desk?",
                        ),
                    )
                    for index in range(5)
                ]
            )
        else:
            parsed = VerificationBatch(
                assignments=[
                    FrameAssignment(
                        observation_id=observation_id,
                        zone_slug="zone-0",
                        visible_landmarks=["desk"],
                        confidence="high",
                    )
                    for observation_id in ids
                ]
            )
        return SimpleNamespace(output_parsed=parsed, model=model)


def test_label_zones_uses_cache_and_freezes_assignments(tmp_path: Path) -> None:
    office = make_office_fixture(tmp_path, frames=2)
    prepared = tmp_path / "prepared"
    prepare_office_dataset(dataset_root=office, output=prepared, expected_frames=2)
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    cache = tmp_path / "cache"

    first = label_zones(
        source=prepared / "train",
        output=tmp_path / "zones-1.json",
        cache_dir=cache,
        client=client,
    )
    call_count = responses.calls
    second = label_zones(
        source=prepared / "train",
        output=tmp_path / "zones-2.json",
        cache_dir=cache,
        client=client,
    )

    assert call_count == 3
    assert responses.calls == call_count
    assert first == second
    assert len(first["assignments"]) == 12
    assert set(first["assignments"].values()) == {"zone-0"}
    assert json.loads((tmp_path / "zones-1.json").read_text()) == first
