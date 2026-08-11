"""Tests for the application-facing memory and evidence service."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from visual_memory_lab.memory_store import StoredMemory
from visual_memory_lab.ui_service import (
    EvaluationCatalog,
    OfficeMemoryService,
    ZoneCatalog,
    failure_tags,
)


class FakeStore:
    model_id = "test/clip"
    model_revision = "1"

    def __init__(self, records: list[dict[str, object]], image: Path) -> None:
        self._records = records
        self._record_by_id = {str(row["observation_id"]): row for row in records}
        self._image = image

    @property
    def count(self) -> int:
        return len(self._records)

    def search(self, vector: np.ndarray, *, top_k: int) -> list[StoredMemory]:
        return [
            StoredMemory(rank=index + 1, score=1.0 - index / 100, record=record)
            for index, record in enumerate(self._records[:top_k])
        ]

    def get(self, observation_id: str) -> dict[str, object]:
        if observation_id not in self._record_by_id:
            raise KeyError(observation_id)
        return self._record_by_id[observation_id]

    def embedding(self, observation_id: str) -> np.ndarray:
        self.get(observation_id)
        return np.array([1.0, 0.0], dtype=np.float32)

    def image_path(self, observation_id: str) -> Path:
        self.get(observation_id)
        return self._image

    def records(self) -> list[dict[str, object]]:
        return self._records


class FakeEncoder:
    model_id = "test/clip"
    model_revision = "1"

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    def encode_pil_images(self, images: list[Image.Image]) -> np.ndarray:
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(images), 1))


def make_service(tmp_path: Path, zone_sequence: list[str] | None = None) -> OfficeMemoryService:
    image = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), "navy").save(image)
    slugs = zone_sequence or ["zone-a"] * 7 + ["zone-b"] * 3
    records = [
        {
            "observation_id": f"memory:{index}",
            "sequence_id": "seq-01",
            "step": index,
            "image_path": "frame.png",
        }
        for index in range(10)
    ]
    query_record = {
        "observation_id": "query:0",
        "sequence_id": "seq-02",
        "step": 0,
        "image_path": "frame.png",
    }
    query_row = {
        "query_id": "query:0",
        "sequence_id": "seq-02",
        "step": 0,
        "strict_eligible": True,
        "strict_hit_at_1": False,
        "strict_hit_at_5": True,
        "strict_hit_at_10": True,
        "relaxed_hit_at_1": True,
        "top1_translation_error_m": 0.6,
        "top1_rotation_error_deg": 35.0,
        "retrievals": [
            {
                "rank": 1,
                "observation_id": "memory:0",
                "score": 0.9,
                "translation_error_m": 0.6,
                "rotation_error_deg": 35.0,
            }
        ],
    }
    zones = ZoneCatalog(
        zones={
            "zone-a": {
                "slug": "zone-a",
                "name": "Window desk",
                "description": "Desk beside a window",
                "stable_landmarks": ["window"],
                "prompts": {},
            },
            "zone-b": {
                "slug": "zone-b",
                "name": "Bookshelf aisle",
                "description": "Aisle beside a bookshelf",
                "stable_landmarks": ["bookshelf"],
                "prompts": {},
            },
        },
        assignments={f"memory:{index}": slug for index, slug in enumerate(slugs)},
        counts={"zone-a": slugs.count("zone-a"), "zone-b": slugs.count("zone-b")},
    )
    evaluation = EvaluationCatalog(
        metrics={"pose": {"query_count": 1}},
        queries=[query_row],
        query_by_id={"query:0": query_row},
    )
    return OfficeMemoryService(
        memory=FakeStore(records, image),
        queries=FakeStore([query_record], image),
        encoder=FakeEncoder(),
        zones=zones,
        evaluation=evaluation,
    )


def test_search_reports_strong_zone_agreement_and_no_calendar_time(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    payload = service.search_text("desk beside a window", display_k=5)

    assert payload["likely_area"] == {
        "slug": "zone-a",
        "name": "Window desk",
        "support_count": 7,
        "evidence_count": 10,
        "strength": "strong",
    }
    assert payload["temporal"]["captured_at"] is None
    assert len(payload["evidence"]) == 10
    assert "image_path" not in payload["evidence"][0]


def test_zone_agreement_is_mixed_on_tie_and_validates_search(tmp_path: Path) -> None:
    service = make_service(tmp_path, ["zone-a", "zone-b"] * 5)

    payload = service.search_text("office", display_k=3)

    assert payload["likely_area"]["slug"] == "zone-a"
    assert payload["likely_area"]["strength"] == "mixed"
    try:
        service.search_text(" ", display_k=5)
    except ValueError as error:
        assert "must not be empty" in str(error)
    else:
        raise AssertionError("blank question should fail")


def test_failure_tags_and_query_page(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    row = service.evaluation.queries[0]

    assert failure_tags(row) == [
        "strict-rescued-at5",
        "relaxed-only-top1",
        "large-translation",
        "large-rotation",
    ]
    page = service.query_page(offset=0, limit=10, sequence=None, tag="large-rotation")
    assert page["total"] == 1
    assert page["items"][0]["query_id"] == "query:0"

